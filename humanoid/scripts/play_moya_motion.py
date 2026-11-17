"""Replay Moya 47-column AMP data in MuJoCo without Isaac Gym or an ONNX policy.

This is kinematic playback: write each saved pose and call mj_forward(), without
physics integration or interpolation. Timing comes from the TXT FrameDuration.
"""

import argparse
import json
import math
import threading
import time
from contextlib import ExitStack
from pathlib import Path

import mujoco
import numpy as np

if __package__:
    from .sim2sim_moya import Config, ROOT, load_model
else:
    from sim2sim_moya import Config, ROOT, load_model


class AmpMotion:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open(encoding="utf-8") as stream:
            document = json.load(stream)
        self.frames = np.asarray(document["Frames"], dtype=np.float64)
        self.dt = float(document["FrameDuration"])
        if self.frames.ndim != 2 or self.frames.shape[1] != 47 or not len(self.frames):
            raise ValueError(f"Expected nonempty AMP Frames [T, 47], got {self.frames.shape}")
        if not np.all(np.isfinite(self.frames)):
            raise ValueError("Motion contains NaN or infinity")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("FrameDuration must be finite and positive")
        norms = np.linalg.norm(self.frames[:, 3:7], axis=1, keepdims=True)
        if np.any(norms < 1e-8):
            raise ValueError("Motion contains a near-zero root quaternion")
        self.frames[:, 3:7] /= norms
        self.duration = (len(self.frames) - 1) * self.dt


class MotionPlayer:
    def __init__(self, model, motion, *, speed=1.0, paused=False, in_place=False):
        self.model = model
        self.data = mujoco.MjData(model)
        self.motion = motion
        self.in_place = in_place
        self.speed = speed
        self.paused = paused
        self.lock = threading.Lock()
        self.frame_index = 0
        self.requested_frame = None
        ids = np.array([model.joint(name).id for name in Config.joint_names])
        self.q_indices = model.jnt_qposadr[ids]
        self.dq_indices = model.jnt_dofadr[ids]
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1 or model.nv != 6 + Config.num_actions:
            raise ValueError("Expected one floating base and 14 Moya joints")
        self.base_q = int(model.jnt_qposadr[free[0]])
        self.base_v = int(model.jnt_dofadr[free[0]])
        self.base_body = int(model.jnt_bodyid[free[0]])

    def set_frame(self, index):
        if not 0 <= index < len(self.motion.frames):
            raise IndexError(f"Frame {index} outside [0, {len(self.motion.frames) - 1}]")
        frame = self.motion.frames[index]
        root_pos = frame[:3].copy()
        if self.in_place:
            root_pos[:2] = 0
        # AMP is xyzw; MuJoCo's free-joint qpos is wxyz.
        root_quat = frame[[6, 3, 4, 5]]
        self.data.qpos[self.base_q:self.base_q + 3] = root_pos
        self.data.qpos[self.base_q + 3:self.base_q + 7] = root_quat
        self.data.qpos[self.q_indices] = frame[7:21]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, root_quat)
        # Free-joint linear velocity is world-frame; angular velocity is local.
        velocity_world = rotation.reshape(3, 3) @ frame[27:30]
        if self.in_place:
            velocity_world[:2] = 0
        self.data.qvel[self.base_v:self.base_v + 3] = velocity_world
        self.data.qvel[self.base_v + 3:self.base_v + 6] = frame[30:33]
        self.data.qvel[self.dq_indices] = frame[33:47]
        self.data.time = index * self.motion.dt
        # Do not mj_step(): gravity, PD and contacts must not alter the poses.
        mujoco.mj_forward(self.model, self.data)
        if not np.all(np.isfinite(self.data.xpos)):
            raise RuntimeError(f"Non-finite body position at frame {index}")
        with self.lock:
            self.frame_index = index

    def key_callback(self, key):
        with self.lock:
            if key == ord(" "):
                self.paused = not self.paused
            elif key == ord("R"):
                self.requested_frame = 0
            elif key in (263, 262):  # GLFW left/right arrow keys.
                origin = self.frame_index if self.requested_frame is None else self.requested_frame
                self.requested_frame = int(np.clip(origin + (-1 if key == 263 else 1),
                                                   0, len(self.motion.frames) - 1))
                self.paused = True
            elif key in (ord("="), ord("+"), ord("-")):
                self.speed = float(np.clip(self.speed * (0.5 if key == ord("-") else 2), 0.125, 8))
            else:
                return
            index = self.frame_index if self.requested_frame is None else self.requested_frame
            print(f"frame={index}/{len(self.motion.frames) - 1}; "
                  f"time={index * self.motion.dt:.3f}s; speed={self.speed:g}x; paused={self.paused}")

    def controls(self):
        with self.lock:
            result = self.paused, self.speed, self.requested_frame
            self.requested_frame = None
            return result


def replay(player, *, viewer=None, loop=False, duration=None, realtime=True):
    """Visit saved frames in order; duration limits source-motion seconds."""
    displayed = 0
    advanced_time = 0.0
    previous_controls = None
    deadline = time.perf_counter()
    while viewer is None or viewer.is_running():
        paused, speed, requested = player.controls()
        now = time.perf_counter()
        if previous_controls != (paused, speed):
            deadline = now + player.motion.dt / speed
            previous_controls = (paused, speed)
        next_frame = requested
        if next_frame is None and not paused and (not realtime or now >= deadline):
            if duration is not None and advanced_time + player.motion.dt > duration + 1e-9:
                break
            next_frame = player.frame_index + 1
            if next_frame >= len(player.motion.frames):
                if not loop:
                    break
                next_frame = 0
            advanced_time += player.motion.dt
        if next_frame is not None:
            if viewer is None:
                player.set_frame(next_frame)
            else:
                with viewer.lock():
                    player.set_frame(next_frame)
            displayed += 1
            deadline = max(deadline + player.motion.dt / speed, now)
        if viewer is not None:
            viewer.sync()
        if realtime or paused:
            # Remain responsive to keyboard/viewer events while paused or waiting.
            time.sleep(0.01 if paused else min(0.01, max(0, deadline - time.perf_counter())))
    return displayed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", "--input", type=Path,
                        default=ROOT / "humanoid/envs/datasets/moya/017.txt",
                        help="47-column AMP JSON/TXT motion (default: moya/017.txt)")
    parser.add_argument("--xml", type=Path,
                        default=ROOT / "resources/robots/MoyaF4.1/MoyaF4.1_scene_lower.xml")
    parser.add_argument("--training-urdf", type=Path, default=Config.training_asset,
                        help="Training URDF supplying the fixed-arm posture and contacts")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--duration", type=float, help="Limit replayed source-motion seconds")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--paused", action="store_true", help="Start paused; arrows move one frame")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--loop", dest="loop", action="store_true", help="Restart from frame 0 after the last frame")
    modes.add_argument("--once", dest="loop", action="store_false", help="Play once (default)")
    parser.set_defaults(loop=False)
    parser.add_argument("--in-place", action="store_true", help="Fix root XY position at the origin")
    parser.add_argument("--show-collisions", action="store_true", help="Show pelvis/foot contact geometry")
    parser.add_argument("--headless", action="store_true", help="Validate playback without opening a window")
    parser.add_argument("--no-realtime", action="store_true", help="Process frames as fast as possible")
    args = parser.parse_args()
    for path in (args.motion, args.xml, args.training_urdf):
        if not path.is_file():
            parser.error(f"File does not exist: {path}")
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive")
    if args.start_frame < 0:
        parser.error("--start-frame must be nonnegative")
    if args.headless and args.paused:
        parser.error("--paused requires a viewer")
    return args


def main():
    args = parse_args()
    motion = AmpMotion(args.motion)
    if args.start_frame >= len(motion.frames):
        raise SystemExit(f"--start-frame must be less than {len(motion.frames)}")
    player = MotionPlayer(load_model(args.xml, args.training_urdf), motion,
                          speed=args.speed, paused=args.paused, in_place=args.in_place)
    player.set_frame(args.start_frame)
    print(f"Motion: {motion.path}\nFrames: {len(motion.frames)} x 47; "
          f"FPS: {1 / motion.dt:g}; duration: {motion.duration:.3f}s; speed: {args.speed:g}x")
    print("Kinematic playback: saved poses only, no physics integration or interpolation.")
    print("Space: pause/resume; Left/Right: one frame and pause; R: first frame; +/-: speed")
    with ExitStack() as stack:
        viewer = None
        if not args.headless:
            import mujoco.viewer

            viewer = stack.enter_context(mujoco.viewer.launch_passive(
                player.model, player.data, key_callback=player.key_callback))
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = player.base_body
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -20
                viewer.opt.geomgroup[3] = int(args.show_collisions)
            viewer.sync()
        try:
            displayed = replay(player, viewer=viewer, loop=args.loop, duration=args.duration,
                               realtime=not args.no_realtime)
        except KeyboardInterrupt:
            displayed = None
        count = "interrupted" if displayed is None else f"{displayed + 1} frames displayed"
        print(f"Finished: {count}; frame={player.frame_index}; motion time={player.data.time:.3f}s")


if __name__ == "__main__":
    main()
