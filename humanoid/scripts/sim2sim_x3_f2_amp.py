#!/usr/bin/env python3
"""Run the X3 F2 AMP policy in MuJoCo with joystick commands.

The observation layout, controller gains, limits, and timing mirror
``humanoid/envs/x3_f2/x3_f2_cfg_amp.py``.

Default Xbox-style mapping (axis/button ids can be changed from the CLI):
  left stick vertical    -> forward/backward velocity
  left stick horizontal  -> lateral velocity
  right stick horizontal -> yaw rate
  A                       -> zero command while held
  B                       -> reset simulation
  Start                   -> quit
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np


NUM_ACTIONS = 14
SINGLE_OBS_DIM = 52
FRAME_STACK = 15
OBS_DIM = SINGLE_OBS_DIM * FRAME_STACK

SIM_DT = 0.005
DECIMATION = 4
POLICY_DT = SIM_DT * DECIMATION
ACTION_SCALE = 0.25
CLIP_OBSERVATIONS = 18.0
CLIP_ACTIONS = 18.0

CYCLE_TIME = 1.2
SLOW_CYCLE_TIME = 1.1
STAND_COMMAND_THRESHOLD = 0.05
MIN_COMMAND = 0.2

COMMAND_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ANG_VEL_SCALE = 0.25

DEFAULT_DOF_POS = np.array(
    [
        -0.10, 0.0, 0.0, 0.20, -0.10, 0.0,
        -0.10, 0.0, 0.0, 0.20, -0.10, 0.0,
        0.0, 0.0,
    ],
    dtype=np.float64,
)
KP = np.array(
    [
        100.0, 100.0, 100.0, 150.0, 30.0, 30.0,
        100.0, 100.0, 100.0, 150.0, 30.0, 30.0,
        200.0, 200.0,
    ],
    dtype=np.float64,
)
KD = np.array(
    [
        2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
        5.0, 5.0,
    ],
    dtype=np.float64,
)
TORQUE_LIMITS = np.array(
    [
        75.0, 87.0, 87.0, 120.0, 89.0, 12.0,
        75.0, 87.0, 87.0, 120.0, 89.0, 12.0,
        87.0, 87.0,
    ],
    dtype=np.float64,
)

JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="X3 F2 AMP Isaac Gym -> MuJoCo sim2sim with joystick control"
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="exported ONNX policy; default: newest x3_f2 policy_blind_est.onnx",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=repo_root / "resources/robots/F2_ZZ1_waiguan/x3_f2_14dof.xml",
    )
    parser.add_argument("--joystick", type=int, default=0, help="pygame joystick index")
    parser.add_argument("--axis-lx", type=int, default=0)
    parser.add_argument("--axis-ly", type=int, default=1)
    parser.add_argument("--axis-rx", type=int, default=3)
    parser.add_argument(
        "--button-stop", type=int, default=0, help="zero command while held (Xbox A)"
    )
    parser.add_argument(
        "--button-reset", type=int, default=1, help="reset simulation (Xbox B)"
    )
    parser.add_argument(
        "--button-quit", type=int, default=7, help="quit (Xbox Start)"
    )
    parser.add_argument("--deadzone", type=float, default=0.12)
    parser.add_argument("--expo", type=float, default=1.5)
    parser.add_argument("--vx-forward", type=float, default=1.0)
    parser.add_argument("--vx-backward", type=float, default=0.6)
    parser.add_argument("--vy-max", type=float, default=0.5)
    parser.add_argument("--yaw-max", type=float, default=1.5)
    parser.add_argument(
        "--no-command-threshold",
        action="store_true",
        help="do not reproduce the training min_vel=0.2 cutoff",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="seconds to run; 0 means until viewer closes/Start is pressed",
    )
    parser.add_argument(
        "--no-realtime", action="store_true", help="run as fast as possible"
    )
    return parser.parse_args()


def find_latest_policy(repo_root: Path) -> Path:
    policy_root = repo_root / "logs/x3_f2"
    candidates = list(policy_root.glob("*/exported/policy_blind_est.onnx"))
    candidates.extend(policy_root.glob("*/policies/policy_blind_est.onnx"))
    if not candidates:
        raise FileNotFoundError(
            "No exported X3 F2 policy found. Pass "
            "--policy /path/to/policy_blind_est.onnx"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def deadzone_expo(value: float, deadzone: float, expo: float) -> float:
    magnitude = abs(float(value))
    if magnitude <= deadzone:
        return 0.0
    normalized = min(1.0, (magnitude - deadzone) / (1.0 - deadzone))
    return math.copysign(normalized**expo, value)


def rotate_inverse_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the body frame."""
    qw = float(quaternion[0])
    qvec = np.asarray(quaternion[1:4], dtype=np.float64)
    vec = np.asarray(vector, dtype=np.float64)
    return (
        vec * (2.0 * qw * qw - 1.0)
        - 2.0 * qw * np.cross(qvec, vec)
        + 2.0 * qvec * np.dot(qvec, vec)
    )


class JoystickCommand:
    def __init__(self, args: argparse.Namespace):
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("pygame is required: pip install pygame") from exc

        self.pygame = pygame
        pygame.display.init()
        pygame.joystick.init()
        pygame.event.pump()
        count = pygame.joystick.get_count()
        if args.joystick < 0 or args.joystick >= count:
            raise RuntimeError(
                f"Joystick index {args.joystick} not found ({count} detected). "
                "Connect the controller and check /dev/input permissions."
            )
        self.device = pygame.joystick.Joystick(args.joystick)
        self.device.init()
        self.args = args
        self.previous_reset = False
        print(
            f"Joystick: {self.device.get_name()} | axes={self.device.get_numaxes()} "
            f"buttons={self.device.get_numbuttons()}"
        )
        for axis in (args.axis_lx, args.axis_ly, args.axis_rx):
            if axis < 0 or axis >= self.device.get_numaxes():
                raise ValueError(
                    f"Axis {axis} is unavailable; controller has "
                    f"{self.device.get_numaxes()} axes"
                )

    def _button(self, index: int) -> bool:
        return 0 <= index < self.device.get_numbuttons() and bool(
            self.device.get_button(index)
        )

    def read(self) -> tuple[np.ndarray, bool, bool]:
        self.pygame.event.pump()
        args = self.args
        lx = deadzone_expo(
            self.device.get_axis(args.axis_lx), args.deadzone, args.expo
        )
        ly = deadzone_expo(
            self.device.get_axis(args.axis_ly), args.deadzone, args.expo
        )
        rx = deadzone_expo(
            self.device.get_axis(args.axis_rx), args.deadzone, args.expo
        )

        # Robot convention: +x forward, +y left, +yaw counter-clockwise.
        vx_stick = -ly
        vx = vx_stick * (
            args.vx_forward if vx_stick >= 0.0 else args.vx_backward
        )
        command = np.array(
            [vx, -lx * args.vy_max, -rx * args.yaw_max], dtype=np.float32
        )
        if not args.no_command_threshold:
            command[np.abs(command) < MIN_COMMAND] = 0.0
        if self._button(args.button_stop):
            command.fill(0.0)

        reset_now = self._button(args.button_reset)
        reset_edge = reset_now and not self.previous_reset
        self.previous_reset = reset_now
        return command, reset_edge, self._button(args.button_quit)

    def close(self) -> None:
        self.device.quit()
        self.pygame.joystick.quit()
        self.pygame.display.quit()


class X3F2Sim2Sim:
    def __init__(self, xml_path: Path, policy_path: Path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required: pip install onnxruntime") from exc

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = SIM_DT

        self.qpos_indices = np.empty(NUM_ACTIONS, dtype=np.int32)
        self.qvel_indices = np.empty(NUM_ACTIONS, dtype=np.int32)
        self.actuator_indices = np.empty(NUM_ACTIONS, dtype=np.int32)
        for index, joint_name in enumerate(JOINT_NAMES):
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
            )
            if joint_id < 0 or actuator_id < 0:
                raise ValueError(f"XML is missing joint or actuator {joint_name}")
            self.qpos_indices[index] = self.model.jnt_qposadr[joint_id]
            self.qvel_indices[index] = self.model.jnt_dofadr[joint_id]
            self.actuator_indices[index] = actuator_id

            dof_id = self.qvel_indices[index]
            self.model.dof_damping[dof_id] = 0.1
            self.model.dof_armature[dof_id] = 0.01
            self.model.dof_frictionloss[dof_id] = 0.0

        self.session = ort.InferenceSession(
            str(policy_path), providers=["CPUExecutionProvider"]
        )
        input_meta = self.session.get_inputs()[0]
        if input_meta.shape[-1] != OBS_DIM:
            raise ValueError(
                f"Policy expects {input_meta.shape[-1]} observations, "
                f"X3 F2 AMP expects {OBS_DIM}"
            )
        self.input_name = input_meta.name
        output_names = {output.name for output in self.session.get_outputs()}
        self.action_output = (
            "action" if "action" in output_names else self.session.get_outputs()[0].name
        )

        self.history: deque[np.ndarray] = deque(maxlen=FRAME_STACK)
        self.action = np.zeros(NUM_ACTIONS, dtype=np.float64)
        self.phase = 0.0
        self.stand = True
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.90
        self.data.qpos[self.qpos_indices] = DEFAULT_DOF_POS
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.history.clear()
        for _ in range(FRAME_STACK):
            self.history.append(np.zeros(SINGLE_OBS_DIM, dtype=np.float32))
        self.action.fill(0.0)
        self.phase = 0.0
        self.stand = True

    def _roll_pitch(self) -> tuple[float, float]:
        # MuJoCo free-joint quaternion order is w, x, y, z.
        w, x, y, z = self.data.qpos[3:7]
        roll = math.atan2(
            2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)
        )
        sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        pitch = math.asin(sin_pitch)
        return roll, pitch

    def _update_phase(self, command: np.ndarray) -> None:
        cycle_time = SLOW_CYCLE_TIME if abs(float(command[0])) < 0.2 else CYCLE_TIME
        local_velocity = rotate_inverse_wxyz(
            self.data.qpos[3:7], self.data.qvel[0:3]
        )
        low_speed = np.linalg.norm(local_velocity[:2]) <= 0.5
        self.stand = (
            np.linalg.norm(command) < STAND_COMMAND_THRESHOLD and low_speed
        )
        self.phase = (self.phase + POLICY_DT / cycle_time) % 1.0
        if self.stand:
            self.phase = 0.0

    def _single_observation(self, command: np.ndarray) -> np.ndarray:
        q = self.data.qpos[self.qpos_indices]
        dq = self.data.qvel[self.qvel_indices]
        # MuJoCo free-joint rotational qvel is expressed in the local body frame.
        gyro = self.data.qvel[3:6]
        roll, pitch = self._roll_pitch()
        if self.stand:
            clock = (0.0, 0.0)
        else:
            angle = 2.0 * math.pi * self.phase
            clock = (math.sin(angle), math.cos(angle))

        obs = np.concatenate(
            (
                np.asarray(clock, dtype=np.float32),
                command * COMMAND_SCALE,
                ((q - DEFAULT_DOF_POS) * DOF_POS_SCALE).astype(np.float32),
                (dq * DOF_VEL_SCALE).astype(np.float32),
                self.action.astype(np.float32),
                (gyro * ANG_VEL_SCALE).astype(np.float32),
                np.asarray((roll, pitch), dtype=np.float32),
            )
        )
        if obs.shape != (SINGLE_OBS_DIM,):
            raise RuntimeError(f"Internal observation shape error: {obs.shape}")
        return obs

    def infer(self, command: np.ndarray) -> None:
        self._update_phase(command)
        self.history.append(self._single_observation(command))
        observation = np.concatenate(self.history).reshape(1, OBS_DIM)
        observation = np.clip(
            observation, -CLIP_OBSERVATIONS, CLIP_OBSERVATIONS
        ).astype(np.float32)
        action = self.session.run(
            [self.action_output], {self.input_name: observation}
        )[0]
        action = np.asarray(action).reshape(-1)
        if action.shape != (NUM_ACTIONS,):
            raise RuntimeError(f"Policy action shape is {action.shape}, expected (14,)")
        self.action[:] = np.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS)

    def apply_pd(self) -> None:
        q = self.data.qpos[self.qpos_indices]
        dq = self.data.qvel[self.qvel_indices]
        target = DEFAULT_DOF_POS + ACTION_SCALE * self.action
        torque = np.clip(
            KP * (target - q) - KD * dq, -TORQUE_LIMITS, TORQUE_LIMITS
        )
        self.data.ctrl[self.actuator_indices] = torque


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.deadzone < 1.0:
        raise ValueError("--deadzone must be in [0, 1)")
    if args.expo <= 0.0:
        raise ValueError("--expo must be positive")

    repo_root = Path(__file__).resolve().parents[2]
    policy_path = args.policy or find_latest_policy(repo_root)
    xml_path = args.xml
    if not policy_path.is_file():
        raise FileNotFoundError(policy_path)
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)

    print(f"Policy: {policy_path}")
    print(f"Robot XML: {xml_path}")
    controller = JoystickCommand(args)
    simulator = X3F2Sim2Sim(xml_path, policy_path)

    viewer = None
    if not args.headless:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(simulator.model, simulator.data)
        pelvis_id = mujoco.mj_name2id(
            simulator.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = pelvis_id
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -15.0

    start_time = time.monotonic()
    next_step_time = start_time
    physics_steps = 0
    last_print = start_time
    command = np.zeros(3, dtype=np.float32)
    print(
        "Controls: left stick=vx/vy, right stick x=yaw, "
        "A=stop, B=reset, Start=quit"
    )

    try:
        while viewer is None or viewer.is_running():
            now = time.monotonic()
            if args.duration > 0.0 and now - start_time >= args.duration:
                break

            command, reset, quit_requested = controller.read()
            if quit_requested:
                break
            if reset:
                simulator.reset()
                physics_steps = 0
                next_step_time = time.monotonic()
                print("Simulation reset")

            if physics_steps % DECIMATION == 0:
                simulator.infer(command)
            simulator.apply_pd()
            mujoco.mj_step(simulator.model, simulator.data)
            physics_steps += 1

            if viewer is not None:
                viewer.sync()
            if now - last_print >= 1.0:
                print(
                    f"\rcommand vx={command[0]:+.2f} vy={command[1]:+.2f} "
                    f"yaw={command[2]:+.2f} stand={simulator.stand}   ",
                    end="",
                    flush=True,
                )
                last_print = now

            if not args.no_realtime:
                next_step_time += SIM_DT
                delay = next_step_time - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
                elif delay < -0.1:
                    next_step_time = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        print()
        controller.close()
        if viewer is not None:
            viewer.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
