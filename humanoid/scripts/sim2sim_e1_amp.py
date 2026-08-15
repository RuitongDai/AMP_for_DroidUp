#!/usr/bin/env python3
"""Run the E1 AMP locomotion policy in MuJoCo with joystick commands.

The observation and controller constants in this file mirror
``humanoid/envs/e1/e1_cfg_amp.py``.  This script deliberately does not import
Isaac Gym, so it can be used in a lightweight MuJoCo deployment environment.

Default Xbox-style mapping (axis/button ids can be changed from the CLI):
  left stick vertical   -> forward/backward velocity
  left stick horizontal -> lateral velocity
  right stick horizontal -> yaw rate
  A                      -> zero command while held
  B                      -> reset simulation
  Start                  -> quit
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


NUM_ACTIONS = 13
SINGLE_OBS_DIM = 49
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
    [-0.10, 0.0, 0.0, 0.23, -0.13, 0.0,
     -0.10, 0.0, 0.0, 0.23, -0.13, 0.0, 0.0],
    dtype=np.float64,
)
KP = np.array(
    [200.0, 200.0, 80.0, 200.0, 80.0, 80.0,
     200.0, 200.0, 80.0, 200.0, 80.0, 80.0, 150.0],
    dtype=np.float64,
)
KD = np.array(
    [5.0, 5.0, 3.0, 5.0, 3.0, 3.0,
     5.0, 5.0, 3.0, 5.0, 3.0, 3.0, 4.0],
    dtype=np.float64,
)
TORQUE_LIMITS = np.array(
    [120.0, 60.0, 36.0, 120.0, 30.0, 30.0,
     120.0, 60.0, 36.0, 120.0, 30.0, 30.0, 60.0],
    dtype=np.float64,
)

JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
)
ACTUATOR_NAMES = tuple(name.replace("_joint", "_motor") for name in JOINT_NAMES)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="E1 AMP Isaac Gym -> MuJoCo sim2sim with joystick control"
    )
    parser.add_argument("--policy", type=Path, default=None,
                        help="exported ONNX policy; default: newest e1_no_hands export")
    parser.add_argument("--xml", type=Path,
                        default=repo_root / "resources/robots/e1/E1_no_hand.xml")
    parser.add_argument("--joystick", type=int, default=0,
                        help="pygame joystick index")
    parser.add_argument("--axis-lx", type=int, default=0)
    parser.add_argument("--axis-ly", type=int, default=1)
    parser.add_argument("--axis-rx", type=int, default=3)
    parser.add_argument("--button-stop", type=int, default=0,
                        help="zero command while held (Xbox A)")
    parser.add_argument("--button-reset", type=int, default=1,
                        help="reset simulation (Xbox B)")
    parser.add_argument("--button-quit", type=int, default=7,
                        help="quit (Xbox Start)")
    parser.add_argument("--deadzone", type=float, default=0.12)
    parser.add_argument("--expo", type=float, default=1.5)
    parser.add_argument("--vx-forward", type=float, default=1.0)
    parser.add_argument("--vx-backward", type=float, default=0.6)
    parser.add_argument("--vy-max", type=float, default=0.5)
    parser.add_argument("--yaw-max", type=float, default=1.0)
    parser.add_argument("--no-command-threshold", action="store_true",
                        help="do not reproduce the training min_vel=0.2 cutoff")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="seconds to run; 0 means until viewer closes/Start is pressed")
    parser.add_argument("--no-realtime", action="store_true",
                        help="run as fast as possible")
    return parser.parse_args()


def find_latest_policy(repo_root: Path) -> Path:
    candidates = list((repo_root / "logs/e1_no_hands").glob(
        "*/exported/policy_blind_est.onnx"
    ))
    if not candidates:
        raise FileNotFoundError(
            "No exported E1 policy found. Pass --policy /path/to/policy_blind_est.onnx"
        )
    return max(candidates, key=lambda path: path.parent.parent.name)


def deadzone_expo(value: float, deadzone: float, expo: float) -> float:
    magnitude = abs(float(value))
    if magnitude <= deadzone:
        return 0.0
    normalized = min(1.0, (magnitude - deadzone) / (1.0 - deadzone))
    return math.copysign(normalized ** expo, value)


class JoystickCommand:
    def __init__(self, args: argparse.Namespace):
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("pygame is required: pip install pygame") from exc

        self.pygame = pygame
        # pygame's event pump requires the display subsystem to be initialized,
        # but it does not require us to create a second window.
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
                    f"Axis {axis} is unavailable; controller has {self.device.get_numaxes()} axes"
                )

    def _button(self, index: int) -> bool:
        return 0 <= index < self.device.get_numbuttons() and bool(self.device.get_button(index))

    def read(self) -> tuple[np.ndarray, bool, bool]:
        self.pygame.event.pump()
        a = self.args
        lx = deadzone_expo(self.device.get_axis(a.axis_lx), a.deadzone, a.expo)
        ly = deadzone_expo(self.device.get_axis(a.axis_ly), a.deadzone, a.expo)
        rx = deadzone_expo(self.device.get_axis(a.axis_rx), a.deadzone, a.expo)

        # Robot convention: +x forward, +y left and +yaw counter-clockwise.
        vx_stick = -ly
        vx = vx_stick * (a.vx_forward if vx_stick >= 0.0 else a.vx_backward)
        command = np.array([vx, -lx * a.vy_max, -rx * a.yaw_max], dtype=np.float32)
        if not a.no_command_threshold:
            command[np.abs(command) < MIN_COMMAND] = 0.0
        if self._button(a.button_stop):
            command.fill(0.0)

        reset_now = self._button(a.button_reset)
        reset_edge = reset_now and not self.previous_reset
        self.previous_reset = reset_now
        return command, reset_edge, self._button(a.button_quit)

    def close(self) -> None:
        self.device.quit()
        self.pygame.joystick.quit()
        self.pygame.display.quit()


class E1Sim2Sim:
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
        for index, (joint_name, actuator_name) in enumerate(zip(JOINT_NAMES, ACTUATOR_NAMES)):
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if joint_id < 0 or actuator_id < 0:
                raise ValueError(f"XML is missing {joint_name} or {actuator_name}")
            self.qpos_indices[index] = self.model.jnt_qposadr[joint_id]
            self.qvel_indices[index] = self.model.jnt_dofadr[joint_id]
            self.actuator_indices[index] = actuator_id

            dof_id = self.qvel_indices[index]
            self.model.dof_damping[dof_id] = 0.1
            self.model.dof_armature[dof_id] = 0.01
            self.model.dof_frictionloss[dof_id] = 0.0

        self.gyro_slice = self._sensor_slice("imu_ang_vel", expected_dim=3)
        self.velocity_slice = self._sensor_slice("imu_lin_vel", expected_dim=3)

        self.session = ort.InferenceSession(
            str(policy_path), providers=["CPUExecutionProvider"]
        )
        input_meta = self.session.get_inputs()[0]
        if input_meta.shape[-1] != OBS_DIM:
            raise ValueError(
                f"Policy expects {input_meta.shape[-1]} observations, E1 AMP expects {OBS_DIM}"
            )
        self.input_name = input_meta.name
        output_names = {output.name for output in self.session.get_outputs()}
        self.action_output = "action" if "action" in output_names else self.session.get_outputs()[0].name

        self.history: deque[np.ndarray] = deque(maxlen=FRAME_STACK)
        self.action = np.zeros(NUM_ACTIONS, dtype=np.float64)
        self.phase = 0.0
        self.stand = True
        self.reset()

    def _sensor_slice(self, name: str, expected_dim: int) -> slice:
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0 or self.model.sensor_dim[sensor_id] != expected_dim:
            raise ValueError(f"XML must contain a {expected_dim}D sensor named {name}")
        start = int(self.model.sensor_adr[sensor_id])
        return slice(start, start + expected_dim)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
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
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        pitch = math.asin(sin_pitch)
        return roll, pitch

    def _update_phase(self, command: np.ndarray) -> None:
        # Matches _gait_style_update() followed by _phase_step_update().
        cycle_time = SLOW_CYCLE_TIME if abs(float(command[0])) < 0.2 else CYCLE_TIME
        local_velocity = self.data.sensordata[self.velocity_slice]
        low_speed = np.linalg.norm(local_velocity[:2]) <= 0.5
        self.stand = np.linalg.norm(command) < STAND_COMMAND_THRESHOLD and low_speed
        self.phase = (self.phase + POLICY_DT / cycle_time) % 1.0
        if self.stand:
            self.phase = 0.0

    def _single_observation(self, command: np.ndarray) -> np.ndarray:
        q = self.data.qpos[self.qpos_indices]
        dq = self.data.qvel[self.qvel_indices]
        gyro = self.data.sensordata[self.gyro_slice]
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
        observation = np.clip(observation, -CLIP_OBSERVATIONS, CLIP_OBSERVATIONS)
        action = self.session.run(
            [self.action_output], {self.input_name: observation.astype(np.float32)}
        )[0]
        self.action[:] = np.clip(
            np.asarray(action).reshape(-1), -CLIP_ACTIONS, CLIP_ACTIONS
        )

    def apply_pd(self) -> None:
        q = self.data.qpos[self.qpos_indices]
        dq = self.data.qvel[self.qvel_indices]
        target = DEFAULT_DOF_POS + ACTION_SCALE * self.action
        torque = np.clip(KP * (target - q) - KD * dq, -TORQUE_LIMITS, TORQUE_LIMITS)
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
    simulator = E1Sim2Sim(xml_path, policy_path)

    viewer = None
    if not args.headless:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(simulator.model, simulator.data)
        pelvis_id = mujoco.mj_name2id(simulator.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = pelvis_id
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -15.0

    start_time = time.monotonic()
    next_step_time = start_time
    physics_steps = 0
    last_print = start_time
    command = np.zeros(3, dtype=np.float32)
    print("Controls: left stick=vx/vy, right stick x=yaw, A=stop, B=reset, Start=quit")

    try:
        while (viewer is None or viewer.is_running()):
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
                    f"yaw={command[2]:+.2f}   ",
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
