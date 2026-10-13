"""Run an f4_amp ONNX policy in MuJoCo, without importing Isaac Gym.

Parameters mirror humanoid/envs/f4/f4_cfg_amp.py. The observation, gait and
PD equations mirror f4_env.py; interactive controls follow x3_amp_deploy.
"""

import argparse
import csv
import math
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


class Config:
    # Isaac Gym order from f4_14dof.urdf: waist ROLL, then YAW.
    joint_names = tuple(
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee",
                      "ankle_pitch", "ankle_roll")
    ) + ("waist_roll_joint", "waist_yaw_joint")
    num_actions = 14
    single_obs = 52
    frame_stack = 15
    timestep = 0.005
    decimation = 4
    cycle_time = 1.2
    cycle_time_low = 1.1
    clock_enable = 1.0
    stand_threshold = 0.05
    stand_speed = 0.5
    action_scale = 0.25
    clip_obs = 18.0
    clip_actions = 18.0
    lin_vel_scale = 2.0
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    euler_scale = 1.0
    default_pos = np.array([-0.1, 0, 0, 0.2, -0.1, 0] * 2 + [0, 0])
    kp = np.array([100, 100, 100, 150, 30, 30] * 2 + [200, 200])
    kd = np.array([2, 2, 2, 4, 2, 2] * 2 + [5, 5])
    torque_limits = np.array([75, 80, 80, 120, 60, 15] * 2 + [80, 80])
    command_lower = np.array([-0.6, -0.5, -1.5])
    command_upper = np.array([1.0, 0.5, 1.5])
    base_pos = np.array([0, 0, 0.875])


def load_model(scene_path):
    """Fix non-policy joints at zero, retaining their bodies and inertias.

    The supplied scene has 28 motors, while training fixes all arm joints.
    Expand includes in memory so the source assets remain reusable.
    """
    def read_xml(path):
        root = ET.parse(path).getroot()
        compiler = root.find("compiler")
        if compiler is not None:
            for key in ("meshdir", "texturedir"):
                compiler.set(key, str((path.parent / compiler.get(key, ".")).resolve()))
        for element in root.iter():
            if element.tag in ("mesh", "texture", "hfield") and element.get("file"):
                directory = path.parent
                if compiler is not None and element.tag in ("mesh", "texture"):
                    directory = Path(compiler.get("meshdir" if element.tag == "mesh" else "texturedir"))
                element.set("file", str((directory / element.get("file")).resolve()))
        for include in list(root.findall("include")):
            child = read_xml((path.parent / include.get("file")).resolve())
            index = list(root).index(include)
            root.remove(include)
            for offset, element in enumerate(child):
                root.insert(index + offset, element)
        return root

    root = read_xml(Path(scene_path).resolve())
    fixed = set()
    for body in root.iter("body"):
        for joint in list(body.findall("joint")):
            name = joint.get("name")
            if joint.get("type", "hinge") != "free" and name not in Config.joint_names:
                fixed.add(name)
                body.remove(joint)
            elif name in Config.joint_names:
                limit = Config.torque_limits[Config.joint_names.index(name)]
                joint.set("actuatorfrcrange", f"{-limit} {limit}")
                joint.set("actuatorfrclimited", "true")
                joint.set("damping", "0.1")
                joint.set("armature", "0.01")
                joint.set("frictionloss", "0")
    for section in root.findall("actuator") + root.findall("sensor"):
        for element in list(section):
            if element.get("joint") in fixed:
                section.remove(element)
            elif section.tag == "actuator" and element.get("joint") in Config.joint_names:
                if element.tag != "motor" or element.get("gear", "1") != "1":
                    raise ValueError("Policy joints require direct torque motors with gear=1")
                limit = Config.torque_limits[Config.joint_names.index(element.get("joint"))]
                element.set("ctrlrange", f"{-limit} {limit}")
                element.set("forcerange", f"{-limit} {limit}")
                element.set("ctrllimited", "true")
                element.set("forcelimited", "true")
    # The 28-DOF home keyframe no longer matches the fixed-arm model.
    for keyframe in list(root.findall("keyframe")):
        root.remove(keyframe)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.opt.timestep = Config.timestep
    return model


def quat_to_euler(quat):
    """MuJoCo wxyz quaternion -> signed roll and pitch, in radians."""
    w, x, y, z = quat
    return np.array([
        math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        math.asin(np.clip(2 * (w * y - z * x), -1, 1)),
    ])


class OnnxPolicy:
    def __init__(self, path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("Expected one stacked-observation ONNX input")
        self.input = inputs[0]
        shape = self.input.shape
        if (len(shape) != 2 or shape[1] != Config.single_obs * Config.frame_stack
                or (isinstance(shape[0], int) and shape[0] != 1)
                or self.input.type != "tensor(float)"):
            raise ValueError(f"Expected float32 [1, 780] input, got {shape}, {self.input.type}")
        outputs = self.session.get_outputs()
        names = [output.name for output in outputs]
        self.action_name = "action" if "action" in names else outputs[0].name
        self.est_name = "est" if "est" in names else None

    def __call__(self, obs):
        names = [self.action_name] + ([self.est_name] if self.est_name else [])
        outputs = self.session.run(names, {self.input.name: obs})
        action = np.asarray(outputs[0])
        if action.shape != (1, Config.num_actions) or not np.all(np.isfinite(action)):
            raise ValueError(f"Expected finite action [1, 14], got {action.shape}")
        est = None
        if self.est_name:
            est = np.asarray(outputs[1])
            if est.shape != (1, 3) or not np.all(np.isfinite(est)):
                raise ValueError("Expected finite velocity estimate [1, 3]")
            est = est[0] / Config.lin_vel_scale
        return action[0], est


class Sim2Sim:
    def __init__(self, model, policy, command=(0, 0, 0), mode="stand_and_walk"):
        self.model = model
        self.data = mujoco.MjData(model)
        self.policy = policy
        self.lock = threading.Lock()
        self.command = np.clip(command, Config.command_lower, Config.command_upper)
        self.mode = mode
        self.reset_requested = False
        self.record_toggle_requested = False
        self.push_enabled = False
        self.push_force = np.zeros(3)
        self.joint_ids = np.array([model.joint(name).id for name in Config.joint_names])
        self.q_indices = model.jnt_qposadr[self.joint_ids]
        self.dq_indices = model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = []
        for joint_id in self.joint_ids:
            matches = np.flatnonzero(
                (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                & (model.actuator_trnid[:, 0] == joint_id)
            )
            if len(matches) != 1:
                raise ValueError("Each policy joint must have exactly one torque motor")
            self.actuator_ids.append(int(matches[0]))
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1 or model.nv != 6 + Config.num_actions:
            raise ValueError("Expected a floating base and exactly 14 movable policy joints")
        self.base_q = int(model.jnt_qposadr[free[0]])
        self.base_v = int(model.jnt_dofadr[free[0]])
        self.base_body = int(model.jnt_bodyid[free[0]])
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.base_q:self.base_q + 3] = Config.base_pos
        self.data.qpos[self.base_q + 3:self.base_q + 7] = [1, 0, 0, 0]
        self.data.qpos[self.q_indices] = Config.default_pos
        self.action = np.zeros(Config.num_actions)
        self.target = Config.default_pos.copy()
        self.est_vel = np.zeros(3)
        self.history = deque(
            [np.zeros(Config.single_obs, dtype=np.float32) for _ in range(Config.frame_stack)],
            maxlen=Config.frame_stack,
        )
        self.phase = 0.0
        self.steps = 0
        self.reset_requested = False
        self.push_enabled = False
        self.push_force[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def base_state(self):
        rotation = self.data.xmat[self.base_body].reshape(3, 3)
        # Free-joint translation velocity is world-frame; rotation velocity is body-frame.
        velocity = rotation.T @ self.data.qvel[self.base_v:self.base_v + 3]
        gyro = self.data.qvel[self.base_v + 3:self.base_v + 6].copy()
        euler = quat_to_euler(self.data.qpos[self.base_q + 3:self.base_q + 7])
        return velocity, gyro, euler

    def get_obs(self, command, mode):
        velocity, gyro, euler = self.base_state()
        standing = (mode == "stand_and_walk"
                    and np.linalg.norm(command) < Config.stand_threshold
                    and np.linalg.norm(velocity[:2]) <= Config.stand_speed)
        # Update once per policy step, matching _phase_step_update in f4_env.py.
        if standing:
            self.phase = 0.0
        elif self.steps:
            cycle = Config.cycle_time_low if abs(command[0]) < 0.2 else Config.cycle_time
            self.phase = (self.phase + Config.timestep * Config.decimation / cycle) % 1.0
        clock = np.zeros(2) if standing else np.array([
            math.sin(2 * math.pi * self.phase), math.cos(2 * math.pi * self.phase),
        ]) * Config.clock_enable
        obs = np.concatenate([
            clock,
            command * [Config.lin_vel_scale, Config.lin_vel_scale, Config.ang_vel_scale],
            (self.data.qpos[self.q_indices] - Config.default_pos) * Config.dof_pos_scale,
            self.data.qvel[self.dq_indices] * Config.dof_vel_scale,
            self.action,
            gyro * Config.ang_vel_scale,
            euler * Config.euler_scale,
        ])
        self.history.append(np.clip(obs, -Config.clip_obs, Config.clip_obs).astype(np.float32))
        return np.concatenate(self.history)[None, :]

    def step(self):
        with self.lock:
            if self.reset_requested:
                self.command[:] = 0
                self.reset()
            command = self.command.copy()
            mode = self.mode
            push_force = self.push_force.copy() if self.push_enabled else np.zeros(3)
        if self.steps % Config.decimation == 0:
            action, est = self.policy(self.get_obs(command, mode))
            self.action = np.clip(action, -Config.clip_actions, Config.clip_actions)
            self.target = Config.default_pos + Config.action_scale * self.action
            # Training clips actions and torques, without clipping position targets.
            if est is not None:
                self.est_vel = est
        torque = Config.kp * (self.target - self.data.qpos[self.q_indices]) \
            - Config.kd * self.data.qvel[self.dq_indices]
        self.data.ctrl[self.actuator_ids] = np.clip(torque, -Config.torque_limits, Config.torque_limits)
        self.data.xfrc_applied[self.base_body, :3] = self.data.xmat[self.base_body].reshape(3, 3) @ push_force
        mujoco.mj_step(self.model, self.data)
        # Refresh poses for the next observation and viewer after integration.
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            raise RuntimeError("MuJoCo state became non-finite")

    def key_callback(self, key):
        with self.lock:
            changes = {ord("W"): (0, 0.05), ord("S"): (0, -0.05),
                       ord("A"): (1, 0.05), ord("D"): (1, -0.05),
                       ord("J"): (2, 0.05), ord("L"): (2, -0.05)}
            if key in changes:
                axis, delta = changes[key]
                self.command[axis] += delta
            elif key == ord(" "):
                self.command[:] = 0
            elif key == ord("R"):
                self.reset_requested = True
            elif key in (ord("1"), ord("2")):
                self.mode = "walk" if key == ord("1") else "stand_and_walk"
            else:
                return
            self.command[:] = np.clip(self.command, Config.command_lower, Config.command_upper)
            print(f"command={self.command.round(2)}, mode={self.mode}")


class Gamepad:
    """Linux evdev controls using the x3_amp_deploy stick/button mapping."""
    def __init__(self, device_path=None):
        try:
            from evdev import InputDevice, ecodes, list_devices
        except ImportError as error:
            raise RuntimeError("手柄控制需要 evdev，请安装: python -m pip install evdev") from error

        self.ecodes = ecodes
        self.input_device = InputDevice
        self.list_devices = list_devices
        self.device_path = device_path
        self.device = None
        self.next_connect = 0.0
        self.axes = {}
        self.buttons = {}
        self.previous = {}
        self.connect()

    def connect(self):
        """Keep the simulation responsive while waiting for or reconnecting a pad."""
        if time.monotonic() < self.next_connect:
            return
        self.next_connect = time.monotonic() + 1.0
        paths = [self.device_path] if self.device_path else self.list_devices()
        for path in paths:
            try:
                device = self.input_device(path)
            except OSError:
                continue
            if self.device_path or any(word in device.name.lower() for word in ("360", "pad")):
                try:
                    buttons = dict.fromkeys(device.active_keys(), True)
                except OSError:
                    device.close()
                    continue
                self.device = device
                self.axes.clear()
                self.previous.clear()
                self.buttons = buttons
                print(f"手柄已连接: {device.name} @ {path}")
                return
            device.close()
        print("未找到可读取的手柄，等待连接（每秒重试）...")

    def axis(self, code, trigger=False):
        if self.device is None:
            return 0.0
        info = self.device.absinfo(code)
        if info is None:
            return 0.0
        value = self.axes.get(code, info.value)
        span = info.max - info.min
        if span <= 0:
            return 0.0
        if trigger:
            return (value - info.min) / span
        result = -(value - (info.min + info.max) / 2) / (span / 2)
        # Match GamepadHandler.__normalize in the reference: invert and round to 2 decimals.
        return round(float(np.clip(result, -1, 1)), 2)

    def update(self, sim):
        c = self.ecodes
        if self.device is None:
            self.connect()
        if self.device is None:
            self.clear_command(sim)
            return
        try:
            for event in self.device.read():
                if event.type == c.EV_ABS:
                    self.axes[event.code] = event.value
                elif event.type == c.EV_KEY:
                    self.buttons[event.code] = bool(event.value)
        except BlockingIOError:
            pass
        except OSError:
            self.close()
            self.clear_command(sim)
            return
        try:
            # Reference LT > 250 for an 8-bit trigger, scaled for other device ranges.
            lt = self.axis(c.ABS_Z, trigger=True) > 250 / 255
            with sim.lock:
                sim.command[:] = np.clip([
                    self.axis(c.ABS_Y) * 1.2,
                    self.axis(c.ABS_RX) * 0.5,
                    self.axis(c.ABS_X) * 1.5 * lt,
                ], Config.command_lower, Config.command_upper)
                if lt and self.buttons.get(c.BTN_START):
                    sim.mode = "walk"
                if lt and self.buttons.get(c.BTN_SELECT):
                    sim.mode = "stand_and_walk"
                # Preserve the reference mapping: BTN_NORTH is X, BTN_WEST is Y.
                if self.buttons.get(c.BTN_NORTH) and not self.previous.get(c.BTN_NORTH):
                    sim.reset_requested = True
                if self.buttons.get(c.BTN_WEST) and not self.previous.get(c.BTN_WEST):
                    sim.record_toggle_requested = True
                if self.buttons.get(c.BTN_TR) and not self.previous.get(c.BTN_TR):
                    sim.push_enabled = not sim.push_enabled
                sim.push_force[:] = [self.axis(c.ABS_RY) * 10, self.axis(c.ABS_RX) * 10, 0]
                if sim.push_enabled:
                    sim.command[1] = 0
        except OSError:
            self.close()
            self.clear_command(sim)
            return
        self.previous = self.buttons.copy()

    @staticmethod
    def clear_command(sim):
        with sim.lock:
            sim.command[:] = 0
            sim.push_force[:] = 0
            sim.push_enabled = False

    def close(self):
        if self.device is not None:
            self.device.close()
            self.device = None
        self.axes.clear()
        self.buttons.clear()
        self.previous.clear()


class CsvRecorder:
    """Y starts/stops recording; --csv starts the first recording immediately."""
    def __init__(self, path=None):
        self.path = path
        self.file = None
        self.writer = None
        self.sessions = 0
        if path is not None:
            self.start()

    def start(self):
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        path = self.path or ROOT / "logs/f4_sim2sim" / f"data_{stamp}.csv"
        if self.path is not None and self.sessions:
            path = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["time", "cmd_vx", "cmd_vy", "cmd_yaw", "vx", "vy", "vz", "roll", "pitch"]
                             + [f"{prefix}_{name}" for prefix in ("q", "dq", "target", "tau")
                                for name in Config.joint_names])
        self.sessions += 1
        print(f"开始录制: {path}")

    def toggle(self):
        if self.file is None:
            self.start()
        else:
            self.close()

    def write(self, sim):
        if self.writer is not None:
            velocity, _, euler = sim.base_state()
            self.writer.writerow([sim.data.time, *sim.command, *velocity, *euler,
                                  *sim.data.qpos[sim.q_indices], *sim.data.qvel[sim.dq_indices],
                                  *sim.target, *sim.data.ctrl[sim.actuator_ids]])

    def close(self):
        if self.file is not None:
            self.file.close()
            print("录制已停止，CSV 已保存")
        self.file = None
        self.writer = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", "--policy", required=True, type=Path, help="f4_amp exported ONNX file")
    parser.add_argument("--xml", type=Path, default=ROOT / "resources/robots/F4/scene_f4.xml")
    parser.add_argument("--cmd", nargs=3, type=float, default=[0, 0, 0], metavar=("VX", "VY", "YAW"),
                        help="Initial command for --keyboard or --fixed-command")
    parser.add_argument("--duration", type=float, default=1000000.0, help="Simulation duration in seconds")
    parser.add_argument("--mode", choices=["walk", "stand_and_walk"], default="stand_and_walk")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer")
    parser.add_argument("--no-realtime", action="store_true", help="Run as fast as possible")
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--gamepad", dest="control", action="store_const", const="gamepad",
                          help="Use a Linux evdev gamepad (default)")
    controls.add_argument("--keyboard", dest="control", action="store_const", const="keyboard",
                          help="Use keyboard controls instead of a gamepad")
    controls.add_argument("--fixed-command", dest="control", action="store_const", const="fixed",
                          help="Use --cmd without a gamepad, e.g. for headless validation")
    parser.set_defaults(control="gamepad")
    parser.add_argument("--gamepad-device", help="Explicit /dev/input/eventN (enables gamepad)")
    parser.add_argument("--csv", type=Path, help="Record policy-rate state and controls to CSV")
    args = parser.parse_args()
    if not args.onnx.is_file():
        parser.error(f"ONNX file does not exist: {args.onnx}")
    if not args.xml.is_file():
        parser.error(f"MuJoCo XML does not exist: {args.xml}")
    if not math.isfinite(args.duration) or args.duration <= 0 or not np.all(np.isfinite(args.cmd)):
        parser.error("Duration must be positive and finite; commands must be finite")
    if args.gamepad_device and args.control != "gamepad":
        parser.error("--gamepad-device requires gamepad control")
    if args.control == "gamepad" and any(args.cmd):
        parser.error("手柄模式从摇杆读取命令；固定命令请添加 --fixed-command，键盘控制请添加 --keyboard")
    return args


def main():
    args = parse_args()
    policy = OnnxPolicy(args.onnx)
    sim = Sim2Sim(load_model(args.xml), policy, args.cmd, args.mode)
    total_steps = math.ceil(args.duration / Config.timestep)
    print(f"Policy: {args.onnx}\nObservation: 15 x 52 = 780; action: 14; policy: 50 Hz; physics: 200 Hz")
    if args.control == "gamepad":
        print("手柄控制: 左摇杆上下前后；右摇杆左右横移；LT+左摇杆左右转向；"
              "LT+Start/Back切换步态；X重置；Y录制；RB切换右摇杆推力")
    elif args.control == "keyboard":
        print("W/S: vx; A/D: vy; J/L: yaw; Space: zero command; R: reset; 1: walk; 2: stand/walk")
    with ExitStack() as stack:
        gamepad = None
        if args.control == "gamepad":
            gamepad = Gamepad(args.gamepad_device)
            stack.callback(gamepad.close)
        viewer = None
        if not args.headless:
            import mujoco.viewer

            viewer = stack.enter_context(mujoco.viewer.launch_passive(
                sim.model, sim.data,
                key_callback=sim.key_callback if args.control == "keyboard" else None,
            ))
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = sim.base_body
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -20
        recorder = CsvRecorder(args.csv)
        stack.callback(recorder.close)
        started = time.perf_counter()
        completed = 0
        try:
            for index in range(total_steps):
                if viewer is not None and not viewer.is_running():
                    break
                if gamepad is not None:
                    gamepad.update(sim)
                if sim.record_toggle_requested:
                    recorder.toggle()
                    sim.record_toggle_requested = False
                sim.step()
                completed += 1
                if index % Config.decimation == 0:
                    if viewer is not None:
                        viewer.sync()
                    recorder.write(sim)
                if not args.no_realtime:
                    delay = started + (index + 1) * Config.timestep - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
        except KeyboardInterrupt:
            pass
        velocity, _, euler = sim.base_state()
        print(f"Finished: {completed * Config.timestep:.3f}s simulated; "
              f"base_z={sim.data.qpos[sim.base_q + 2]:.3f}m; "
              f"velocity={velocity.round(3)}; roll/pitch={euler.round(3)}")


if __name__ == "__main__":
    main()
