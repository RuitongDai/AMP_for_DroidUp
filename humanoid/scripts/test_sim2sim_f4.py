"""Deployment checks; run with python -m unittest discover -s humanoid/scripts -p 'test_sim2sim_f4.py'."""

import importlib.util
import math
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from sim2sim_f4 import Config, CsvRecorder, Gamepad, OnnxPolicy, ROOT, Sim2Sim, load_model, parse_args


class Sim2SimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model(ROOT / "resources/robots/F4/scene_f4.xml")

    def setUp(self):
        self.calls = []

        def policy(obs):
            self.calls.append(obs.copy())
            return np.zeros(14), None

        self.sim = Sim2Sim(self.model, policy)

    def test_fixed_arms_and_joint_mapping(self):
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu), (21, 20, 14))
        for name, index in zip(Config.joint_names, self.sim.actuator_ids):
            joint_id = self.model.actuator_trnid[index, 0]
            self.assertEqual(self.model.joint(joint_id).name, name)
        np.testing.assert_allclose(self.model.actuator_ctrlrange[self.sim.actuator_ids, 1],
                                   Config.torque_limits)
        np.testing.assert_allclose(self.sim.data.qpos[self.sim.q_indices], Config.default_pos)

    def test_observation_layout_and_body_frame(self):
        sim = self.sim
        sim.data.qpos[sim.base_q + 3:sim.base_q + 7] = [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]
        sim.data.qvel[sim.base_v:sim.base_v + 3] = [0, 0.3, 0]
        sim.data.qvel[sim.base_v + 3:sim.base_v + 6] = [0.4, -0.8, 1.2]
        offset = np.arange(14) * 0.01
        sim.data.qpos[sim.q_indices] += offset
        sim.data.qvel[sim.dq_indices] = np.arange(14)
        sim.action = np.arange(14) * 0.1
        mujoco.mj_forward(sim.model, sim.data)
        command = np.array([0.4, -0.2, 0.8])
        obs = sim.get_obs(command, "stand_and_walk")
        expected = np.concatenate([
            [0, 1], [0.8, -0.4, 0.2], offset, np.arange(14) * 0.05,
            sim.action, [0.1, -0.2, 0.3], [0, 0],
        ])
        self.assertEqual(obs.shape, (1, 780))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_allclose(obs[0, :-52], 0)
        np.testing.assert_allclose(obs[0, -52:], expected, atol=1e-6)
        np.testing.assert_allclose(sim.base_state()[0], [0.3, 0, 0], atol=1e-6)
        sim.steps = 4
        next_obs = sim.get_obs(command, "stand_and_walk")
        np.testing.assert_allclose(next_obs[0, -104:-52], expected, atol=1e-6)
        self.assertAlmostEqual(sim.phase, 0.02 / 1.2)

    def test_standing_slow_gait_and_coasting(self):
        sim = self.sim
        sim.steps = 4
        sim.phase = 0.4
        obs = sim.get_obs(np.zeros(3), "stand_and_walk")
        np.testing.assert_array_equal(obs[0, -52:-50], [0, 0])
        self.assertEqual(sim.phase, 0)
        sim.data.qvel[sim.base_v] = 0.6
        sim.get_obs(np.zeros(3), "stand_and_walk")
        self.assertAlmostEqual(sim.phase, 0.02 / 1.1)
        sim.data.qvel[sim.base_v] = 0
        sim.get_obs(np.zeros(3), "walk")
        self.assertAlmostEqual(sim.phase, 0.04 / 1.1)

    def test_policy_rate_torque_limits_and_reset(self):
        for _ in range(12):
            self.sim.step()
        self.assertEqual(len(self.calls), 3)
        self.assertAlmostEqual(self.sim.data.time, 0.06)
        self.sim.policy = lambda obs: (np.full(14, 100.0), None)
        self.sim.step()
        np.testing.assert_array_equal(self.sim.action, np.full(14, 18))
        np.testing.assert_allclose(self.sim.data.ctrl[self.sim.actuator_ids], Config.torque_limits)
        self.sim.reset()
        self.assertEqual(self.sim.steps, 0)
        self.assertEqual(self.sim.phase, 0)
        self.assertEqual(self.sim.data.time, 0)
        np.testing.assert_allclose(np.concatenate(self.sim.history), 0)
        np.testing.assert_allclose(self.sim.action, 0)
        np.testing.assert_allclose(self.sim.data.qvel, 0)
        np.testing.assert_allclose(self.sim.data.xfrc_applied, 0)

    def test_keyboard_and_requested_reset(self):
        self.sim.key_callback(ord("W"))
        self.assertAlmostEqual(self.sim.command[0], 0.05)
        self.sim.key_callback(ord("1"))
        self.assertEqual(self.sim.mode, "walk")
        self.sim.step()
        self.sim.key_callback(ord("R"))
        self.sim.step()
        self.assertEqual(self.sim.steps, 1)
        np.testing.assert_allclose(self.sim.command, 0)
        np.testing.assert_allclose(self.calls[-1][0, :-52], 0)

    def test_csv_recording_toggle_keeps_previous_session(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.csv"
            recorder = CsvRecorder(path)
            self.sim.step()
            recorder.write(self.sim)
            recorder.toggle()
            first = path.read_text()
            self.assertEqual(len(first.splitlines()), 2)
            recorder.toggle()
            recorder.write(self.sim)
            recorder.close()
            self.assertEqual(path.read_text(), first)
            self.assertEqual(len(list(Path(directory).glob("*.csv"))), 2)


@unittest.skipUnless(importlib.util.find_spec("evdev"), "Gamepad checks require evdev")
class GamepadTests(unittest.TestCase):
    def setUp(self):
        from evdev import ecodes

        self.c = ecodes
        self.events = []
        self.disconnected = False
        test = self

        class FakeDevice:
            name = "Xbox 360 pad"

            def active_keys(self):
                return []

            def absinfo(self, code):
                if code == test.c.ABS_Z:
                    return SimpleNamespace(value=0, min=0, max=255)
                return SimpleNamespace(value=0, min=-32767, max=32767)

            def read(self):
                if test.disconnected:
                    raise OSError("disconnected")
                events, test.events = test.events, []
                return events

            def close(self):
                pass

        self.device = FakeDevice()
        with patch("evdev.InputDevice", return_value=self.device), patch("evdev.list_devices", return_value=["fake"]):
            self.pad = Gamepad()
        self.sim = SimpleNamespace(lock=threading.Lock(), command=np.zeros(3),
                                   mode="stand_and_walk", push_enabled=False, push_force=np.zeros(3),
                                   reset_requested=False, record_toggle_requested=False)

    def event(self, code, value, button=False):
        self.events.append(SimpleNamespace(type=self.c.EV_KEY if button else self.c.EV_ABS,
                                           code=code, value=value))

    def test_sticks_turn_gate_and_mode_buttons(self):
        c = self.c
        self.event(c.ABS_Y, -32767)
        self.event(c.ABS_RX, 32767)
        self.event(c.ABS_X, -32767)
        self.pad.update(self.sim)
        np.testing.assert_allclose(self.sim.command, [1, -0.5, 0])
        self.event(c.ABS_Z, 251)
        self.event(c.BTN_START, 1, button=True)
        self.pad.update(self.sim)
        np.testing.assert_allclose(self.sim.command, [1, -0.5, 1.5])
        self.assertEqual(self.sim.mode, "walk")
        self.event(c.BTN_START, 0, button=True)
        self.event(c.BTN_SELECT, 1, button=True)
        self.pad.update(self.sim)
        self.assertEqual(self.sim.mode, "stand_and_walk")

    def test_button_edges_push_and_disconnect(self):
        c = self.c
        self.event(c.ABS_RX, -32767)
        self.event(c.ABS_RY, -32767)
        for code in (c.BTN_NORTH, c.BTN_WEST, c.BTN_TR):
            self.event(code, 1, button=True)
        self.pad.update(self.sim)
        self.assertTrue(self.sim.reset_requested)
        self.assertTrue(self.sim.record_toggle_requested)
        self.assertTrue(self.sim.push_enabled)
        np.testing.assert_allclose(self.sim.push_force, [10, 10, 0])
        self.assertEqual(self.sim.command[1], 0)
        self.sim.reset_requested = self.sim.record_toggle_requested = False
        self.pad.update(self.sim)
        self.assertFalse(self.sim.reset_requested)
        self.assertFalse(self.sim.record_toggle_requested)
        self.assertTrue(self.sim.push_enabled)
        self.disconnected = True
        self.pad.update(self.sim)
        self.assertIsNone(self.pad.device)
        np.testing.assert_allclose(self.sim.command, 0)
        np.testing.assert_allclose(self.sim.push_force, 0)
        self.assertFalse(self.sim.push_enabled)
        self.disconnected = False
        self.pad.next_connect = 0
        with patch.object(self.pad, "input_device", return_value=self.device), \
                patch.object(self.pad, "list_devices", return_value=["fake"]):
            self.pad.update(self.sim)
        self.assertIs(self.pad.device, self.device)
        np.testing.assert_allclose(self.sim.command, 0)


class ArgumentTests(unittest.TestCase):
    def test_default_gamepad_and_explicit_fixed_control(self):
        with tempfile.NamedTemporaryFile(suffix=".onnx") as file:
            with patch("sys.argv", ["sim2sim_f4.py", "--onnx", file.name]):
                self.assertEqual(parse_args().control, "gamepad")
            with patch("sys.argv", ["sim2sim_f4.py", "--onnx", file.name, "--fixed-command", "--cmd", "0.4", "0", "0"]):
                self.assertEqual(parse_args().control, "fixed")


@unittest.skipUnless(importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
                     "ONNX test requires onnx and onnxruntime")
class OnnxTests(unittest.TestCase):
    def test_export_interface_and_invalid_shapes(self):
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.onnx"
            for obs_dim, action_dim, valid in [(780, 14, True), (52, 14, False), (780, 12, False)]:
                graph = helper.make_graph([
                    helper.make_node("Constant", [], ["action"],
                                     value=numpy_helper.from_array(np.zeros((1, action_dim), np.float32))),
                    helper.make_node("Constant", [], ["est"],
                                     value=numpy_helper.from_array(np.array([[2, 4, 6]], np.float32))),
                ], "test", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, obs_dim])], [
                    helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, action_dim]),
                    helper.make_tensor_value_info("est", TensorProto.FLOAT, [1, 3]),
                ])
                model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
                model.ir_version = 8
                onnx.save(model, path)
                if valid:
                    policy = OnnxPolicy(path)
                    action, est = policy(np.zeros((1, 780), np.float32))
                    np.testing.assert_array_equal(action, np.zeros(14))
                    np.testing.assert_array_equal(est, [1, 2, 3])
                else:
                    with self.assertRaises(ValueError):
                        OnnxPolicy(path)(np.zeros((1, 780), np.float32))


if __name__ == "__main__":
    unittest.main()
