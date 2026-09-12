#!/usr/bin/env python3
"""Convert GMR F2 motion data to the 47-column AMP motion format.

Accepted inputs:

* CSV without a header: ``root_pos(3), root_quat_xyzw(4), joint_pos(14)``.
* GMR pickle: keys ``root_pos``, ``root_rot`` (xyzw), ``dof_pos`` and
  optionally ``fps``.

The output is a JSON document (normally named ``*.txt``) consumed by
``humanoid.algo.utils.motion_loader_x3.AMPLoader``.  Root velocities and
foot positions are expressed in the root/base frame, matching
``F2Env.get_amp_obs_for_expert_trans``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


F2_JOINT_NAMES = (
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

F2_JOINT_BODY_NAMES = tuple(name[:-6] + "_link" for name in F2_JOINT_NAMES)
F2_NUM_JOINTS = len(F2_JOINT_NAMES)
AMP_FRAME_SIZE = 47


def normalize_quaternions_xyzw(quaternions: np.ndarray) -> np.ndarray:
    """Normalize xyzw quaternions and make their signs time-continuous."""
    quaternions = np.asarray(quaternions, dtype=np.float64).copy()
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1.0e-8):
        bad = int(np.flatnonzero(norms[:, 0] < 1.0e-8)[0])
        raise ValueError(f"root quaternion at frame {bad} has near-zero norm")
    quaternions /= norms
    for frame_idx in range(1, len(quaternions)):
        if np.dot(quaternions[frame_idx - 1], quaternions[frame_idx]) < 0.0:
            quaternions[frame_idx] *= -1.0
    return quaternions


def quat_xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    return np.concatenate([quaternion[..., 3:4], quaternion[..., 0:3]], axis=-1)


def quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def quat_conjugate_wxyz(quaternion: np.ndarray) -> np.ndarray:
    return quaternion * np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float64)


def rotate_inverse_xyzw(quaternion_xyzw: np.ndarray, vector_world: np.ndarray) -> np.ndarray:
    """Rotate world-frame vectors into the local frame of quaternion_xyzw."""
    quaternion_wxyz = quat_xyzw_to_wxyz(quaternion_xyzw)
    vector_quaternion = np.concatenate(
        [np.zeros_like(vector_world[..., 0:1]), vector_world], axis=-1
    )
    vector_local = quat_multiply_wxyz(
        quat_multiply_wxyz(quat_conjugate_wxyz(quaternion_wxyz), vector_quaternion),
        quaternion_wxyz,
    )
    return vector_local[..., 1:4]


def rotation_vector_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Return the shortest axis-angle rotation vector for unit quaternions."""
    quaternion = quaternion.copy()
    quaternion *= np.where(quaternion[..., 0:1] < 0.0, -1.0, 1.0)
    xyz = quaternion[..., 1:4]
    xyz_norm = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(xyz_norm, np.clip(quaternion[..., 0], -1.0, 1.0))
    scale = np.empty_like(angle)
    small = xyz_norm < 1.0e-8
    scale[small] = 2.0
    scale[~small] = angle[~small] / xyz_norm[~small]
    return xyz * scale[..., None]


def slerp_xyzw(first: np.ndarray, second: np.ndarray, blend: float) -> np.ndarray:
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    if dot > 0.9995:
        result = first + blend * (second - first)
        return result / np.linalg.norm(result)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_angle = np.sin(angle)
    return (
        np.sin((1.0 - blend) * angle) / sin_angle * first
        + np.sin(blend * angle) / sin_angle * second
    )


def resample_motion(
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    joint_pos: np.ndarray,
    input_fps: float,
    output_fps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample onto an exactly uniform output time grid."""
    if input_fps <= 0.0 or output_fps <= 0.0:
        raise ValueError("input_fps and output_fps must be positive")
    if len(root_pos) < 2:
        raise ValueError("motion must contain at least two frames")

    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    duration = (len(root_pos) - 1) * input_dt
    output_times = np.arange(int(np.floor(duration / output_dt + 1.0e-9)) + 1) * output_dt
    input_coordinate = output_times * input_fps
    index_low = np.floor(input_coordinate).astype(np.int64)
    index_high = np.minimum(index_low + 1, len(root_pos) - 1)
    blend = input_coordinate - index_low

    resampled_root_pos = (
        root_pos[index_low] * (1.0 - blend[:, None])
        + root_pos[index_high] * blend[:, None]
    )
    resampled_joint_pos = (
        joint_pos[index_low] * (1.0 - blend[:, None])
        + joint_pos[index_high] * blend[:, None]
    )
    resampled_quat = np.stack(
        [
            slerp_xyzw(root_quat_xyzw[low], root_quat_xyzw[high], amount)
            for low, high, amount in zip(index_low, index_high, blend)
        ]
    )
    return resampled_root_pos, resampled_quat, resampled_joint_pos


def finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    edge_order = 2 if len(values) >= 3 else 1
    return np.gradient(values, dt, axis=0, edge_order=edge_order)


def angular_velocity_world(root_quat_xyzw: np.ndarray, dt: float) -> np.ndarray:
    """Compute world-frame angular velocity from body-to-world quaternions."""
    quat_wxyz = quat_xyzw_to_wxyz(root_quat_xyzw)
    angular_velocity = np.zeros((len(quat_wxyz), 3), dtype=np.float64)
    if len(quat_wxyz) == 2:
        relative = quat_multiply_wxyz(quat_wxyz[1], quat_conjugate_wxyz(quat_wxyz[0]))
        value = rotation_vector_from_wxyz(relative) / dt
        angular_velocity[:] = value
        return angular_velocity

    relative_middle = quat_multiply_wxyz(
        quat_wxyz[2:], quat_conjugate_wxyz(quat_wxyz[:-2])
    )
    angular_velocity[1:-1] = rotation_vector_from_wxyz(relative_middle) / (2.0 * dt)
    relative_first = quat_multiply_wxyz(
        quat_wxyz[1], quat_conjugate_wxyz(quat_wxyz[0])
    )
    relative_last = quat_multiply_wxyz(
        quat_wxyz[-1], quat_conjugate_wxyz(quat_wxyz[-2])
    )
    angular_velocity[0] = rotation_vector_from_wxyz(relative_first) / dt
    angular_velocity[-1] = rotation_vector_from_wxyz(relative_last) / dt
    return angular_velocity


def load_input(path: Path, input_fps: Optional[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if path.suffix.lower() == ".csv":
        data = np.loadtxt(path, delimiter=",", dtype=np.float64)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] != 7 + F2_NUM_JOINTS:
            raise ValueError(
                f"F2 CSV must have 21 columns (3 root position + 4 xyzw quaternion + "
                f"14 joints), got {data.shape[1]}"
            )
        if input_fps is None:
            raise ValueError("CSV input requires --input-fps")
        root_pos, root_quat, joint_pos = data[:, :3], data[:, 3:7], data[:, 7:]
        fps = input_fps
    elif path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as stream:
            data = pickle.load(stream)
        missing = {"root_pos", "root_rot", "dof_pos"} - set(data)
        if missing:
            raise KeyError(f"GMR pickle is missing keys: {sorted(missing)}")
        root_pos = np.asarray(data["root_pos"], dtype=np.float64)
        root_quat = np.asarray(data["root_rot"], dtype=np.float64)
        joint_pos = np.asarray(data["dof_pos"], dtype=np.float64)
        fps = input_fps if input_fps is not None else data.get("fps")
        if fps is None:
            raise ValueError("pickle has no 'fps'; provide --input-fps")
    else:
        raise ValueError("input must be a .csv, .pkl, or .pickle file")

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must have shape [T, 3], got {root_pos.shape}")
    if root_quat.ndim != 2 or root_quat.shape[1] != 4:
        raise ValueError(f"root quaternion must have shape [T, 4], got {root_quat.shape}")
    if joint_pos.ndim != 2 or joint_pos.shape[1] != F2_NUM_JOINTS:
        raise ValueError(f"joint_pos must have shape [T, 14], got {joint_pos.shape}")
    if not (len(root_pos) == len(root_quat) == len(joint_pos)):
        raise ValueError("root_pos, root_rot, and joint_pos must have the same frame count")
    if len(root_pos) < 2:
        raise ValueError("motion must contain at least two frames")
    if not all(np.all(np.isfinite(value)) for value in (root_pos, root_quat, joint_pos)):
        raise ValueError("input contains NaN or infinity")
    return root_pos, normalize_quaternions_xyzw(root_quat), joint_pos, float(fps)

#创建运动学模型
def load_f2_kinematics(gmr_root: Path):
    gmr_root = gmr_root.resolve()
    xml_path = gmr_root / "assets" / "F2_ZZ1_waiguan" / "x3_f2_14dof.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(f"F2 GMR XML not found: {xml_path}")
    if str(gmr_root) not in sys.path:
        sys.path.insert(0, str(gmr_root))
    try:
        from general_motion_retargeting.kinematics_model import KinematicsModel
    except ImportError as error:
        raise ImportError(
            f"cannot import GMR from {gmr_root}; check --gmr-root and its dependencies"
        ) from error

    kinematics = KinematicsModel(str(xml_path), device="cpu")
    if kinematics.num_dof != F2_NUM_JOINTS:
        raise ValueError(
            f"F2 kinematics XML has {kinematics.num_dof} DOFs; expected {F2_NUM_JOINTS}"
        )
    active_joint_bodies = tuple(
        kinematics.body_names[index]
        for index in range(1, kinematics.num_joint)
        if kinematics._joints[index].dof_dim > 0
    )
    if active_joint_bodies != F2_JOINT_BODY_NAMES:
        raise ValueError(
            "F2 kinematics DOF order does not match the AMP order:\n"
            f"expected={F2_JOINT_BODY_NAMES}\nactual={active_joint_bodies}"
        )
    return kinematics


def build_amp_frames(
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    joint_pos: np.ndarray,
    fps: float,
    kinematics,
) -> np.ndarray:
    dt = 1.0 / fps
    with torch.no_grad():
        body_pos_world, _ = kinematics.forward_kinematics(
            torch.from_numpy(root_pos).float(),
            torch.from_numpy(root_quat_xyzw).float(),
            torch.from_numpy(joint_pos).float(),
        )
    body_pos_world = body_pos_world.cpu().numpy().astype(np.float64)
    left_foot_idx = kinematics.get_body_idx("left_ankle_roll_link")
    right_foot_idx = kinematics.get_body_idx("right_ankle_roll_link")
    root_position_world = body_pos_world[:, 0]
    left_foot_relative_world = body_pos_world[:, left_foot_idx] - root_position_world
    right_foot_relative_world = body_pos_world[:, right_foot_idx] - root_position_world
    foot_pos_base = np.concatenate(
        [
            rotate_inverse_xyzw(root_quat_xyzw, left_foot_relative_world),
            rotate_inverse_xyzw(root_quat_xyzw, right_foot_relative_world),
        ],
        axis=1,
    )

    root_lin_vel_world = finite_difference(root_pos, dt)
    root_ang_vel_world = angular_velocity_world(root_quat_xyzw, dt)
    root_lin_vel_base = rotate_inverse_xyzw(root_quat_xyzw, root_lin_vel_world)
    root_ang_vel_base = rotate_inverse_xyzw(root_quat_xyzw, root_ang_vel_world)
    joint_vel = finite_difference(joint_pos, dt)

    frames = np.concatenate(
        [
            root_pos,
            root_quat_xyzw,
            joint_pos,
            foot_pos_base,
            root_lin_vel_base,
            root_ang_vel_base,
            joint_vel,
        ],
        axis=1,
    )
    if frames.shape[1] != AMP_FRAME_SIZE:
        raise AssertionError(f"internal error: expected 47 columns, got {frames.shape[1]}")
    if not np.all(np.isfinite(frames)):
        raise ValueError("generated AMP frames contain NaN or infinity")
    return frames


def write_amp_json(path: Path, frames: np.ndarray, fps: float, motion_weight: float) -> None:
    if motion_weight <= 0.0:
        raise ValueError("motion weight must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_duration = 1.0 / fps
    document: Dict[str, object] = {
        "LoopMode": "Wrap",
        "FrameDuration": frame_duration,
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": True,
        "TotalTime": (len(frames) - 1) * frame_duration,
        "MotionWeight": motion_weight,
        "Frames": frames.tolist(),
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def default_gmr_root() -> Path:
    return Path(__file__).resolve().parents[2].parent / "GMR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 14-DOF GMR F2 CSV/PKL motion to the 47-column AMP TXT format."
    )
    parser.add_argument("--input", type=Path, required=True, help="GMR .csv or .pkl input")
    parser.add_argument("--output", type=Path, required=True, help="output AMP .txt path")
    parser.add_argument(
        "--input-fps",
        type=float,
        default=None,
        help="input FPS; required for CSV, optional override for PKL",
    )
    parser.add_argument("--output-fps", type=float, default=100.0, help="output FPS (default: 100)")
    parser.add_argument("--motion-weight", type=float, default=1.0)
    parser.add_argument(
        "--gmr-root",
        type=Path,
        default=default_gmr_root(),
        help="GMR repository root used for F2 forward kinematics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_pos, root_quat, joint_pos, input_fps = load_input(args.input, args.input_fps)
    root_pos, root_quat, joint_pos = resample_motion(
        root_pos, root_quat, joint_pos, input_fps, args.output_fps
    )
    kinematics = load_f2_kinematics(args.gmr_root)
    frames = build_amp_frames(root_pos, root_quat, joint_pos, args.output_fps, kinematics)
    write_amp_json(args.output, frames, args.output_fps, args.motion_weight)

    print(f"input:  {args.input} ({input_fps:g} Hz)")
    print(f"output: {args.output} ({args.output_fps:g} Hz)")
    print(f"frames: {frames.shape[0]} x {frames.shape[1]}")
    print("AMP observation: joint_pos(14) + root_lin_vel_base(3) + "
          "root_ang_vel_base(3) + joint_vel(14) = 34")


if __name__ == "__main__":
    main()
