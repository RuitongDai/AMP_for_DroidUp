from isaacgym.torch_utils import quat_rotate_inverse
import glob
import json
import torch
import numpy as np
from pybullet_utils import transformations
from .utils import quaternion_slerp
from humanoid.algo.utils import pose3d
from humanoid.algo.utils import motion_util

class AMPLoader:
    POS_SIZE = 3  # 位置维度(x, y, z)
    ROT_SIZE = 4  # 旋转维度(四元数)
    JOINT_POS_SIZE = 14  # 关节位置维度(14个关节)
    FOOT_POS_SIZE = 6  # 足部位置维度(2只脚 × 3个坐标)
    LINEAR_VEL_SIZE = 3  # 线速度维度(x, y, z)
    ANGULAR_VEL_SIZE = 3  # 角速度维度(x, y, z)
    JOINT_VEL_SIZE = 14  # 关节速度维度(14个关节)

    # 根节点位置索引范围
    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE

    # 根节点旋转索引范围
    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE

    # 关节位置索引范围
    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    # 足部位置索引范围
    FOOT_POSE_START_IDX = JOINT_POSE_END_IDX
    FOOT_POSE_END_IDX = FOOT_POSE_START_IDX + FOOT_POS_SIZE

    # 线速度索引范围
    LINEAR_VEL_START_IDX = FOOT_POSE_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    # 角速度索引范围
    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    # 关节速度索引范围
    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    def __init__(
            self,
            device,
            time_between_frames,
            data_dir="",
            preload_transitions=False,
            num_preload_transitions=1000000,
            motion_files=glob.glob("datasets/motion_amp_expert/*"),
    ):
        """从动作捕捉数据集中提供AMP观察数据。

        参数:
            device: 计算设备(CPU或GPU)
            time_between_frames: 相邻帧之间的时间间隔(秒)
            data_dir: 数据目录路径
            preload_transitions: 是否预加载转移数据
            num_preload_transitions: 预加载的转移数据数量
            motion_files: 运动文件列表
        """
        self.device = device
        self.time_between_frames = time_between_frames

        # 存储每个轨迹的数据
        self.trajectories = []  # 不含根节点位置和旋转的轨迹
        self.trajectories_full = []  # 完整轨迹数据
        self.trajectory_names = []  # 轨迹名称列表
        self.trajectory_idxs = []  # 轨迹索引列表
        self.trajectory_lens = []  # 轨迹长度(秒)
        self.trajectory_weights = []  # 轨迹采样权重
        self.trajectory_frame_durations = []  # 每帧时长
        self.trajectory_num_frames = []  # 每条轨迹的帧数
        self.amp_obs_dim = 0

        # 逐个加载运动文件
        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])
                motion_data = self.data_process(motion_data)  # 数据预处理

                # 规范化并标准化四元数
                for f_i in range(motion_data.shape[0]):
                    root_rot = AMPLoader.get_root_rot(motion_data[f_i])
                    root_rot = pose3d.QuaternionNormalize(root_rot)  # 归一化四元数
                    root_rot = motion_util.standardize_quaternion(root_rot)  # 标准化四元数
                    motion_data[f_i, AMPLoader.POS_SIZE : (AMPLoader.POS_SIZE + AMPLoader.ROT_SIZE)] = root_rot

                # 提取不包含根节点位置和旋转的观察数据(用于AMP)
                self.trajectories.append(
                    torch.tensor(motion_data[:,AMPLoader.ROOT_ROT_END_IDX:AMPLoader.JOINT_VEL_END_IDX],
                                 dtype=torch.float32, device=device)
                )
                # 保存完整的轨迹数据
                self.trajectories_full.append(
                    torch.tensor(motion_data[:, :AMPLoader.JOINT_VEL_END_IDX],
                    dtype=torch.float32, device=device)
                )
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))  # 从JSON读取权重
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration  # 计算轨迹总长度
                print(f"traj_len:{traj_len}")
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # 归一化轨迹权重用于加权采样
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # 预加载转移数据(当前帧和下一帧)
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f'Preloading {num_preload_transitions} transitions')
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)  # 按权重采样轨迹
            times = self.traj_time_sample_batch(traj_idxs)  # 为每条轨迹采样时间

            # 获取当前状态和下一个状态
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print(f'Finished preloading')

        # 将所有完整轨迹堆叠成一个张量
        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def data_process(self, motion_data):
        """数据预处理：从原始动作数据中提取各个分量"""
        root_pos = AMPLoader.get_root_pos_batch(motion_data)  # 提取根节点位置
        root_rot = AMPLoader.get_root_rot_batch(motion_data)  # 提取根节点旋转
        gravity_vec = np.array([0., 0., -1.], dtype=np.float32)  # 重力向量(未使用)

        joint_pos = AMPLoader.get_joint_pose_batch(motion_data)  # 提取关节位置
        # joint_pos[:, [4, 10]] = 0.  # 可选：将某些关节位置设为0

        foot_pos = AMPLoader.get_foot_pose_batch(motion_data)  # 提取足部位置

        lin_vel = AMPLoader.get_linear_vel_batch(motion_data)  # 提取线速度
        ang_vel = AMPLoader.get_angular_vel_batch(motion_data)  # 提取角速度

        joint_vel = AMPLoader.get_joint_vel_batch(motion_data)  # 提取关节速度
        # joint_vel[:, [4, 10]] = 0.  # 可选：将某些关节速度设为0

        # 将所有分量水平堆叠成完整的动作向量
        return np.hstack([root_pos, root_rot, joint_pos, foot_pos, lin_vel, ang_vel, joint_vel])

    def weighted_traj_idx_sample(self):
        """按权重随机采样一条轨迹索引"""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """批量按权重采样轨迹索引"""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """为单条轨迹采样随机时间点"""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """为多条轨迹批量采样时间点"""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, frame1, frame2, blend):
        """线性插值混合两个帧(用于位置、速度等)"""
        return (1.0 - blend) * frame1 + blend * frame2

    def get_trajectory(self, traj_idx):
        """返回指定轨迹的完整观察数据"""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """获取指定轨迹在指定时间的帧(插值)"""
        p = float(time) / self.trajectory_lens[traj_idx]  # 计算相对位置(0-1)
        n = self.trajectories[traj_idx].shape[0]  # 总帧数
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))  # 前后帧索引
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low  # 插值系数(0-1)
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """批量获取指定时间的帧"""
        p = times / self.trajectory_lens[traj_idxs]  # 计算相对位置
        n = self.trajectory_num_frames[traj_idxs]  # 各轨迹的总帧数
        idx_low, idx_high = np.floor(p * n).astype(np.int64), np.ceil(p * n).astype(np.int64)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)

        # 为每条轨迹提取对应的帧
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx  # 标记属于该轨迹的样本
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]

        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """获取指定轨迹在指定时间的完整帧(包括根节点位置和旋转)"""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        """批量获取完整帧，使用专门的混合方法处理旋转"""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int64), np.ceil(p * n).astype(np.int64)

        # 预创建张量以存储位置、旋转和AMP观察分量
        all_frame_pos_starts = torch.zeros(len(traj_idxs), AMPLoader.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), AMPLoader.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), AMPLoader.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), AMPLoader.ROT_SIZE, device=self.device)
        all_frame_amp_starts = torch.zeros(len(traj_idxs), AMPLoader.JOINT_VEL_END_IDX - AMPLoader.JOINT_POSE_START_IDX, device=self.device)
        all_frame_amp_ends = torch.zeros(len(traj_idxs), AMPLoader.JOINT_VEL_END_IDX - AMPLoader.JOINT_POSE_START_IDX, device=self.device)

        # 为每条轨迹提取对应的数据分量
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = AMPLoader.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = AMPLoader.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = AMPLoader.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = AMPLoader.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_VEL_END_IDX]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_VEL_END_IDX]

        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        # 分别对位置、旋转和关节数据进行插值
        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)  # 四元数球面线性插值
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)

        # 拼接所有分量
        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def get_frame(self):
        """返回随机采样的观察帧"""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """返回随机采样的完整帧"""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        """批量获取完整帧"""
        if self.preload_transitions:
            # 如果预加载了数据，直接从预加载数据中采样
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            # 否则在线采样和插值
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """线性插值两个帧，包括位置和旋转。

        参数:
            frame0: 第一个帧(blend=0时对应)
            frame1: 第二个帧(blend=1时对应)
            blend: 0-1之间的插值系数，指定两个帧之间的插值位置

        返回:
            两个帧的插值结果
        """
        # 分别提取两个帧的各个分量
        root_pos0, root_pos1 = AMPLoader.get_root_pos(frame0), AMPLoader.get_root_pos(frame1)
        root_rot0, root_rot1 = AMPLoader.get_root_rot(frame0), AMPLoader.get_root_rot(frame1)
        joints0, joints1 = AMPLoader.get_joint_pose(frame0), AMPLoader.get_joint_pose(frame1)
        foot_pos0, foot_pos1 = AMPLoader.get_foot_pose(frame0), AMPLoader.get_foot_pose(frame1)
        linear_vel_0, linear_vel_1 = AMPLoader.get_linear_vel(frame0), AMPLoader.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = AMPLoader.get_angular_vel(frame0), AMPLoader.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = AMPLoader.get_joint_vel(frame0), AMPLoader.get_joint_vel(frame1)

        # 对各分量进行插值
        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = transformations.quaternion_slerp(
            root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend)
        blend_root_rot = torch.tensor(
            motion_util.standardize_quaternion(blend_root_rot),
            dtype=torch.float32, device=self.device)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_foot_pos = self.slerp(foot_pos0, foot_pos1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        # 拼接所有插值后的分量成完整帧
        return torch.cat([
            blend_root_pos, blend_root_rot, blend_joints, blend_foot_pos,
            blend_linear_vel, blend_angular_vel, blend_joints_vel])

    def compute_obs_dim(self):
        """计算观察维度(不包括根节点位置和旋转)"""
        s = self.preloaded_s[0, AMPLoader.JOINT_POSE_START_IDX: AMPLoader.JOINT_VEL_END_IDX]
        return s.shape[0] - 6  # 减去6维

    def _build_amp_observation(self, full_frames):
        """从完整 motion frame 构造与策略侧一致的 AMP 观测。

        motion 文件中的根线速度和根角速度是世界坐标系；
        AMP 策略侧使用机体局部坐标系，因此只在这里转换。
        """

        # 根姿态，格式为 Isaac Gym 使用的 xyzw
        root_quat = AMPLoader.get_root_rot_batch(full_frames)

        # 防御性归一化，quat_rotate_inverse 要求单位四元数
        quat_norm = torch.linalg.vector_norm(
            root_quat,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)

        root_quat = root_quat / quat_norm

        # 关节状态不涉及世界系/机体系转换
        joint_pos = AMPLoader.get_joint_pose_batch(full_frames)
        joint_vel = AMPLoader.get_joint_vel_batch(full_frames)

        # motion 文件中保存的是世界坐标系速度
        root_lin_vel_world = AMPLoader.get_linear_vel_batch(
            full_frames
        )
        root_ang_vel_world = AMPLoader.get_angular_vel_batch(
            full_frames
        )

        # 世界坐标系 → 机体局部坐标系
        root_lin_vel_body = quat_rotate_inverse(
            root_quat,
            root_lin_vel_world,
        )

        root_ang_vel_body = quat_rotate_inverse(
            root_quat,
            root_ang_vel_world,
        )

        # 必须和 x3_f2_env.py 的策略侧 AMP 顺序完全相同
        return torch.cat(
            (
                joint_pos,  # 14
                root_lin_vel_body,  # 3
                root_ang_vel_body,  # 3
                joint_vel,  # 14
            ),
            dim=-1,
        )

    def feed_forward_generator(
            self,
            num_mini_batch,
            mini_batch_size,
    ):
        """生成坐标系对齐后的专家 AMP transition。"""

        for _ in range(num_mini_batch):
            idxs = np.random.choice(
                self.preloaded_s.shape[0],
                size=mini_batch_size,
            )

            # 完整 motion frame，里面仍保存世界系根速度
            full_frames = self.preloaded_s[idxs]
            full_frames_next = self.preloaded_s_next[idxs]

            # 只在构造 AMP 观测时转换为机体系
            s = self._build_amp_observation(full_frames)
            s_next = self._build_amp_observation(
                full_frames_next
            )

            yield s, s_next

    @property
    def observation_dim(self):
        """AMP观察的维度"""
        return self.compute_obs_dim()

    @property
    def num_motions(self):
        """已加载的运动数据数量"""
        return len(self.trajectory_names)

    # 以下是静态方法用于提取帧中的各个分量
    def get_root_pos(pose):
        """提取单个帧的根节点位置"""
        return pose[AMPLoader.ROOT_POS_START_IDX : AMPLoader.ROOT_POS_END_IDX]

    def get_root_pos_batch(poses):
        """提取批量帧的根节点位置"""
        return poses[:, AMPLoader.ROOT_POS_START_IDX : AMPLoader.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        """提取单个帧的根节点旋转"""
        return pose[AMPLoader.ROOT_ROT_START_IDX : AMPLoader.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        """提取批量帧的根节点旋转"""
        return poses[:, AMPLoader.ROOT_ROT_START_IDX : AMPLoader.ROOT_ROT_END_IDX]

    def get_joint_pose(pose):
        """提取单个帧的关节位置"""
        return pose[AMPLoader.JOINT_POSE_START_IDX : AMPLoader.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        """提取批量帧的关节位置"""
        return poses[:, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.JOINT_POSE_END_IDX]

    def get_foot_pose(pose):
        """提取单个帧的足部位置"""
        return pose[AMPLoader.FOOT_POSE_START_IDX : AMPLoader.FOOT_POSE_END_IDX]

    def get_foot_pose_batch(poses):
        """提取批量帧的足部位置"""
        return poses[:, AMPLoader.FOOT_POSE_START_IDX : AMPLoader.FOOT_POSE_END_IDX]

    def get_linear_vel(pose):
        """提取单个帧的线速度"""
        return pose[AMPLoader.LINEAR_VEL_START_IDX : AMPLoader.LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(poses):
        """提取批量帧的线速度"""
        return poses[:, AMPLoader.LINEAR_VEL_START_IDX : AMPLoader.LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        """提取单个帧的角速度"""
        return pose[AMPLoader.ANGULAR_VEL_START_IDX : AMPLoader.ANGULAR_VEL_END_IDX]

    def get_angular_vel_batch(poses):
        """提取批量帧的角速度"""
        return poses[:, AMPLoader.ANGULAR_VEL_START_IDX : AMPLoader.ANGULAR_VEL_END_IDX]

    def get_joint_vel(pose):
        """提取单个帧的关节速度"""
        return pose[AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        """提取批量帧的关节速度"""
        return poses[:, AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]