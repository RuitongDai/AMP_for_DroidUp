import os
import time
from typing import Union

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from collections import deque

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.utils.math import quat_apply_yaw, wrap_to_pi,quat_yaw
from humanoid.utils.helpers import class_to_dict
from .x3_f2_cfg_amp import X3F2AMPCfg
import torch
from humanoid.envs.base.base_task import BaseTask
from humanoid.utils.terrain import Terrain
from humanoid.algo.utils.motion_loader_x3 import AMPLoader
from copy import deepcopy

def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

class X3F2Env(BaseTask):
    def __init__(self, cfg: Union[X3F2AMPCfg], sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg:Union[X3F2AMPCfg] = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = getattr(self.cfg.viewer, "debug_viz", True)
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self._init_random_motor_paras()
        self.init_done = True
        # -- amp motion show
        if getattr(self.cfg.env, "amp_motion_files_display", False):
            self._init_amp_motion()
        self.reset()

    # =================================================== MDP（step(action) observation reward reset） ========================================================
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.actions[:] = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions).to(self.device)

        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            action_delayed = self.update_cmd_action_latency_buffer()
            self.torques[:] = self._compute_torques(action_delayed).view(self.torques.shape)
            # -- 计算手臂力矩
            if self.arm_dof_enable:
                self.arm_torques[:] = self._compute_arm_torques()
                all_torques = torch.cat([self.torques, self.arm_torques], dim=-1)
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(all_torques.clone()))
            else:
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        reset_env_ids, terminal_amp_states, termination_privileged_obs = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)


        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, reset_env_ids, terminal_amp_states

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        p_gains = self.p_gains * self.kp_factor
        d_gains = self.d_gains * self.kd_factor
        torques = p_gains * (
                    actions_scaled + self.default_dof_pos - self.dof_pos - self.motor_offset) - d_gains * self.dof_vel
        torques = torques * self.motor_strength
        return torch.clip(torques, -self.q_torque_limits, self.q_torque_limits)

    def _compute_arm_torques(self):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        if self.cfg.domain_rand.randomize_arm_pos and  ((self.common_step_counter-1) % self.cfg.domain_rand.arm_pos_interval == 0):
            dr = self.cfg.domain_rand
            for i in range(len(dr.min_arm_pos)):
                self.arm_actions[:, i] = torch.empty_like(self.arm_actions[:, i]).uniform_(dr.min_arm_pos[i], dr.max_arm_pos[i])

        self.arm_actions_fil = 0.98 * self.arm_actions_fil + 0.02 * self.arm_actions
        torques = self.arm_kp * (self.arm_actions_fil - self.arm_pos) - self.arm_kd * self.arm_vel
        return torch.clip(torques, -self.arm_torque_limit, self.arm_torque_limit)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        # ============================ prepare quantities ==================================
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_euler_xyz[:] = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # -- 计算feet contact
        self.feet_forces[:] = self.contact_forces[:, self.feet_indices]
        self.contact[:] = self.feet_forces[:, :, 2] > 1.
        self.contact_filt[:] = torch.logical_or(self.contact, self.last_contact)
        self.feet_forces_history[:] = torch.roll(self.feet_forces_history, shifts=-1, dims=1)
        self.feet_forces_history[:, -1] = self.feet_forces
        self.feet_vel_history[:] = torch.roll(self.feet_vel_history, shifts=-1, dims=1)
        self.feet_vel_history[:, -1] = self.rigid_body_vel[:,self.feet_indices]
        # -- 计算feet contact air time
        self.first_contact[:] = (self.current_air_time > 0.) * self.contact_filt
        self.current_air_time[:] += self.dt
        self.current_air_time *= ~self.contact_filt
        self.current_contact_time[:] += self.dt
        self.current_contact_time[:] *= self.contact_filt
        # -- terrain_mask
        self._get_terrain_indices_from_pos()
        # -- vel mask
        self.forward_motion_mask[:] = self.commands[:, 0] > 0.1
        # ============================ Event ==================================
        self._post_physics_step_callback()

        # ============================ compute observations, rewards, resets, ... ==========
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_obs_for_expert_trans()[env_ids]
        termination_privileged_obs = self.compute_termination_privileged_obs(env_ids)
        self.reset_idx(env_ids)

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        # ============ hist data save ==========
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_rigid_body_vel[:] = self.rigid_body_vel
        self.last_contact[:] = self.contact[:]
        self.last_torques[:] = self.torques[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz and not self.headless:
            self._draw_debug_vis()

        return env_ids, terminal_amp_states, termination_privileged_obs

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # =================================== 更新步态指令 =======================================
        if self.cfg.rewards.gait_radio:
            self._gait_style_update()
        self._phase_step_update()
        # =================================== 更新速度指令 =======================================
        if self.cfg.commands.gait_enable:
            self._resample_gait_commands()
        else:
            env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
            self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), self.cfg.commands.ranges.ang_vel_yaw[0], self.cfg.commands.ranges.ang_vel_yaw[1])
            self.commands[self.stand_flg, 2] = 0.
        self._zero_small_commands()
        # =================================== 更新地形高度图 =======================================
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            self.left_feet_height_maps = self._get_feet_heights(feet_idx=0)
            self.right_feet_height_maps = self._get_feet_heights(feet_idx=1)
        # self.left_feet_hold_maps = self._get_feet_hold(feet_idx=0)
        # self.right_feet_hold_maps = self._get_feet_hold(feet_idx=1)
        # self.base_height_maps = self._get_base_heights()
        # =================================== 更新外部推力 =======================================
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        else:
            self.rand_push_force.zero_()
            self.rand_push_torque.zero_()

        if self.cfg.domain_rand.small_push_robots and  (self.common_step_counter % self.cfg.domain_rand.small_push_interval == 0):
            self._small_push_robots()
        else:
            self.rand_small_push_force.zero_()
            self.rand_small_push_torque.zero_()

        if self.cfg.domain_rand.apply_force_torque:
            self._apply_force_torque()

    def compute_observations(self):

        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
        sin_pos[self.stand_command] = 0.
        cos_pos[self.stand_command] = 0.

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 1.

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        privileged_obs_buf = torch.cat([
            sin_pos * self.cfg.rewards.clock_enable,
            cos_pos * self.cfg.rewards.clock_enable,
            self.commands[:, :3] * self.commands_scale,
            q,  # 10
            dq,  # 10
            self.actions,  # 10
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz[:, :2] * self.obs_scales.quat,  # 2
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.env_frictions,  # 1
            self.body_mass / 12.,  # 1
            stance_mask,  # 2
            contact_mask,  # 2
        ], dim=-1)

        obs_buf = torch.cat([
            sin_pos * self.cfg.rewards.clock_enable,
            cos_pos * self.cfg.rewards.clock_enable,
            self.commands[:, :3] * self.commands_scale,
            q,  # 14
            dq,  # 14
            self.actions,  # 14
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz[:, :2] * self.obs_scales.quat,  # 2
        ], dim=-1)

        if self.cfg.terrain.measure_heights:
            terrain_height_maps = (torch.clip(self.root_states[:, 2].unsqueeze(1) - self.measured_heights - self.cfg.normalization.height_offset, -1, 1.)
                                * self.obs_scales.height_measurements)

            left_height_maps = (torch.clip(self.root_states[:, 2].unsqueeze(1) - self.left_feet_height_maps - self.cfg.normalization.height_offset, -1, 1.)
                                * self.obs_scales.height_measurements)

            right_height_maps = (torch.clip(self.root_states[:, 2].unsqueeze(1) - self.right_feet_height_maps - self.cfg.normalization.height_offset, -1, 1.)
                                * self.obs_scales.height_measurements)

            privileged_obs_buf = torch.cat((privileged_obs_buf, left_height_maps, right_height_maps, terrain_height_maps), dim=-1)

        if self.add_noise:
            obs_now = obs_buf + (torch.empty_like(obs_buf).uniform_(-1., 1.)) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf
        self.obs_history.append(obs_now)
        self.critic_history.append(privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def compute_termination_privileged_obs(self, env_ids):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
        sin_pos[self.stand_command] = 0.
        cos_pos[self.stand_command] = 0.

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 1.

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        privileged_obs_buf = torch.cat((
            # self.command_input,  # 2 + 3
            sin_pos,
            cos_pos,
            self.commands[:, :3] * self.commands_scale,
            q,# 10
            dq,# 10
            self.actions,  # 10
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz[:, :2] * self.obs_scales.quat,  # 2

            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            # dim = 6 可选配
            self.env_frictions,  # 1
            self.body_mass / 12., # 1
            stance_mask, # 2
            contact_mask, # 2
        ), dim=-1)
        privileged_termination_obs_buf = self.critic_history.copy()
        privileged_termination_obs_buf.pop()
        privileged_termination_obs_buf.append(privileged_obs_buf)
        privileged_obs_buf_cat = torch.cat([privileged_termination_obs_buf[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

        return privileged_obs_buf_cat[env_ids]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        roll_cutoff = torch.abs(self.base_euler_xyz[:,0]) > 1.3
        pitch_cutoff = torch.abs(self.base_euler_xyz[:,1]) > 1.3
        height_cutoff = self.root_states[:, 2] < -5.
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        self.reset_buf |= height_cutoff

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def reset(self):
        """ Reset all robots"""
        obs, privileged_obs, _, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return obs, privileged_obs

    def reset_idx(self, env_ids: torch) -> None:
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        self.reset_env_ids = env_ids
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            # self._update_terrain_curriculum(env_ids)
            self._update_terrain_curriculum_vel(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        if getattr(self.cfg.env, "reference_state_initialization", False):
            frames = self.amp_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_amp(env_ids, frames)
            self._reset_root_states_amp(env_ids, frames)
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

        # Randomize joint parameters, like torque gain friction ...
        self.randomize_dof_props(env_ids)

        # reset buffers
        self.last_last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_contact[env_ids] = False
        self.last_root_vel[env_ids] = 0.
        self.last_rigid_body_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.current_air_time[env_ids] = 0.
        self.current_contact_time[env_ids] = 0.
        self.feet_both_contact_time[env_ids] = 0.
        self.feet_forces_history[env_ids] = 0.
        self.feet_vel_history[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.commands[env_ids] = 0.
        self.phase[env_ids] = 0.
        # resample command
        if self.cfg.commands.gait_enable:
            self.generate_gait_time(env_ids)
            self._resample_gait_commands()
        else:
            self._resample_commands(env_ids)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.rewards.penalize_curriculum:
            self.extras["episode"]["curriculum"] = self.curriculum_scale
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0
        # reset latency buffers and randomization
        self._reset_latency_buffer(env_ids)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if len(env_ids) == 0:
            return
        dr = self.cfg.domain_rand
        self.dof_pos[env_ids] = torch.empty_like(self.dof_pos[env_ids]).uniform_(dr.joint_pos_range[0], dr.joint_pos_range[1]) * self.default_dof_pos
        self.dof_vel[env_ids] = torch.empty_like(self.dof_vel[env_ids]).uniform_(dr.joint_vel_range[0], dr.joint_vel_range[1])
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        if len(env_ids) == 0:
            return
        dr = self.cfg.domain_rand
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch.empty_like(self.root_states[env_ids, :2]).uniform_(dr.pose_xy[0], dr.pose_xy[1])
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        roll = torch.empty(len(env_ids), device=self.device).uniform_(-0., 0.)
        pitch = torch.empty(len(env_ids), device=self.device).uniform_(-0., 0.)
        yaw = torch.empty(len(env_ids), device=self.device).uniform_(dr.pose_yaw[0], dr.pose_yaw[1])
        base_quat = quat_from_euler_xyz(roll, pitch, yaw)
        self.root_states[env_ids, 3:7] = base_quat  # [3:7]: base quat
        # base velocities
        self.root_states[env_ids, 7:10] = torch.empty_like(self.root_states[env_ids, 7:10]).uniform_(dr.lin_vel[0], dr.lin_vel[1])  # linear vel
        self.root_states[env_ids, 10:13] = torch.empty_like(self.root_states[env_ids, 10:13]).uniform_(dr.ang_vel[0], dr.ang_vel[1])  # angular vel
        if self.cfg.asset.fix_base_link:
            self.root_states[env_ids, 7:13] = 0
            self.root_states[env_ids, 2] += 0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.dof_pos[env_ids] = AMPLoader.get_joint_pose_batch(frames)
        self.dof_vel[env_ids] = AMPLoader.get_joint_vel_batch(frames)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_amp(self, env_ids, frames):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        dr = self.cfg.domain_rand
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch.empty_like(self.root_states[env_ids, :2]).uniform_(dr.pose_xy[0], dr.pose_xy[1])
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:10] = AMPLoader.get_linear_vel_batch(frames)

        if self.cfg.asset.fix_base_link:
            self.root_states[env_ids, 7:13] = 0
            self.root_states[env_ids, 2] += 0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    # ==================================================== Env Init, Simulation Create、Terrain Create ===============================================
    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.small_push_interval = np.ceil(self.cfg.domain_rand.small_push_interval_s / self.dt)
        self.cfg.domain_rand.apply_interval = np.ceil(self.cfg.domain_rand.apply_interval_s / self.dt)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            # self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _arm_enable_check(self):
        # -- 加载模型时判断是否有手臂自由度
        self.arm_dof_enable = self.num_actions < self.num_dof
        self.num_arms = self.num_dof - self.num_actions
        # self.cfg.env.num_arms
        print("\n============  arm_dof_enable ============")
        print("arm_dof_enable: ", self.arm_dof_enable)
        print("num_arms      : ", self.num_arms)
        print("============  arm_dof_enable ============\n")

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.robot_asset = robot_asset
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        # -- check arm dof
        self._arm_enable_check()

        # body and joint paras
        self.p_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.d_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.joint_damping = torch.zeros(self.num_actions, device=self.device)
        self.joint_armature = torch.zeros(self.num_actions, device=self.device)
        self.joint_friction = torch.zeros(self.num_actions, device=self.device)
        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.default_dof_pos = torch.zeros(self.num_actions, dtype=torch.float, device=self.device)

        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.body_names = body_names
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        knee_names = [s for s in body_names if self.cfg.asset.knee_name in s]
        penalized_contact_names = []
        termination_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # find body idx
        self.feet_indices = np.zeros(len(feet_names), dtype=np.longlong)
        self.knee_indices = np.zeros(len(knee_names), dtype=np.longlong)
        self.torso_index = self.gym.find_asset_rigid_body_index(
            robot_asset,
            self.cfg.asset.torso_name,
        )

        if self.torso_index < 0:
            raise RuntimeError(
                f"Cannot find torso body: {self.cfg.asset.torso_name}"
            )
        self.penalised_contact_indices = np.zeros(len(penalized_contact_names), dtype=np.longlong)
        self.termination_contact_indices = np.zeros(len(termination_contact_names), dtype=np.longlong)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, knee_names[i])
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, penalized_contact_names[i])
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, termination_contact_names[i])

        # joint positions offsets and PD gains
        for i in range(self.num_actions):
            name = self.dof_names[i]
            # print(name)
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():

                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    self.joint_damping[i] = self.cfg.control.joint_damping[dof_name]
                    self.joint_armature[i] = self.cfg.control.joint_armature[dof_name]
                    self.joint_friction[i] = self.cfg.control.joint_friction[dof_name]
                    found = True
            if not found:
                self.p_gains[:, i] = 0.
                self.d_gains[:, i] = 0.
                self.joint_damping[i] = 0.
                self.joint_armature[i] = 0.
                self.joint_friction[i] = 0.
                print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self._show_default_joint_paras()

        # init states
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        # env actor
        self.actor_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(self.cfg.domain_rand.pose_xy[0], self.cfg.domain_rand.pose_xy[1], (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

    def _show_default_joint_paras(self):
        print("\n===================== default joint paras =====================")
        print(f"{'Joint':<30} {'kp':>6} {'kd':>6} {'damping':>10} {'armature':>10} {'friction':>10} {'default_pos':>10}")
        print("-" * 75)
        # 打印每个关节
        for i, name in enumerate(self.dof_names):
            if i >= self.num_actions:
                break
            # kp/kd 从第一个环境取即可（假设每个env相同）
            kp = self.p_gains[0, i].item() if self.p_gains.numel() > 0 else 0.0
            kd = self.d_gains[0, i].item() if self.d_gains.numel() > 0 else 0.0

            # damping, armature, friction
            damping = self.joint_damping[i].item()
            armature = self.joint_armature[i].item()
            friction = self.joint_friction[i].item()

            # default dof pos
            default_pos = self.default_dof_pos[0][i].item()
            print(f"{name:<30} {kp:>6.1f} {kd:>6.1f} {damping:>10.3f} {armature:>10.4f} {friction:>10.4f} {default_pos:>10.3f}")
        print("================================================================\n")

    def _show_robot_paras(self, body_props):
        print("\n===================== body & joint info =====================")
        print(f"{'Index':<6} {'Body Name':<25} {'Mass(kg)':>10}")
        print("-" * 50)
        for i in range(self.num_bodies):
            print(f"{i:<6} {self.body_names[i]:<25} {body_props[i].mass:>10.4f}")

        print("\n" + f"{'Index':<6} {'Joint Name':<25}")
        print("-" * 40)
        for i in range(self.num_dof):
            print(f"{i:<6} {self.dof_names[i]:<25}")
        print("===================== end =====================\n")

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        # -- robot terrain level, type, mask
        self.level_idx = torch.ones(self.num_envs,dtype=torch.long,device=self.device)
        self.type_idx = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.terrain_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.plane_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.slope_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.step_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        # -- robot terrain level, type, mask
        self.level_idx = torch.ones(self.num_envs,dtype=torch.long,device=self.device)
        self.type_idx = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.terrain_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.plane_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.slope_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.step_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = np.clip(self.cfg.terrain.max_init_terrain_level, 0, self.cfg.terrain.num_rows)
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows
            self.terrain_levels = torch.randint(0, max_init_level, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _get_terrain_indices_from_pos(self):
        """
        根据机器人世界坐标 (base_pos)，计算对应的地形索引。
        同时区分是平面 (plane) 还是地形 (terrain)。
        Returns:
            level_idx (Tensor): 每个环境的 terrain level 索引
            type_idx  (Tensor): 每个环境的 terrain type 索引
            terrain_mask (Tensor): 0 表示 plane, 1 表示 terrain
        """
        if not self.cfg.terrain.curriculum:
            return
        # 获取配置
        num_rows = self.terrain.cfg.num_rows  # 行 (level)
        num_cols = self.terrain.cfg.num_cols  # 列 (type)
        env_length = self.terrain.env_length  # 每块地形长度
        env_width = self.terrain.env_width  # 每块地形宽度

        # --- 计算行列索引 ---
        self.level_idx = torch.floor(self.base_pos[:, 0] / env_length)
        self.type_idx = torch.floor(self.base_pos[:, 1] / env_width)
        # -- plane设为-1
        self.type_idx[self.type_idx >= num_cols] = -1
        self.type_idx[self.type_idx <= -1]       = -1
        type_mask_plane = self.type_idx.clone()
        if self.terrain.plane_type_idx:
            # 将 plane 对应环境 mask 置 -1
            plane_mask = torch.isin(type_mask_plane, torch.tensor(self.terrain.plane_type_idx, device=self.device))
            type_mask_plane[plane_mask] = -1
        if self.terrain.slope_type_idx:
            # 将 slope 对应环境 mask 置 -2
            slope_mask = torch.isin(type_mask_plane, torch.tensor(self.terrain.slope_type_idx, device=self.device))
            type_mask_plane[slope_mask] = -2
        # --- 判断是否为 terrain ---
        # terrain_mask: 1 表示在step地形区间, 0 表示和在 plane\slope
        # type_idx >= 0, type_idx < (num_cols)
        # level_idx >= 0, level_idx < (num_rows)
        self.terrain_mask[:] = (
                (type_mask_plane >= 0) & (type_mask_plane < num_cols) &
                (self.level_idx >= 2) & (self.level_idx < num_rows)
        )
        self.step_mask[:] = self.terrain_mask[:]
        self.plane_mask[:] = (type_mask_plane == -1)
        self.slope_mask[:] = (type_mask_plane == -2)
        # if not self.headless:
        #     print("\n===================================")
        #     print("all name:", self.terrain.terrain_type_name)
        #     print("type:",self.type_idx[self.lookat_id].item())
        #     print("level:", self.level_idx[self.lookat_id].item())
        #     print("terrain:", self.terrain_mask[self.lookat_id].item())
        #     print("plane:", self.plane_mask[self.lookat_id].item())
        #     print("slope:", self.slope_mask[self.lookat_id].item())

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[:, :self.num_actions, 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[:, :self.num_actions, 1]
        if self.arm_dof_enable:
            self.arm_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[:, -self.num_arms:, 0]
            self.arm_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[:, -self.num_arms:, 1]
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)
        self.rigid_body_pos = self.rigid_state[..., 0:3]
        self.rigid_body_quat = self.rigid_state[..., 3:7]
        self.rigid_body_vel = self.rigid_state[..., 7:10]
        self.rigid_body_ang = self.rigid_state[..., 10:13]
        self.left_feet_quat = self.rigid_state[:, self.feet_indices[0], 3:7]
        self.right_feet_quat = self.rigid_state[:, self.feet_indices[1], 3:7]


        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        if self.arm_dof_enable:
            self.arm_torques = torch.zeros(self.num_envs, self.num_arms, dtype=torch.float, device=self.device, requires_grad=False)
            self.arm_actions = torch.zeros(self.num_envs, self.num_arms, dtype=torch.float, device=self.device, requires_grad=False)
            self.arm_actions_fil = torch.zeros(self.num_envs, self.num_arms, dtype=torch.float, device=self.device, requires_grad=False)
            self.cfg.domain_rand.arm_pos_interval = np.ceil(self.cfg.domain_rand.arm_pos_interval_s / self.dt)

        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_rigid_body_vel = torch.zeros_like(self.rigid_body_vel)
        self.stand_flg = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.forward_motion_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.current_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.current_contact_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_both_contact_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.first_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_forces = self.contact_forces[:, self.feet_indices, :]
        history_len = 3
        self.feet_forces_history = torch.zeros(self.num_envs, history_len, len(self.feet_indices), 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_vel_history = torch.zeros(self.num_envs, history_len, len(self.feet_indices), 3, dtype=torch.float, device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.left_feet_euler = get_euler_xyz_tensor(self.left_feet_quat)
        self.right_feet_euler = get_euler_xyz_tensor(self.right_feet_quat)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_height = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        # === height maps ===
        self.num_height_points = 0
        self.num_feet_height_points = 0
        self.num_feet_hold_points = 0
        self.num_base_height_points = 0
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            self.feet_height_points = self._init_feet_height_points()
        self.feet_hold_points = self._init_feet_hold_points()
        self.base_height_points = self._init_base_height_points()
        self.measured_heights = torch.zeros(self.num_envs, self.num_height_points, dtype=torch.float, device=self.device, requires_grad=False)
        self.left_feet_height_maps = torch.zeros(self.num_envs, self.num_feet_height_points, dtype=torch.float, device=self.device, requires_grad=False)
        self.right_feet_height_maps = torch.zeros(self.num_envs, self.num_feet_height_points, dtype=torch.float, device=self.device, requires_grad=False)
        self.left_feet_hold_maps = torch.zeros(self.num_envs, self.num_feet_hold_points, dtype=torch.float, device=self.device, requires_grad=False)
        self.right_feet_hold_maps = torch.zeros(self.num_envs, self.num_feet_hold_points, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_height_maps = torch.zeros(self.num_envs, self.num_base_height_points, dtype=torch.float, device=self.device, requires_grad=False)
        # feet ray caster
        self.feet_front_dist_ray = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_back_dist_ray = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        # === 推力 ===
        self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.rand_small_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.rand_small_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.apply_force = torch.zeros((self.num_envs,self.num_bodies, 3), dtype=torch.float32, device=self.device)
        self.apply_torque = torch.zeros((self.num_envs,self.num_bodies, 3), dtype=torch.float32, device=self.device)

        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(torch.zeros(self.num_envs, self.cfg.env.num_single_privileged_obs, dtype=torch.float, device=self.device))
        # gait paras
        self.gait_time = torch.zeros(self.num_envs, len(self.cfg.commands.gait), dtype=torch.int, device=self.device, requires_grad=False)
        self.stand_command = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.low_speed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.gait_start = torch.randint(0, 2, (self.num_envs,)).to(self.device) * 0.5
        self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.phase_left = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.phase_right = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.cycle_time = torch.full((self.num_envs,), self.cfg.rewards.cycle_time, device=self.device, dtype=torch.float)
        self.add_cycle_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.phase_offset = self.cfg.rewards.phase_offset
        self.stand_radio = torch.full((self.num_envs,), self.cfg.rewards.stand_radio, device=self.device, dtype=torch.float)
        self.curriculum_scale = deepcopy(self.cfg.rewards.curriculum_init)
        if not self.cfg.rewards.penalize_curriculum:
            self.curriculum_scale = 1

        self.cmd_action_latency_buffer = torch.zeros(self.num_envs,self.num_actions,self.cfg.domain_rand.range_cmd_action_latency[1]+1,device=self.device)
        self.cmd_action_latency_simstep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reset_latency_buffer(torch.arange(self.num_envs, device=self.device))

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


    # =============== 机器人参数随机化：质量、惯量、质心、关节阻尼、电机转子惯量、Kp、Kd、扭矩、外部推力、action delay、 sensor niose ===============================
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = self.cfg.domain_rand.num_buckets
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

            self.env_frictions[env_id] = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                num_buckets = self.cfg.domain_rand.num_buckets
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets, 1), device='cpu')
                restitution_coeffs = restitution_buckets[bucket_ids]
                self.restitution_coeffs = restitution_coeffs[:, 0].to(self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_actions, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.q_torque_limits = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(self.num_actions):
                props["effort"][i] = self.cfg.control.dof_torque_max[i]
                props["velocity"][i] = self.cfg.control.dof_vel_max[i]
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()

                self.dof_vel_limits[i] = self.cfg.control.dof_vel_limits[i]
                self.q_torque_limits[i] = self.cfg.control.dof_torque_limits[i]
                # set default joint paras
                props["damping"][i] = self.joint_damping[i]
                props["armature"][i] = self.joint_armature[i]
                props["friction"][i] = self.joint_friction[i]
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2.
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

            if self.arm_dof_enable:
                self.arm_torque_limit = 70
                self.arm_vel_limit = 10
                self.arm_damping = 0.1
                self.arm_armature = 0.01
                self.arm_friction = 0.00
                self.arm_kp = torch.full((self.num_envs, self.num_arms), 100, dtype=torch.float, device=self.device)
                self.arm_kd = torch.full((self.num_envs, self.num_arms), 2, dtype=torch.float, device=self.device)
                arm_kp = [100, 100, 100, 100, 100, 100, 100, 100]
                arm_kd = [  2,   2,   2,   2,   2,   2,   2,   2]
                for i in range(self.num_arms):
                    props["effort"][i + self.num_actions] = self.arm_torque_limit
                    props["velocity"][i + self.num_actions] = self.arm_vel_limit
                    props["damping"][i + self.num_actions] = self.arm_damping
                    props["armature"][i + self.num_actions] = self.arm_armature
                    props["friction"][i + self.num_actions] = self.arm_friction
                    self.arm_kp[:, i] = arm_kp[i]
                    self.arm_kd[:, i] = arm_kd[i]


        # rand joint damping armature friction
        for i in range(self.num_actions):
            if self.cfg.domain_rand.randomize_joint_damping:
                rd_num = np.random.uniform(self.cfg.domain_rand.joint_damping_range[0], self.cfg.domain_rand.joint_damping_range[1])
                if self.cfg.domain_rand.damping_operation == "abs":
                    props["damping"][i] = rd_num
                elif self.cfg.domain_rand.damping_operation == "scale":
                    props["damping"][i] = self.joint_damping[i] * rd_num
                else:
                    raise NotImplementedError(f"Unknown operation: '{self.cfg.domain_rand.damping_operation}' for property randomization. ")

            if self.cfg.domain_rand.randomize_joint_armature:
                rd_num = np.random.uniform(self.cfg.domain_rand.joint_armature_range[0], self.cfg.domain_rand.joint_armature_range[1])
                if self.cfg.domain_rand.armature_operation == "abs":
                    props["armature"][i] = rd_num
                elif self.cfg.domain_rand.armature_operation == "scale":
                    props["armature"][i] = self.joint_armature[i] * rd_num
                else:
                    raise NotImplementedError(f"Unknown operation: '{self.cfg.domain_rand.armature_operation}' for property randomization. ")

            if self.cfg.domain_rand.randomize_joint_friction:
                rd_num = np.random.uniform(self.cfg.domain_rand.joint_friction_range[0], self.cfg.domain_rand.joint_friction_range[1])
                if self.cfg.domain_rand.friction_operation == "abs":
                    props["friction"][i] = rd_num
                elif self.cfg.domain_rand.friction_operation == "scale":
                    props["friction"][i] = self.joint_friction[i] * rd_num
                else:
                    raise NotImplementedError(f"Unknown operation: '{self.cfg.domain_rand.friction_operation}' for property randomization. ")
        # show random para
        if env_id == 0:
            print("\n===================== props random paras =========================")
            print("effort: ",props["effort"])
            print("velocity: ", props["velocity"])
            print("damping: ", props["damping"])
            print("armature: ", props["armature"])
            print("friction: ", props["friction"])
            print("===================== props random paras =========================\n")

        return props

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if env_id == 0:
            self._show_robot_paras(props)
            self.rd_mass_body_idx = self.gym.find_asset_rigid_body_index(self.robot_asset, self.cfg.domain_rand.randomize_mass_body_name)
            self.rd_com_body_idx = self.gym.find_asset_rigid_body_index(self.robot_asset, self.cfg.domain_rand.randomize_com_body_name)
            print("rd_mass_body_idx:",self.rd_mass_body_idx)
            print("rd_com_body_idx:", self.rd_com_body_idx)

        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_base_mass_range
            props[self.rd_mass_body_idx].mass += np.random.uniform(rng[0], rng[1])
        self.body_mass[env_id] = props[self.rd_mass_body_idx].mass

        # randomize base com
        if self.cfg.domain_rand.randomize_base_com:
            rng_comx, rng_comy, rng_comz = self.cfg.domain_rand.added_base_com_range
            rand_com_x = np.random.uniform(rng_comx[0], rng_comx[1])
            rand_com_y = np.random.uniform(rng_comy[0], rng_comy[1])
            rand_com_z = np.random.uniform(rng_comz[0], rng_comz[1])
            props[self.rd_com_body_idx].com.x += 1. * rand_com_x
            props[self.rd_com_body_idx].com.y += 1. * rand_com_y
            props[self.rd_com_body_idx].com.z += 1. * rand_com_z

        # randomize link mass
        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.multiplied_link_mass_range
            for i in range(len(props)):
                if i == self.rd_mass_body_idx:
                    continue
                props[i].mass *= np.random.uniform(rng[0], rng[1])

        # randomize com of all link
        if self.cfg.domain_rand.randomize_link_com:
            for s in range(len(props)):
                if s == self.rd_com_body_idx:
                    continue
                rng_com = self.cfg.domain_rand.added_link_com_range
                rand_com_x = np.random.uniform(rng_com[0], rng_com[1])
                rand_com_y = np.random.uniform(rng_com[0], rng_com[1])
                rand_com_z = np.random.uniform(rng_com[0], rng_com[1])
                props[s].com.x += 1.0 * rand_com_x
                props[s].com.y += 1.0 * rand_com_y
                props[s].com.z += 1.0 * rand_com_z

        # randomize inertia of all body
        if self.cfg.domain_rand.randomize_inertia:
            rng = self.cfg.domain_rand.multiplied_inertia_range
            for s in range(len(props)):
                rd_num = np.random.uniform(rng[0], rng[1])
                props[s].inertia.x.x *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                props[s].inertia.x.y *= rd_num
                props[s].inertia.y.x *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                props[s].inertia.x.z *= rd_num
                props[s].inertia.z.x *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                props[s].inertia.y.y *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                props[s].inertia.y.z *= rd_num
                props[s].inertia.z.y *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                props[s].inertia.z.z *= rd_num
        return props

    def _init_random_motor_paras(self):
        # random Kp Kd strength offset of joint
        self.kp_factor = torch.ones((self.num_envs, self.num_actions), device=self.device, requires_grad=False)
        self.kd_factor = torch.ones((self.num_envs, self.num_actions), device=self.device, requires_grad=False)
        self.motor_strength = torch.ones((self.num_envs, self.num_actions), device=self.device, requires_grad=False)
        self.motor_offset = torch.zeros((self.num_envs, self.num_actions), device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_pd_factor:
            self.kp_factor.uniform_(self.cfg.domain_rand.Kp_factor_range[0], self.cfg.domain_rand.Kp_factor_range[1])
            self.kd_factor.uniform_(self.cfg.domain_rand.Kd_factor_range[0], self.cfg.domain_rand.Kd_factor_range[1])

        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength.uniform_(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1])

        if self.cfg.domain_rand.randomize_motor_offset:
            self.motor_offset.uniform_(self.cfg.domain_rand.motor_offset_range[0], self.cfg.domain_rand.motor_offset_range[1])

    def randomize_dof_props(self, env_ids):
        # rand the motor strength:
        if self.cfg.domain_rand.randomize_motor_strength:
            rng = self.cfg.domain_rand.motor_strength_range
            self.motor_strength[env_ids, :] = torch.empty_like(self.motor_strength[env_ids, :]).uniform_(rng[0], rng[1])
        # rand motor position offset
        if self.cfg.domain_rand.randomize_motor_offset:
            rng = self.cfg.domain_rand.motor_offset_range
            self.motor_offset[env_ids, :] = torch.empty_like(self.motor_offset[env_ids, :]).uniform_(rng[0], rng[1])
        # rand kp kd gain
        if self.cfg.domain_rand.randomize_pd_factor:
            rng = self.cfg.domain_rand.Kp_factor_range
            self.kp_factor[env_ids, :] = torch.empty_like(self.kp_factor[env_ids, :]).uniform_(rng[0], rng[1])
            rng = self.cfg.domain_rand.Kd_factor_range
            self.kd_factor[env_ids, :] = torch.empty_like(self.kd_factor[env_ids, :]).uniform_(rng[0], rng[1])

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        self.rand_push_force.uniform_(-self.cfg.domain_rand.max_push_vel_xy, self.cfg.domain_rand.max_push_vel_xy)
        self.rand_push_torque.uniform_(-self.cfg.domain_rand.max_push_ang_vel, self.cfg.domain_rand.max_push_ang_vel)

        self.root_states[:, 7:9] = self.rand_push_force[:, :2]
        self.root_states[:, 10:13] = self.rand_push_torque

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _small_push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        self.rand_small_push_force.uniform_(-self.cfg.domain_rand.max_small_push_vel_xy, self.cfg.domain_rand.max_small_push_vel_xy)
        self.rand_small_push_torque.uniform_(-self.cfg.domain_rand.max_small_push_ang_vel, self.cfg.domain_rand.max_small_push_ang_vel)

        self.root_states[:, 7:9] += self.rand_small_push_force[:, :2]
        self.root_states[:, 10:13] += self.rand_small_push_torque

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _apply_force_torque(self):
        if (self.common_step_counter-1) % self.cfg.domain_rand.apply_interval == 0:
            self.apply_force[:, 0, :2].uniform_(-self.cfg.domain_rand.max_apply_force * self.cfg.control.decimation, self.cfg.domain_rand.max_apply_force * self.cfg.control.decimation)
            self.apply_torque[:, 0, :3].uniform_(-self.cfg.domain_rand.max_apply_torque * self.cfg.control.decimation, self.cfg.domain_rand.max_apply_torque * self.cfg.control.decimation)

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            forceTensor=gymtorch.unwrap_tensor(self.apply_force),
            torqueTensor=gymtorch.unwrap_tensor(self.apply_torque),
            space=gymapi.CoordinateSpace.LOCAL_SPACE
        )

    def update_cmd_action_latency_buffer(self):
        if not self.cfg.domain_rand.add_cmd_action_latency:
            return self.actions

        actions = self.actions.clone()
        self.cmd_action_latency_buffer[:, :, 1:] = self.cmd_action_latency_buffer[:, :,
                                                   :self.cfg.domain_rand.range_cmd_action_latency[1]].clone()
        self.cmd_action_latency_buffer[:, :, 0] = actions.clone()
        action_delayed = self.cmd_action_latency_buffer[torch.arange(self.num_envs), :,
                         self.cmd_action_latency_simstep.long()]

        return action_delayed

    def _reset_latency_buffer(self, env_ids):
        if self.cfg.domain_rand.add_cmd_action_latency:
            self.cmd_action_latency_buffer[env_ids, :, :] = 0.0
            if self.cfg.domain_rand.randomize_cmd_action_latency:
                self.cmd_action_latency_simstep[env_ids] = torch.randint(
                    self.cfg.domain_rand.range_cmd_action_latency[0],
                    self.cfg.domain_rand.range_cmd_action_latency[1] + 1, (len(env_ids),), device=self.device)
            else:
                self.cmd_action_latency_simstep[env_ids] = self.cfg.domain_rand.range_cmd_action_latency[1]

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        n_cmd = self.cfg.env.num_commands
        n_dof = self.cfg.env.num_actions
        noise_vec[0                   : n_cmd              ] = 0.  # commands
        noise_vec[n_cmd               : n_cmd + n_dof      ] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[n_cmd + 1*n_dof     : n_cmd + 2*n_dof    ] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[n_cmd + 2*n_dof     : n_cmd + 3*n_dof    ] = 0.  # previous actions
        noise_vec[n_cmd + 3*n_dof     : n_cmd + 3*n_dof + 3] = noise_scales.ang_vel * self.obs_scales.ang_vel  # ang vel
        noise_vec[n_cmd + 3*n_dof + 3 : n_cmd + 3*n_dof + 5] = noise_scales.quat * self.obs_scales.quat  # euler x,y
        print("noise = :", noise_vec)
        return noise_vec


    #==================================================== 地形高程图 可视化绘图 ==========================================================================
    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # self._draw_base_height_maps(self.lookat_id)
        if not self.cfg.terrain.measure_heights:
            return
        self._draw_height_maps(self.lookat_id)
        self._draw_feet_hold_maps(self.lookat_id, 0)
        self._draw_feet_hold_maps(self.lookat_id, 1)
        # self._draw_feet_height_maps(self.lookat_id, 0)
        # self._draw_feet_height_maps(self.lookat_id, 1)

    def _draw_height_maps(self, env_idx):
        # ===================== measured_heights ==========================
        i = env_idx
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        base_pos = (self.root_states[i, :3]).cpu().numpy()
        heights = self.measured_heights[i].cpu().numpy()
        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _draw_base_height_maps(self, env_idx):
        # ===================== measured_heights ==========================
        i = env_idx
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 0, 1))
        base_pos = (self.root_states[i, :3]).cpu().numpy()
        heights = self.base_height_maps[i].cpu().numpy()
        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.base_height_points[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _draw_feet_height_maps(self, env_idx, feet_idx):
        # ===================== feet_heights ==========================
        i = env_idx
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0))
        if feet_idx == 0:
            base_pos = (self.rigid_body_pos[i, self.feet_indices[0]]).cpu().numpy()
            heights = self.left_feet_height_maps[i].cpu().numpy()
            height_points = quat_apply_yaw(self.left_feet_quat[i].repeat(heights.shape[0]), self.feet_height_points[i]).cpu().numpy()
        else:
            base_pos = (self.rigid_body_pos[i, self.feet_indices[1]]).cpu().numpy()
            heights = self.right_feet_height_maps[i].cpu().numpy()
            height_points = quat_apply_yaw(self.right_feet_quat[i].repeat(heights.shape[0]), self.feet_height_points[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _draw_feet_hold_maps(self, env_idx, feet_idx):
        # ===================== feet_heights ==========================
        i = env_idx
        sphere_geom = gymutil.WireframeSphereGeometry(0.005, 4, 4, None, color=(0, 1, 0))
        if feet_idx == 0:
            base_pos = (self.rigid_body_pos[i, self.feet_indices[0]]).cpu().numpy()
            heights = self.left_feet_hold_maps[i].cpu().numpy()
            height_points = quat_apply_yaw(self.left_feet_quat[i].repeat(heights.shape[0]), self.feet_hold_points[i]).cpu().numpy()
        else:
            base_pos = (self.rigid_body_pos[i, self.feet_indices[1]]).cpu().numpy()
            heights = self.right_feet_hold_maps[i].cpu().numpy()
            height_points = quat_apply_yaw(self.right_feet_quat[i].repeat(heights.shape[0]), self.feet_hold_points[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _init_base_height_points(self):
        y = torch.tensor(self.cfg.terrain.base_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.base_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_base_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _init_feet_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.feet_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.feet_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_feet_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_feet_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _init_feet_hold_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.feet_hold_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.feet_hold_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_feet_hold_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_feet_hold_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_base_heights(self, env_ids=None):
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_base_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_base_height_points), self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_base_height_points), self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_feet_heights(self, feet_idx=0 ,env_ids=None):
        # feet_idx = 0: 左腿
        # feet_idx = 1: 右腿
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_feet_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            feet_quat = self.rigid_body_quat[env_ids, self.feet_indices[feet_idx]]
            points = quat_apply_yaw(feet_quat.repeat(1, self.num_feet_height_points), self.feet_height_points[env_ids]) + (self.rigid_body_pos[env_ids, self.feet_indices[feet_idx]]).unsqueeze(1)
        else:
            feet_quat = self.rigid_body_quat[:, self.feet_indices[feet_idx]]
            points = quat_apply_yaw(feet_quat.repeat(1, self.num_feet_height_points), self.feet_height_points) + (self.rigid_body_pos[:, self.feet_indices[feet_idx]]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_feet_hold(self, feet_idx=0 ,env_ids=None):
        # feet_idx = 0: 左腿
        # feet_idx = 1: 右腿
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_feet_hold_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            feet_quat = self.rigid_body_quat[env_ids, self.feet_indices[feet_idx]]
            points = quat_apply_yaw(feet_quat.repeat(1, self.num_feet_hold_points), self.feet_hold_points[env_ids]) + (self.rigid_body_pos[env_ids, self.feet_indices[feet_idx]]).unsqueeze(1)
        else:
            feet_quat = self.rigid_body_quat[:, self.feet_indices[feet_idx]]
            points = quat_apply_yaw(feet_quat.repeat(1, self.num_feet_hold_points), self.feet_hold_points) + (self.rigid_body_pos[:, self.feet_indices[feet_idx]]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def terrain_heights(self, base_pos):
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(len(base_pos), dtype=torch.float, device=self.device)
        else:
            x = self.terrain.border + base_pos[:, 0].cpu().numpy() / self.terrain.cfg.horizontal_scale
            y = self.terrain.border + base_pos[:, 1].cpu().numpy() / self.terrain.cfg.horizontal_scale
            x1 = np.floor(x).astype(int)
            y1 = np.floor(y).astype(int)
            x1 = np.clip(x1, 0, self.terrain.height_field_raw.shape[0]-2)
            y1 = np.clip(y1, 0, self.terrain.height_field_raw.shape[1]-2)
            x2 = x1 + 1
            y2 = y1 + 1
            return torch.tensor(
                (
                    (x2 - x) * (y2 - y) * self.terrain.height_field_raw[x1, y1]
                    + (x - x1) * (y2 - y) * self.terrain.height_field_raw[x2, y1]
                    + (x2 - x) * (y - y1) * self.terrain.height_field_raw[x1, y2]
                    + (x - x1) * (y - y1) * self.terrain.height_field_raw[x2, y2]
                )
                * self.terrain.cfg.vertical_scale,
                dtype=torch.float,
                device=self.device,
            )

    def _get_ankle_heights(self):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.num_envs,
                len(self.feet_indices),
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = self.rigid_state[:, self.feet_indices, :2] + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        return heights

    # =========================================================== Walk Gait ==========================================================================
    def _get_phase(self):
        return self.phase

    def _phase_step_update(self):
        if self.cfg.commands.sw_switch:
            self.low_speed = torch.norm(self.base_lin_vel[:, :2], dim=1) <= 0.5
            self.stand_command = (torch.norm(self.commands[:, :3], dim=1) < self.cfg.commands.stand_com_threshold) * self.low_speed
            self.phase += self.dt / (self.cycle_time + self.add_cycle_time)
            self.phase[self.stand_command] = 0
        else:
            self.phase += self.dt / (self.cycle_time + self.add_cycle_time)
        self.phase_left = self.phase % 1
        self.phase_right = (self.phase + self.phase_offset) % 1

    def _gait_style_update(self):
        vel = torch.abs(self.commands[:, 0])
        slow = vel < 0.2

        self.cycle_time[slow] = 1.1
        self.cycle_time[~slow] = self.cfg.rewards.cycle_time

        self.stand_radio[slow] = 0.65
        self.stand_radio[~slow] = self.cfg.rewards.stand_radio

    def _get_gait_phase(self):
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        # left foot stance
        stance_mask[:, 0] = self.phase_left <= self.stand_radio
        # right foot stance
        stance_mask[:, 1] = self.phase_right <= self.stand_radio
        return stance_mask


    # ============================================= Event Command  Sensor Noise ======================================================================
    def _resample_commands(self, env_ids):
        """Randomly select commands for some environments.
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids) == 0:
            return
        # --------------------------------------------------
        # Linear velocity commands
        # --------------------------------------------------
        # x velocity
        self.commands[env_ids, 0] = torch.empty_like(self.commands[env_ids, 0]).uniform_(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1])
        self.commands[env_ids, 1] = torch.empty_like(self.commands[env_ids, 1]).uniform_(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1])
        # --------------------------------------------------
        # Heading / yaw command
        # --------------------------------------------------
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.empty_like(self.commands[env_ids, 3]).uniform_(self.command_ranges["heading"][0] ,self.command_ranges["heading"][1])
        else:
            self.commands[env_ids, 2] = torch.empty_like(self.commands[env_ids, 2]).uniform_(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1])
        # --------------------------------------------------
        # Sample some envs as sagittal walking (no lateral / yaw)
        # --------------------------------------------------
        walking_mask = torch.rand(len(env_ids), device=self.device) < (2.0 / 10.0)
        walking_env_ids = env_ids[walking_mask]
        self.commands[walking_env_ids, 1] = 0.0
        self.commands[walking_env_ids, 2] = 0.0
        # --------------------------------------------------
        # Sample some envs as standing
        # --------------------------------------------------
        standing_mask = torch.rand(len(env_ids), device=self.device) < (2.0 / 10.0)
        standing_env_ids = env_ids[standing_mask]
        self.commands[standing_env_ids, :3] = 0.0

    def _resample_gait_commands(self):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        for i in range(len(self.cfg.commands.gait)):
            # if env finish current gait type, resample command for next gait
            env_ids = (self.episode_length_buf == self.gait_time[:, i]).nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                # according to gait type create a name
                name = '_resample_' + self.cfg.commands.gait[i] + '_command'
                # get function from self based on name
                resample_command = getattr(self, name)
                # resample_command stands for _resample_stand_command/_resample_walk_sagittal_command/...
                resample_command(env_ids)

    def generate_gait_time(self, envs):
        if len(envs) == 0:
            return

        # rand sample
        random_tensor_list = []
        for i in range(len(self.cfg.commands.gait)):
            name = self.cfg.commands.gait[i]
            gait_time_range = self.cfg.commands.gait_time_range[name]
            random_tensor_single = torch_rand_float(gait_time_range[0],
                                                    gait_time_range[1],
                                                    (len(envs), 1), device=self.device)
            random_tensor_list.append(random_tensor_single)

        random_tensor = torch.cat([random_tensor_list[i] for i in range(len(self.cfg.commands.gait))], dim=1)
        current_sum = torch.sum(random_tensor, dim=1, keepdim=True)
        # scaled_tensor store proportion for each gait type
        scaled_tensor = random_tensor * (self.max_episode_length / current_sum)
        scaled_tensor[:, 1:] = scaled_tensor[:, :-1].clone()
        scaled_tensor[:, 0] *= 0.0
        # self.gait_time accumulate gait_duration_tick
        # self.gait_time = |__gait1__|__gait2__|__gait3__|
        # self.gait_time triger resample gait command
        self.gait_time[envs] = torch.cumsum(scaled_tensor, dim=1).int()

    def _resample_stand_command(self, env_ids):
        self.stand_flg[env_ids] = True
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)

    def _resample_walk_sagittal_command(self, env_ids):
        self.stand_flg[env_ids] = False
        self.commands[env_ids, 0] = torch.empty_like(self.commands[env_ids, 0]).uniform_(self.command_ranges["lin_vel_x"][0],
                                                                                         self.command_ranges["lin_vel_x"][1])
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)

    def _resample_walk_lateral_command(self, env_ids):
        self.stand_flg[env_ids] = False
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch.empty_like(self.commands[env_ids, 1]).uniform_(self.command_ranges["lin_vel_y"][0],
                                                                                         self.command_ranges["lin_vel_y"][1])
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)

    def _resample_rotate_command(self, env_ids):
        self.stand_flg[env_ids] = False
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.empty_like(self.commands[env_ids, 3]).uniform_(self.command_ranges["heading"][0],
                                                                                             self.command_ranges["heading"][1])
        else:
            self.commands[env_ids, 2] = torch.empty_like(self.commands[env_ids, 2]).uniform_(self.command_ranges["ang_vel_yaw"][0],
                                                                                             self.command_ranges["ang_vel_yaw"][1])

    def _resample_walk_omnidirectional_command(self, env_ids):
        self.stand_flg[env_ids] = False
        self.commands[env_ids, 0] = torch.empty_like(self.commands[env_ids, 0]).uniform_(self.command_ranges["lin_vel_x"][0],
                                                                                         self.command_ranges["lin_vel_x"][1])
        self.commands[env_ids, 1] = torch.empty_like(self.commands[env_ids, 1]).uniform_(self.command_ranges["lin_vel_y"][0],
                                                                                         self.command_ranges["lin_vel_y"][1])
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.empty_like(self.commands[env_ids, 3]).uniform_(self.command_ranges["heading"][0],
                                                                                             self.command_ranges["heading"][1])
        else:
            self.commands[env_ids, 2] = torch.empty_like(self.commands[env_ids, 2]).uniform_(self.command_ranges["ang_vel_yaw"][0],
                                                                                             self.command_ranges["ang_vel_yaw"][1])

    def _zero_small_commands(self):
        # set small commands to zero
        self.commands[:, :] *= (torch.abs(self.commands[:, :]) >= self.cfg.commands.min_vel)

    # =========================================================== 课程学习 =============================================================================
    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2],
                                           dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids],
                                                                      self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids],
                                                              0))  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _update_terrain_curriculum_vel(self, env_ids):
        """Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1
        )
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        if "tracking_lin_vel" in self.episode_sums.keys():
            move_down = (
                self.episode_sums["tracking_lin_vel"][env_ids] / self.max_episode_length_s
                < (self.reward_scales["tracking_lin_vel"] / self.dt) * 0.5
            ) * ~move_up
            self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        elif "tracking_lin_vel_x" in self.episode_sums.keys():
            move_down = (
                self.episode_sums["tracking_lin_vel_x"][env_ids] / self.max_episode_length_s
                < (self.reward_scales["tracking_lin_vel_x"] / self.dt) * 0.5
            ) * ~move_up
            self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        else:
            print("no tracking_lin_vel in reward for terrain_curriculum")
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * \
                self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5,
                                                          -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0.,
                                                          self.cfg.commands.max_curriculum)

    def training_curriculum(self):
        super().training_curriculum()
        if self.cfg.rewards.penalize_curriculum and (self.learning_iter % 100 == 0):
            self.curriculum_scale = pow(self.curriculum_scale, self.cfg.rewards.penalize_curriculum_sigma)
            if self.curriculum_scale > 0.98:
                self.curriculum_scale = 1.

    # ============================================================ AMP ===============================================================================
    def _init_amp_motion(self):
        self.amp_loader = AMPLoader(
            motion_files=self.cfg.env.amp_motion_files_display, device=self.device, time_between_frames=self.dt
        )
        self.motion_len = self.amp_loader.trajectory_num_frames[0]
        self.trajectory_frame_durations = self.amp_loader.trajectory_frame_durations[0]

    def visualize_amp_motion(self, _time):
        time.sleep(self.trajectory_frame_durations)
        visual_motion_frame = self.amp_loader.get_full_frame_at_time(0, _time)
        device = self.device
        env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=device)
        # 0       3          7         21          27            30            33        47
        # root_pos, root_quat, joint_pos,  foot_pos, root_lin_vel, root_ang_vel, joint_vel

        root_pos = visual_motion_frame[0:3].clone()
        quat_xyzw = visual_motion_frame[3:7].clone()
        quat_wxyz = torch.tensor([quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]], dtype=torch.float32, device=device)
        lin_vel = visual_motion_frame[27:30].clone()
        ang_vel = visual_motion_frame[30:33].clone()
        self.dof_pos[:, 0:self.num_actions] = visual_motion_frame[7  : 21]
        self.dof_vel[:, 0:self.num_actions] = visual_motion_frame[33 : 47]


        # 0       3          7         21          24            27          41
        # root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel
        # root_pos = visual_motion_frame[0:3].clone()
        # quat_xyzw = visual_motion_frame[3:7].clone()
        # quat_wxyz = torch.tensor([quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]], dtype=torch.float32, device=device)
        # lin_vel = visual_motion_frame[21:24].clone() * 0
        # ang_vel = visual_motion_frame[24:27].clone() * 0
        # self.dof_pos[:, 0:self.num_actions] = visual_motion_frame[7  : 21]
        # self.dof_vel[:, 0:self.num_actions] = visual_motion_frame[27 : 41]

        # root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        self.root_states[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
        self.root_states[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (self.num_envs, 1))
        self.root_states[:, 7:10] = torch.tile(lin_vel.unsqueeze(0), (self.num_envs, 1))
        self.root_states[:, 10:13] = torch.tile(ang_vel.unsqueeze(0), (self.num_envs, 1))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids), len(env_ids))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids), len(env_ids))
        self.render()
        self.gym.simulate(self.sim)

    def get_amp_obs_for_expert_trans(self):
        """Gets amp obs from policy"""
        joint_pos = self.dof_pos.clone()

        left_foot_w = self.rigid_state[:, self.feet_indices[0], 0:3] - self.root_states[:, 0:3]
        right_foot_w = self.rigid_state[:, self.feet_indices[1], 0:3] - self.root_states[:, 0:3]
        left_foot_b = quat_rotate_inverse(self.base_quat, left_foot_w)
        right_foot_b = quat_rotate_inverse(self.base_quat, right_foot_w)

        base_lin_vel = self.base_lin_vel.clone()
        base_ang_vel = self.base_ang_vel.clone()
        joint_vel = self.dof_vel.clone()

        # return torch.cat((joint_pos, left_foot_b, right_foot_b, base_lin_vel, base_ang_vel, joint_vel), dim=-1)
        return torch.cat((joint_pos, base_lin_vel, base_ang_vel, joint_vel), dim=-1)

    # ============================================================ Rewards reference motion tracking==================================================
    def _reward_survival(self):
        # Reward survival
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_feet_swing_under_target(self):
        feet_height = self.rigid_state[:, self.feet_indices, 2] - self.cfg.rewards.feet_height - self._get_ankle_heights()
        height_err = feet_height - self.cfg.rewards.target_feet_height
        mask = height_err <= 0  # 只保留低于目标高度的脚
        reward = torch.sum(torch.square(height_err) * ~self.contact * mask, dim=1)
        reward[self.stand_command] = 0.
        return reward

    def _reward_feet_gait_contact(self):
        stance_mask = self._get_gait_phase().bool()
        contact = self.contact_forces[:, self.feet_indices, 2] > 20.0
        # reward: contact == stance
        reward = (contact == stance_mask).sum(dim=1).to(dtype=torch.float)
        # stand command override
        reward[self.stand_command] = 2.0
        return reward

    def _reward_feet_single_contact_time(self):
        """足部接触奖励"""
        rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # 单足接触
        single_feet_contact = torch.logical_xor(self.contact_filt[:, 0], self.contact_filt[:, 1])
        # 双足接触
        both_feet_contact = torch.logical_and(self.contact_filt[:, 0], self.contact_filt[:, 1])

        # 更新双足接触时间
        self.feet_both_contact_time[both_feet_contact] += self.dt
        self.feet_both_contact_time *= both_feet_contact

        # 奖励条件：单足接触或双足接触时间小于0.2秒，或站立命令
        rew_filter = torch.logical_or(single_feet_contact, self.feet_both_contact_time < 0.2)
        rew_filter = torch.logical_or(rew_filter, self.stand_command)
        rew[rew_filter] = 1.0
        return rew

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - self.cfg.rewards.feet_air_time) * first_contact, dim=1) # reward only on first contact with the ground
        self.feet_air_time *= ~self.contact_filt
        rew_airTime[self.stand_command] = 0.
        return rew_airTime

    def _reward_feet_air_time_positive(self):
        air_time = self.current_air_time
        contact_time = self.current_contact_time
        in_contact = self.contact_filt
        in_mode_time = torch.where(in_contact, contact_time, air_time)
        single_stance = torch.sum(in_contact.int(), dim=1) == 1
        reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
        reward = torch.clamp(reward, max=self.cfg.rewards.feet_air_time)
        # no reward for zero command
        reward[self.stand_command] = 0.
        return reward

    def _reward_feet_gait_stance(self):
        """
        Calculates reward for each foot during the stance phase of the gait.
        Fully vectorized for speed.
        """
        # 支撑相掩码 (num_envs, 2)
        stance_mask = self._get_gait_phase()
        # 获取脚在 Z 方向的速度分量 (num_envs, 2, 2)
        # 取 feet_indices 对应的脚, 第7:9维作为 velocity_x, velocity_y
        foot_velocities = self.rigid_state[:, self.feet_indices, 7:9]
        # 计算速度大小并归一化
        foot_vel_mag = torch.norm(foot_velocities, dim=-1) / 5  # [num_envs, 2]
        # 奖励计算：1 - exp(-|v|)
        reward_vel = (1 - torch.exp(-foot_vel_mag)) * stance_mask  # [num_envs, 2]
        # 总奖励
        total_reward = reward_vel.sum(dim=1)
        total_reward[self.stand_command] = 0.0
        return total_reward

    def _reward_feet_gait_swing(self):
        """
        Calculates reward for the swing phase of the gait.
        Fully vectorized for speed.
        """
        # 获取摆动相掩码 (num_envs, 2)
        swing_mask = 1 - self._get_gait_phase()
        # 获取左右脚接触力 (num_envs, 2, 3)
        foot_contact_forces = self.contact_forces[:, self.feet_indices, 0:3]
        # 计算每只脚接触力的范数并归一化
        contact_force_mag = torch.norm(foot_contact_forces, dim=-1) / 50  # [num_envs, 2]
        # 奖励计算：1 - exp(-|force|) 并乘以摆动相掩码
        reward_force = (1 - torch.exp(-contact_force_mag)) * swing_mask  # [num_envs, 2]
        # 总奖励：左右脚相加
        total_reward = reward_force.sum(dim=1)
        total_reward[self.stand_command] = 0.0
        return total_reward

    def _reward_feet_slide(self):
        # scale: -0.25
        contact = self.feet_forces_history.norm(dim=-1).max(dim=1)[0] > 1.
        feet_vel = self.rigid_body_vel[:, self.feet_indices, :2]
        reward = torch.sum(feet_vel.norm(dim=-1) * contact, dim=1)
        return reward

    def _reward_feet_y_distance(self):
        """Penalize foot y-distance when the commanded y-velocity is low, to maintain a reasonable spacing."""
        leftfoot = self.rigid_state[:, self.feet_indices[0], 0:3] - self.root_states[:, 0:3]
        rightfoot = self.rigid_state[:, self.feet_indices[1], 0:3] - self.root_states[:, 0:3]
        leftfoot_b = quat_rotate_inverse(self.base_quat, leftfoot)
        rightfoot_b = quat_rotate_inverse(self.base_quat, rightfoot)
        # x_vel_flag = (torch.abs(self.commands[:, 0]) > 0.2)
        y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1])
        rew = torch.clip(self.cfg.rewards.close_feet_threshold - y_distance_b, min=0, max=1)
        return rew #* x_vel_flag

    def _reward_feet_x_distance(self):
        """
        Penalize foot x-distance when commanded x-velocity is low to maintain reasonable spacing.
        Quat rotation is done separately for left and right foot.
        """
        # 左脚相对根部位置
        leftfoot_rel = self.rigid_body_pos[:, self.feet_indices[0]] - self.base_pos
        leftfoot_b = quat_rotate_inverse(self.base_quat, leftfoot_rel)
        # # 右脚相对根部位置
        rightfoot_rel = self.rigid_body_pos[:, self.feet_indices[1]] - self.base_pos
        rightfoot_b = quat_rotate_inverse(self.base_quat, rightfoot_rel)
        # # 计算 x 方向的绝对距离
        x_distance_b = torch.abs(leftfoot_b[:, 0]) + torch.abs(rightfoot_b[:, 0])
        # # 速度 flag
        vel_flag = (torch.abs(self.commands[:, 0]) < 0.2).float()
        # # 奖励裁剪
        rew = torch.clip(x_distance_b, 0, 1)
        return rew  * vel_flag

    def _reward_feet_too_near(self):
        # 惩罚小于close_feet_threshold
        feet_pos = self.rigid_body_pos[:, self.feet_indices, :]
        distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
        return (self.cfg.rewards.close_feet_threshold - distance).clamp(min=0)

    # ==================================== vel tracking  =========================================== #
    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the x and y axes
        with separate sigmas.
        """
        err_x = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        err_y = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])

        return torch.exp(
            - err_x * self.cfg.rewards.tracking_sigma_x
            - err_y * self.cfg.rewards.tracking_sigma_y
        )

    def _reward_tracking_lin_vel_l2(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:,0:2] - self.base_lin_vel[:, 0:2]), dim=1)
        return lin_vel_error

    def _reward_tracking_stuck(self):
        reward = torch.abs(self.base_lin_vel[:, 0] < 0.3) * torch.abs(self.commands[:, 0] > 0.3)
        return reward

    def _reward_tracking_ang_vel_l2(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return ang_vel_error

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """
        # print(self.commands[self.lookat_id, 2], self.base_ang_vel[self.lookat_id, 2])
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma_z)

    # ==================================== base pos  =========================================== #
    def _reward_torso_gravity(self):
        torso_quat = self.rigid_body_quat[:, self.torso_index]
        projected_gravity = quat_rotate_inverse(torso_quat, self.gravity_vec)
        rew = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
        return rew

    def _reward_torso_ang_vel_xy(self):
        torso_quat = self.rigid_body_quat[:, self.torso_index]
        torso_ang_vel = quat_rotate_inverse(
            torso_quat,
            self.rigid_body_ang[:, self.torso_index],
        )
        rew = torch.sum(torch.square(torso_ang_vel[:, :2]), dim=1)
        return rew

    def _reward_torso_lin_vel_z(self):
        # Penalize z axis base linear velocity
        torso_lin_vel = self.rigid_body_vel[:, self.torso_index]
        return torch.square(torso_lin_vel[:, 2])

    def _reward_base_height(self):
        # Penalize base height away from target
        terrain_height = torch.mean(self.base_height_maps, dim=-1)
        base_height = self.root_states[:, 2] - terrain_height
        reward = torch.square(base_height - self.cfg.rewards.base_height_target)
        return reward

    def _reward_base_feet_height(self):
        feet_height = torch.minimum(
            self.rigid_state[:, self.feet_indices[0], 2],  # left_height
            self.rigid_state[:, self.feet_indices[1], 2]  # right_height
        )
        base_height = self.root_states[:, 2] - feet_height + self.cfg.rewards.feet_height
        reward = torch.square(base_height - self.cfg.rewards.base_height_target)
        reward[self.stand_command] *= 2
        return reward

    def _reward_base_gravity(self):
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return rew

    def _reward_base_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_base_ang_vel_xy(self):
        rew = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        return rew

    def _reward_hip_pos_l1(self):
        joint_diff = self.dof_pos - self.default_dof_pos
        return torch.sum(torch.abs(joint_diff[:, [1, 2, 7, 8]]), dim=1)

    def _reward_hip_pos_l2(self):
        joint_diff = self.dof_pos - self.default_dof_pos
        return torch.sum(torch.square(joint_diff[:, [1, 2, 7, 8]]), dim=1)

    def _reward_hip_roll_pos_mask(self):
        vel_mask = (torch.abs(self.commands[:, 1]) < 0.3)
        joint_diff = self.dof_pos - self.default_dof_pos
        return torch.norm(joint_diff[:, [1, 7]], dim=1) * vel_mask

    def _reward_hip_yaw_pos_mask(self):
        vel_mask = (torch.abs(self.commands[:, 2]) < 0.3)
        return torch.norm(self.dof_pos[:, [2, 8]], dim=1) * vel_mask

    def _reward_waist_pos(self):
        return torch.norm(self.dof_pos[:, [12, 13]], dim=1)

    def _reward_hip_roll_exp(self):
        move_y = (self.commands[:, 1].abs() > 0.2)
        joint_diff = self.dof_pos - self.default_dof_pos
        hip_roll_err = joint_diff[:, [1, 7]].norm(dim=1)
        hip_roll_err *= torch.where(move_y, 0.01, 1.0)
        # 合并并生成奖励
        hip_roll_err.clamp_(max=10)
        return torch.exp(-5* hip_roll_err)

    def _reward_hip_yaw_exp(self):
        move_yaw = (self.commands[:, 2].abs() > 0.2)
        joint_diff = self.dof_pos - self.default_dof_pos
        hip_yaw_err = joint_diff[:, [2, 8]].norm(dim=1)
        hip_yaw_err *= torch.where(move_yaw, 0.01, 1.0)
        hip_yaw_err.clamp_(max=10.0)
        return torch.exp(-5 * hip_yaw_err)

    def _reward_knee_pos_swing(self):
        # 摆动腿，前半周期，惩罚过小的膝关节弯曲，确保抬腿，拟人行走
        swing_mask = torch.zeros((self.num_envs, 2), device=self.device)
        swing_mask[:, 0] = (self.stand_radio <= self.phase_left) & (self.phase_left <= 0.5 + 0.5 * self.stand_radio)
        swing_mask[:, 1] = (self.stand_radio <= self.phase_right) & (self.phase_right <= 0.5 + 0.5 * self.stand_radio)
        target_pos = self.cfg.rewards.target_knee_swing_pos
        pos_error = (target_pos - self.dof_pos[:, [3, 9]])
        pos_error = torch.clamp(pos_error, min= -0.2)  # 将误差限制在[-0.2, 0.8]
        reward = pos_error * ~self.contact * swing_mask
        reward[self.stand_command] = 0.
        return torch.sum(reward, dim=1)

    def _reward_knee_pos_swing_v1(self):
        vel_mask = self.commands[:, 0] > 0.1
        # 摆动腿，前半周期，惩罚过小的膝关节弯曲，确保抬腿，拟人行走
        swing_mask = torch.zeros((self.num_envs, 2), device=self.device)
        swing_mask[:, 0] = (0.1+0.9*self.stand_radio <= self.phase_left) & (self.phase_left <= 0.5 + 0.5 * self.stand_radio)
        swing_mask[:, 1] = (0.1+0.9*self.stand_radio <= self.phase_right) & (self.phase_right <= 0.5 + 0.5 * self.stand_radio)
        target_pos = self.cfg.rewards.target_knee_swing_pos
        pos_error = (target_pos - self.dof_pos[:, [3, 9]])
        pos_error = torch.where(torch.abs(self.commands[:,0:1]) < 0.0, (0.4 - self.dof_pos[:, [3, 9]]), pos_error)
        pos_error = torch.clamp(pos_error, min= -0.2)  # 将误差限制在[-0.2, 0.8]
        reward = pos_error * ~self.contact * swing_mask
        reward[self.stand_command] = 0.
        return torch.sum(reward, dim=1) * vel_mask

    def _reward_ankle_pitch_pos(self):
        # 踝关节保持默认位置，脚尖和脚后跟触地，拟人行走
        vel_mask = torch.abs(self.commands[:,0]) > 0.3
        joint_diff = self.dof_pos - self.default_dof_pos
        return torch.norm(joint_diff[:, [4, 10]], dim=1) * vel_mask

    def _reward_ankle_roll_pos(self):
        joint_diff = self.dof_pos - self.default_dof_pos
        return torch.norm(joint_diff[:, [5, 11]], dim=1)

    def _reward_joint_pos_l2(self):
        joint_diff = self.dof_pos
        reward = torch.sum(torch.square(joint_diff), dim=1)
        reward[~self.stand_command] = 0.
        return reward

    def _reward_feet_ori(self):
        terrain_mask = self.plane_mask | self.step_mask
        left_gravity = quat_rotate_inverse(self.left_feet_quat, self.gravity_vec)
        right_gravity = quat_rotate_inverse(self.right_feet_quat, self.gravity_vec)
        reward = torch.norm(left_gravity[:, 0:2], dim=1)
        reward += torch.norm(right_gravity[:, 0:2], dim=1)
        return reward * terrain_mask

    def _reward_feet_ori_mask(self):
        # 低速时，脚面水平
        vel_mask = torch.abs(self.commands[:,0]) < 0.3
        left_quat = self.rigid_state[:, self.feet_indices[0], 3:7]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.rigid_state[:, self.feet_indices[1], 3:7]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        reward = torch.sum(torch.square(left_gravity[:, 0:2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, 0:2]), dim=1)**0.5
        return reward * vel_mask

    # ==================================== energy  =========================================== #
    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :],
                                     dim=-1) - self.cfg.rewards.max_contact_force).clip(0, ), dim=1)

    def _reward_feet_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.rigid_state[:, self.feet_indices, 7:10] * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))

    def _reward_feet_acc(self):
        feet_acc = (self.last_rigid_body_vel[:, self.feet_indices] - self.rigid_body_vel[:, self.feet_indices]) / self.dt
        reward = torch.sum(torch.norm(feet_acc, dim=2), dim=1)
        return reward

    def _reward_action_rate(self):
        return  torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        return torch.sum(torch.square(self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)

    def _reward_dof_torque(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_torque_ankle(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques[:,[4, 5, 10, 11]]), dim=1)

    def _reward_dof_torque_knee(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques[:,[3, 9]]), dim=1)

    def _reward_dof_torque_rate(self):
        # -1e-7
        return  torch.sum(torch.square(self.last_torques - self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_vel_knee(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.dof_vel[:,[3, 9]]), dim=1)

    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_dof_energy(self):
        # scale -1e-3
        return torch.norm((torch.abs(self.dof_vel) * torch.abs(self.torques)), dim=-1)

    # ==================================== safety  =========================================== #
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_pos_knee_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos[:,[3, 9]] - 0.0).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos[:,[3, 9]] - 2.3).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.q_torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0.),dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)