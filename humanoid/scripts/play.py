import os
from typing import Union
from humanoid.envs.x3.x3_zq_cfg_amp import X3zqAMPCfg
from humanoid.envs import *
from humanoid.utils import  get_args, export_policy_as_jit, export_policy_as_onnx, task_registry
from humanoid.utils.helpers import get_load_path
import torch

def play(args):
    env_cfg: Union[X3zqAMPCfg]
    env: X2BaseEnv

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.env.num_envs = 49
    env_cfg.env.episode_length_s = 1500
    env_cfg.commands.gait_enable = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 6000
    env_cfg.terrain.mesh_type = 'trimesh'
    # env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.border_size = 5
    env_cfg.terrain.num_rows = 5 # level
    env_cfg.terrain.num_cols = 5 # type
    env_cfg.terrain.max_init_terrain_level = 5  # starting curriculum state
    # env_cfg.terrain.terrain_dict = {"smooth slope": 0.3,
    #                 "rough slope": 0.3,
    #                 "stairs up": 0.1,
    #                 "stairs down": 0.0,
    #                 "discrete": 0.0,
    #                 "large stairs up": 0.1,
    #                 "large stairs down": 0.0,
    #                 "step stone": 0.0,
    #                 "plane": 0.2, }
    # env_cfg.terrain.terrain_dict = {"smooth slope": 0.0,
    #                                 "rough slope": 0.0,
    #                                 "stairs up": 0.0,
    #                                 "stairs down": 0.0,
    #                                 "discrete": 0.0,
    #                                 "large stairs up": 0.0,
    #                                 "large stairs down": 0.0,
    #                                 "stepping stones": 0.0,
    #                                 "plane": 1.0, }
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = True

    env_cfg.noise.add_noise = False
    env_cfg.noise.noise_level = 0.5
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.small_push_robots = False
    env_cfg.domain_rand.push_interval_s = 7.
    env_cfg.domain_rand.max_push_vel_xy = 1.
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.apply_force_torque = False
    env_cfg.domain_rand.apply_interval_s = 7.
    env_cfg.domain_rand.max_apply_force = 10.
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_link_com = False
    env_cfg.domain_rand.randomize_inertia = False
    env_cfg.domain_rand.randomize_pd_factor = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_motor_offset = False
    env_cfg.domain_rand.randomize_joint_damping = False
    env_cfg.domain_rand.randomize_joint_armature = False
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.add_action_noise = False
    env_cfg.domain_rand.randomize_arm_pos = False
    # env_cfg.domain_rand.arm_pos_interval_s = 10.

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg= task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    # estimator = ppo_runner.get_inference_estimator(device=env.device)
    #获取策略名
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    policy_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
    # 使用 split('/') 将路径按 '/' 分割，然后取倒数第二部分
    folder_name = policy_path.split('/')[-2]
    # 使用 split('_') 将文件名按 '_' 分割，然后取最后一部分并去掉 '.pt'
    model_number = policy_path.split('_')[-1].split('.')[0]
    # 合并结果
    policy_name = folder_name + "_" + model_number

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, folder_name, 'exported')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, policy_name)
        print('Exported policy as jit script to: ', path)

    for i in range(1000*int(env.max_episode_length)):
        if FIX_COMMAND:
            mask = torch.ones(env.num_envs, dtype=torch.bool)
            mask[env.lookat_id] = False
            env.commands[mask, 0] = 1.2
            env.commands[mask, 1] = 0
            env.commands[mask, 2] = 0.0
            env.commands[mask, 3] = 0.
        env.commands[:, 0] = torch.clip(env.commands[:, 0], env_cfg.commands.ranges.lin_vel_x[0], env_cfg.commands.ranges.lin_vel_x[1])
        env.commands[:, 1] = torch.clip(env.commands[:, 1], env_cfg.commands.ranges.lin_vel_y[0], env_cfg.commands.ranges.lin_vel_y[1])
        env.commands[:, 2] = torch.clip(env.commands[:, 2], env_cfg.commands.ranges.ang_vel_yaw[0], env_cfg.commands.ranges.ang_vel_yaw[1])
        env.commands[:, 3] = torch.clip(env.commands[:, 3], env_cfg.commands.ranges.heading[0], env_cfg.commands.ranges.heading[1])

        actions = policy(obs.detach())
        obs, critic_obs, rews, dones, infos, _, _ = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = True
    FIX_COMMAND = True
    args = get_args()
    play(args)
