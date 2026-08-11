import os
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from humanoid.utils.helpers import create_point_list

E1_NUM_ACTIONS = 13
CLOCK_INPUT = 2              # sin(phase), cos(phase)
CMD_DIM = 3                  # vx, vy, yaw_rate
PROPRIOCEPTION_DIM = E1_NUM_ACTIONS * 3 + 5  # 13 q + 13 dq + 13 action + 3 gyro + 2 base roll/pitch
Feet_height_obs_dim = 42*0
Height_obs_dim = 121*0
PRIVILEGED_DIM = 3 + 3 + 3

Single_obs_dim = CLOCK_INPUT + CMD_DIM + PROPRIOCEPTION_DIM
Single_priv_obs_dim = Single_obs_dim + PRIVILEGED_DIM + Feet_height_obs_dim + Height_obs_dim

E1_MOTION_DIR = os.environ.get(
    "E1_MOTION_DIR",
    os.path.join(LEGGED_GYM_ROOT_DIR, "humanoid", "envs", "datasets", "E1_13dof"),
)

E1_AMP_MOTION_FILES = [
    os.path.join(E1_MOTION_DIR, "e1_left.txt"),
    os.path.join(E1_MOTION_DIR, "e1_right.txt"),
    os.path.join(E1_MOTION_DIR, "e1_stand.txt"),
    os.path.join(E1_MOTION_DIR, "e1_stand_to_walk.txt"),
    os.path.join(E1_MOTION_DIR, "e1_turn_left.txt"),
    os.path.join(E1_MOTION_DIR, "e1_turn_right.txt"),
    os.path.join(E1_MOTION_DIR, "e1_walk.txt"),
    os.path.join(E1_MOTION_DIR, "e1_walk_back.txt"),
]

# 脚部、底座、地形采样点网格
Feet_hold_x, Feet_hold_y = create_point_list(resolution=0.01,
                                             range_x=(-0.09-0.02, 0.18+0.02),
                                             range_y=(-0.04, 0.04))
Base_point_x, Base_point_y = create_point_list(resolution=0.05,
                                               range_x=(-0.4, 0.4),
                                               range_y=(-0.3, 0.3),
                                               debug=True)
Terrain_point_x, Terrain_point_y = create_point_list(resolution=0.1,
                                                     range_x=(-0.5, 0.5),
                                                     range_y=(-0.5, 0.5),
                                                     debug=True)

class E1AMPCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        reference_state_initialization = True
        amp_motion_files_display = E1_AMP_MOTION_FILES

        frame_stack = 15
        c_frame_stack = 10
        num_single_obs = Single_obs_dim
        num_single_privileged_obs = Single_priv_obs_dim
        num_observations = int(frame_stack * num_single_obs)
        num_privileged_obs = int(c_frame_stack * num_single_privileged_obs)
        num_commands = CMD_DIM + CLOCK_INPUT
        num_actions = E1_NUM_ACTIONS
        num_arms = 0
        num_envs = 4096
        episode_length_s = 24

        lin_vel_idx = num_single_obs
        feet_height_idx = lin_vel_idx + PRIVILEGED_DIM
        height_idx = feet_height_idx + Feet_height_obs_dim

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        edge_width_thresh = 0.05
        simplify_grid = True

        measure_heights = False
        measured_points_x = Terrain_point_x
        measured_points_y = Terrain_point_y
        feet_points_x = [-0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20]
        feet_points_y = [-0.04, 0., 0.04]
        feet_hold_x = Feet_hold_x
        feet_hold_y = Feet_hold_y
        base_points_x = Base_point_x
        base_points_y = Base_point_y

        static_friction = 1.0
        dynamic_friction = 0.9
        restitution = 0.0

        step_width = 0.30
        slop_range = [0.0, 0.2]
        step_height = [0.00, 0.05]
        step_height_1 = [0.00, 0.05]
        discrete_obstacles_height = [0.00, 0.02]
        step_stone = [0.00, 0.05]
        wave_amplitude = [0.05, 0.2]
        curriculum = False
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        max_init_terrain_level = 5
        terrain_dict = {"smooth slope": 0.2,
                        "rough slope": 0.2,
                        "stairs up": 0.0,
                        "stairs down": 0.0,
                        "discrete": 0.0,
                        "large stairs up": 0.1,
                        "large stairs down": 0.1,
                        "step stone": 0.0,
                        "plane": 0.4,
                        "wave": 0.0,
                        "boxes": 0.0}
        terrain_proportions = list(terrain_dict.values())
        slope_treshold = 0.6

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 10.
        heading_command = False
        gait_enable = False
        gait = ["walk_sagittal", "stand", "walk_lateral", 'rotate', "walk_omnidirectional"]
        gait_time_range = {"walk_sagittal": [4,6],
                           "walk_lateral": [4,6],
                           "rotate": [4,6],
                           "stand": [4,6],
                           "walk_omnidirectional": [4,6]}
        stand_com_threshold = 0.05
        sw_switch = True
        min_vel = 0.2
        class ranges:
            lin_vel_x   = [-0.6, 1.0]
            lin_vel_y   = [-0.5, 0.5]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.675]
        default_joint_angles = {
           'left_hip_pitch_joint': -0.10,
           'left_hip_roll_joint': 0,
           'left_hip_yaw_joint' : 0.,
           'left_knee_joint' : 0.23,
           'left_ankle_pitch_joint' : -0.13,
           'left_ankle_roll_joint' : 0,
           'right_hip_pitch_joint': -0.10,
           'right_hip_roll_joint': 0,
           'right_hip_yaw_joint' : 0.,
           'right_knee_joint' : 0.23,
           'right_ankle_pitch_joint': -0.13,
           'right_ankle_roll_joint' : 0,
           'waist_yaw_joint': 0,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {'hip_pitch': 200,
                     'hip_yaw':   80,
                     'hip_roll':  200,
                     'knee': 200,
                     'ankle': 80,
                     'waist': 150,
                     }
        damping = {  'hip_pitch': 5,
                     'hip_yaw': 3,
                     'hip_roll': 5,
                     'knee': 5,
                     'ankle': 3,
                     'waist': 4,
                     }
        joint_damping = {
            'hip_yaw': 0.1,
            'hip_roll': 0.1,
            'hip_pitch': 0.1,
            'knee': 0.1,
            'ankle': 0.1,
            'waist': 0.1,
        }
        joint_armature = {
            'hip_yaw': 0.01,
            'hip_roll': 0.01,
            'hip_pitch': 0.01,
            'knee': 0.01,
            'ankle': 0.01,
            'waist': 0.01,
        }
        joint_friction = {
            'hip_yaw': 0.0,
            'hip_roll': 0.0,
            'hip_pitch': 0.0,
            'knee': 0.0,
            'ankle': 0.0,
            'waist': 0.0,
        }

        dof_torque_max = [120., 60., 36., 120., 30., 30.,
                          120., 60., 36., 120., 30., 30.,
                          60.]

        dof_vel_max = [
            12.04, 13.09, 13.61, 12.04, 15.71, 15.71,
            12.04, 13.09, 13.61, 12.04, 15.71, 15.71,
            13.09,
        ]

        dof_torque_limits = [120., 60., 36., 120., 30., 30.,
                             120., 60., 36., 120., 30., 30.,
                             60.]

        dof_vel_limits = [
            12.04, 13.09, 13.61, 12.04, 15.71, 15.71,
            12.04, 13.09, 13.61, 12.04, 15.71, 15.71,
            13.09,
        ]

        action_scale = 0.25
        decimation = 4  # 50hz

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/e1/E1_no_hand.xml'
        name = "e1"
        foot_name = "ankle_roll_link"
        knee_name = "knee_link"
        terminate_after_contacts_on = ['pelvis']
        penalize_contacts_on = ["pelvis", "knee_link"]
        self_collisions = 0
        flip_visual_attachments = False
        replace_cylinder_with_capsule = True
        fix_base_link = False
        collapse_fixed_joints = True

    class domain_rand:
        # 瞬时推力扰动
        push_robots = True
        push_interval_s = 7.
        max_push_vel_xy = 1.5
        max_push_ang_vel = 0.0

        small_push_robots = False
        small_push_interval_s = 5.
        max_small_push_vel_xy = 0.5
        max_small_push_ang_vel = 0.2

        # 持续推力扰动
        apply_force_torque = True
        apply_interval_s = 7.
        max_apply_force = 2.0
        max_apply_torque = 0.0

        randomize_friction = True
        num_buckets = 256
        friction_range = [-0.1, 1.0]
        randomize_restitution = False
        restitution_range = [0., 1.0]
        randomize_base_mass = True
        randomize_mass_body_name = "torso_link"
        added_base_mass_range = [-2., 6.]
        randomize_base_com = True
        randomize_com_body_name = "torso_link"
        added_base_com_range = [[-0.02, 0.02],
                                [-0.02, 0.02],
                                [-0.02, 0.02]]
        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]
        randomize_link_com = True
        added_link_com_range = [-0.01, 0.01]
        randomize_inertia = True
        multiplied_inertia_range = [0.9, 1.1]
        randomize_pd_factor = True
        Kp_factor_range = [0.8, 1.2]
        Kd_factor_range = [0.8, 1.2]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_motor_offset = True
        motor_offset_range = [-0.02, 0.02]
        randomize_joint_damping = True
        damping_operation = "abs"
        joint_damping_range = [0.05, 0.1]
        randomize_joint_friction = False
        friction_operation = "abs"
        joint_friction_range = [0.001, 0.005]
        randomize_joint_armature = True
        armature_operation = "abs"
        joint_armature_range = [0.005, 0.01]
        add_cmd_action_latency = True
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [0, 2]
        add_action_noise = False
        action_noise = 0.001
        reset_base_pose = True
        pose_xy = [-0.75, 0.75]
        pose_yaw = [-3.14*1, 3.14*1]
        lin_vel = [-0.5, 0.5]
        ang_vel = [-0.5, 0.5]
        reset_joint = True
        joint_pos_range = [0.5, 1.5]
        joint_vel_range = [-0, 0]
        randomize_arm_pos = False
        arm_pos_interval_s = 10.
        min_arm_pos = [-1.57,  0.00, -1.57,  0.00,   -1.57, -0.57, -1.57,  0.00]
        max_arm_pos = [ 1.57,  0.57,  1.57,  1.57,    1.57,  0.00,  1.57,  1.57]

    class rewards:
        penalize_curriculum = False
        curriculum_init = 0.3
        penalize_curriculum_sigma = 0.8
        base_height_target = 0.672
        feet_height = 0.025
        target_feet_height = 0.08
        target_knee_swing_pos = 0.8
        target_hip_swing_pos = -0.9

        clock_enable = 1
        cycle_time = 1.2
        stand_radio = 0.65
        gait_radio = True
        feet_air_time = 0.6
        phase_offset = 0.5
        only_positive_rewards = False
        tracking_sigma_x = 4
        tracking_sigma_y = 6
        tracking_sigma_z = 6
        max_contact_force = 400
        max_contact_force_xy = 50
        max_feet_vel_z = 2.5
        soft_dof_pos_limit = 0.9
        soft_torque_limit = 0.9
        soft_dof_vel_limit = 0.9
        close_feet_threshold = 0.20

        class scales:
            feet_swing_under_target = -20
            feet_gait_contact = 0.5
            feet_gait_stance = -1.0
            feet_gait_swing = -2.0
            feet_slide = -0.2
            feet_x_distance = -2.
            tracking_stuck = -0.5
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            torso_gravity = -8.
            torso_ang_vel_xy = -0.5
            base_ang_vel_xy = -0.2
            waist_pos = -0.2
            hip_yaw_pos_mask = -2.0
            hip_roll_pos_mask = -2.0
            knee_pos_swing_v1 = -1.0
            ankle_pitch_pos = -0.2
            ankle_roll_pos = -0.2
            feet_contact_forces = -0.01
            action_rate = -0.01
            action_smoothness = -0.01
            dof_torque = -1e-5
            dof_torque_rate = -1e-5
            dof_torque_knee = -3e-5
            dof_vel = -1e-3
            dof_acc = -5e-7
            dof_pos_knee_limits = -2.
            feet_stumble = -5
            termination = -200

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 0.25
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        height_offset = 0.675
        clip_observations = 18.
        clip_actions = 18.

    class noise:
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.02
            dof_vel = 1.50
            ang_vel = 0.30
            lin_vel = 0.10
            quat = 0.02
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11., 5, 3.]

    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.1
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2

Short_obs = 10 * E1AMPCfg.env.num_single_obs
Long_obs = 10 * E1AMPCfg.env.num_single_obs

class E1AMPCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'AmpOnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        noise_std_type = "scalar"
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

        class policy_cfg:
            num_short_obs = Short_obs
            num_long_obs = Long_obs

            class estimator_vel:
                name = 'estimator_vel'
                input_dim = Long_obs
                hidden_dims = [256, 128, 64]
                output_dim = 3 + 9
                activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        desired_kl = 0.01
        entropy_coef = 0.005
        gamma = 0.99
        lam = 0.95
        max_grad_norm = 1.0
        learning_rate = 0.001
        num_learning_epochs = 5
        num_mini_batches = 4
        schedule = 'adaptive'
        learning_rate_print = False
        amp_enable = True

        class amp_cfg:
            step_dt = 0.02
            amp_motion_files = E1_AMP_MOTION_FILES
            amp_num_preload_transitions = 200000
            amp_replay_buffer_size = 100000
            normalizer = True
            amp_reward_coef = 0.4
            amp_task_reward_lerp = 0.7
            amp_discr_hidden_dims = [512, 256, 128]
            amp_loss_coef = 1.0
            loss_type = "LSGAN"
            eta_wgan = 0.5
            amp_loader = "e1"

        # 左右对称增强：索引映射用于镜像数据增强
        class symmetry_cfg:
            sym_loss = True
            obs_permutation = [
                -0.0001, -1, 2, -3, -4,
                11, -12, -13, 14, 15, -16, 5, -6, -7, 8, 9, -10, -17,
                24, -25, -26, 27, 28, -29, 18, -19, -20, 21, 22, -23, -30,
                37, -38, -39, 40, 41, -42, 31, -32, -33, 34, 35, -36, -43,
                -44, 45, -46, -47, 48
            ]
            act_permutation = [6, -7, -8, 9, 10, -11, 0.0001, -1, -2, 3, 4, -5, -12]
            frame_stack = E1AMPCfg.env.frame_stack
            sym_coef = 1.0

        # 特权状态估计配置
        class priv_est_cfg:
            obs_recon_loss = False
            obs_cur_idx = E1AMPCfg.env.num_single_privileged_obs * (E1AMPCfg.env.c_frame_stack - 1)
            obs_cur_dim = Single_obs_dim

            lin_vel_loss = True
            lin_vel_idx = E1AMPCfg.env.num_single_privileged_obs * (E1AMPCfg.env.c_frame_stack - 1) + E1AMPCfg.env.lin_vel_idx
            lin_vel_dim = 3

            feet_height_loss = False
            feet_height_idx = E1AMPCfg.env.num_single_privileged_obs * (E1AMPCfg.env.c_frame_stack - 1) + E1AMPCfg.env.feet_height_idx
            feet_height_dim = Feet_height_obs_dim

            height_recon_loss = False
            height_idx = E1AMPCfg.env.num_single_privileged_obs * (E1AMPCfg.env.c_frame_stack - 1) + E1AMPCfg.env.height_idx
            height_dim = Height_obs_dim

    class runner:
        policy_class_name = 'ActorCriticEST'
        algorithm_class_name = 'AmpPPO'
        num_steps_per_env = 24
        max_iterations = 20000

        save_interval = 500
        save_envs = "e1"
        experiment_name = 'e1_no_hands'
        run_name = 'AMP_e1_13dof_50hz'
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        use_wandb = True
