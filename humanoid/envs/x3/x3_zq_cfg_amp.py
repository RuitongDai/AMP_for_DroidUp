from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from humanoid.utils.helpers import create_point_list

CLOCK_INPUT = 2              # sin cos
CMD_DIM = 3                  # vx vy v_yaw
PROPRIOCEPTION_DIM = 14*3 + 5  # joint pos vel action gyro euler
Feet_height_obs_dim = 42*0     # left_feet_map + right_feet_map
Height_obs_dim = 121*0         # base_height_map
PRIVILEGED_DIM = 3 + 3 + 3   # vel ...

Single_obs_dim = CLOCK_INPUT + CMD_DIM + PROPRIOCEPTION_DIM
Single_priv_obs_dim = Single_obs_dim + PRIVILEGED_DIM + Feet_height_obs_dim + Height_obs_dim

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
class X3zqAMPCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        reference_state_initialization = True
        # amp_motion_files_display = ["/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3/Walk_Lrw/small_walk_065_model_01_15.txt"
        #                             "/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3/Walk_Guo/walk_yaw_065.txt",
        #                             "/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3/Walk_Guo/walk_y_065.txt"]
        # amp_motion_files_display = ["/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/NormalBoy_000_CatBoy_forward_100.txt",
        #                             "/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/small_walk_065_model_back_100.txt",
        #                             "/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/Take_003_guo_walk_turn_120.txt",
        #                             "/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/Take_004_guo_walk_side_120.txt"]

        amp_motion_files_display = [
            # '/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/walk_fb_12s_100hz.txt',
            '/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/walk_forward_12s_100hz.txt',
            '/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/walk_back_12s_100hz.txt',
            '/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/walk_circle_left_12s_100hz.txt',
            '/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/walk_circle_right_12s_100hz.txt',
            "/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/Take_003_guo_walk_turn_120.txt",
            "/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/Take_004_guo_walk_side_120.txt",
        ]

        # amp_motion_files_display = ["/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/from_e3/walk_s1_1.txt"]

        # change the observation dim
        frame_stack = 15
        c_frame_stack = 10
        num_single_obs = Single_obs_dim # 40
        num_single_privileged_obs = Single_priv_obs_dim
        num_observations = int(frame_stack * num_single_obs) # 600
        num_privileged_obs = int(c_frame_stack * num_single_privileged_obs)  # 390
        num_commands = CMD_DIM + CLOCK_INPUT
        num_actions = 14
        num_arms = 8*0 # for domain rand
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds

        lin_vel_idx = num_single_obs  # 40:40+3
        feet_height_idx = lin_vel_idx + PRIVILEGED_DIM  # 45:45+42
        height_idx = feet_height_idx + Feet_height_obs_dim  # 45+42 : 45+42+121

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'plane'
        mesh_type = 'trimesh'
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25  # [m]
        edge_width_thresh = 0.05
        simplify_grid = True

        measure_heights = False
        measured_points_x = Terrain_point_x
        measured_points_y = Terrain_point_y
        feet_points_x = [-0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20]
        feet_points_y = [-0.04, 0., 0.04]
        # for reward feet hold
        feet_hold_x = Feet_hold_x
        feet_hold_y = Feet_hold_y
        # for base_height in terrain
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
        curriculum = True
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 5  # starting curriculum state
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
        # trimesh only:
        slope_treshold = 0.6 # slopes above this threshold will be corrected to vertical surfaces

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        gait_enable = False
        gait = ["walk_sagittal", "stand", "walk_lateral", 'rotate', "walk_omnidirectional"]  # gait type during training
        # proportion during whole life time
        gait_time_range = {"walk_sagittal": [4,6],
                           "walk_lateral": [4,6],
                           "rotate": [4,6],
                           "stand": [4,6],
                           "walk_omnidirectional": [4,6]}
        stand_com_threshold = 0.05 # if (lin_vel_x, lin_vel_y, ang_vel_yaw).norm < this, robot should stand
        sw_switch = True # use stand_com_threshold or not
        min_vel = 0.2 # set lower vel to zero
        class ranges:
            lin_vel_x   = [-0.6, 1.0]  # min max [m/s]
            lin_vel_y   = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-1.5, 1.5]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.90]  # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_pitch_joint': -0.10,
           'left_hip_roll_joint': 0,
           'left_hip_yaw_joint' : 0. ,
           'left_knee_joint' : 0.20,
           'left_ankle_pitch_joint' : -0.10,
           'left_ankle_roll_joint' : 0,
           'right_hip_pitch_joint': -0.10,
           'right_hip_roll_joint': 0,
           'right_hip_yaw_joint' : 0.,
           'right_knee_joint' : 0.20,
           'right_ankle_pitch_joint': -0.10,
           'right_ankle_roll_joint' : 0,
           'waist_roll_joint': 0,
           'waist_yaw_joint': 0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'hip_pitch': 100,
                     'hip_yaw':   100,
                     'hip_roll':  100,
                     'knee': 150,
                     'ankle': 30,
                     'waist': 200,
                     }  # [N*m/rad]
        damping = {  'hip_pitch': 2,
                     'hip_yaw': 2,
                     'hip_roll': 2,
                     'knee': 4,
                     'ankle': 2,
                     'waist': 5,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # ==============joint damping armature ==============
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
        # ============== motor limit and dof limit ==============
        dof_torque_max = [80.,  80.,  80.,  120., 60.,  15.,
                          80.,  80.,  80.,  120., 60.,  15., 80., 80.]

        dof_vel_max = [10.,  10.,  10.,   10.,  10.,  5.,
                       10.,  10.,  10.,   10.,  10.,  5., 6., 6.]


        dof_torque_limits = [75.,  80.,  80.,  120.,  60.,  15.,
                             75.,  80.,  80.,  120.,  60.,  15., 80., 80.]

        dof_vel_limits = [ 10.,   10.,   10.,  10.,  10.,  5.,
                           10.,   10.,   10.,  10.,  10.,  5., 6., 6.]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        # decimation = 2  # 100hz
        decimation = 4  # 50hz

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/x3_zhi_qu/x3_zq_14dof.urdf'
        name = "x3_zhi_qu"
        foot_name = "ankle_roll"
        knee_name = "knee"
        terminate_after_contacts_on = ['pelvis']
        penalize_contacts_on = ["pelvis", "knee"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = True
        fix_base_link = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">


    class domain_rand:
        # 瞬时扰动
        push_robots = True
        push_interval_s = 7.
        max_push_vel_xy = 1.5
        max_push_ang_vel = 0.0

        small_push_robots = False
        small_push_interval_s = 5.
        max_small_push_vel_xy = 0.5
        max_small_push_ang_vel = 0.2

        # 持续推力
        apply_force_torque = True
        apply_interval_s = 7.
        max_apply_force = 2.0
        max_apply_torque = 0.0

        # 摩擦系数
        randomize_friction = True
        num_buckets = 256
        friction_range = [-0.1, 1.0]
        # 恢复系数
        randomize_restitution = False
        restitution_range = [0., 1.0]
        # base质量
        randomize_base_mass = True
        randomize_mass_body_name = "torso_link" # "pelvis" "torso_link"
        added_base_mass_range = [-2., 6.]
        # base质心
        randomize_base_com = True
        randomize_com_body_name = "torso_link"
        added_base_com_range = [[-0.02, 0.02],
                                [-0.02, 0.02],
                                [-0.02, 0.02]]
        # 其他link质量
        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]
        # 其他link质心
        randomize_link_com = True
        added_link_com_range = [-0.01, 0.01]
        # 惯量
        randomize_inertia = True
        multiplied_inertia_range = [0.9, 1.1]
        # PD刚度阻尼
        randomize_pd_factor = True
        Kp_factor_range = [0.8, 1.2]
        Kd_factor_range = [0.8, 1.2]
        # torque
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        # dof offset
        randomize_motor_offset = True
        motor_offset_range = [-0.02, 0.02]
        # joint 阻尼
        randomize_joint_damping = True
        damping_operation = "abs"        # ["scale", "abs"]
        joint_damping_range = [0.05, 0.1]
        # joint 静摩擦
        randomize_joint_friction = False
        friction_operation = "abs"  # ["scale", "abs"]
        joint_friction_range = [0.001, 0.005]
        # 电枢惯量
        randomize_joint_armature = True
        armature_operation = "abs"  # ["scale", "abs"]
        joint_armature_range = [0.005, 0.01]
        # action延迟
        add_cmd_action_latency = True
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [0, 2] # [0 - 2*5] ms
        # action噪声
        add_action_noise= False
        action_noise = 0.001
        # base state 重置
        reset_base_pose = True
        pose_xy = [-0.75, 0.75]
        pose_yaw =[-3.14*1, 3.14*1]
        lin_vel = [-0.5, 0.5]
        ang_vel = [-0.5, 0.5]
        # joint state 重置
        reset_joint = True
        joint_pos_range = [0.5, 1.5] # scale
        joint_vel_range = [-0, 0] # range rad/s
        # arm noise
        randomize_arm_pos = False
        arm_pos_interval_s = 10.
        min_arm_pos = [-1.57,  0.00, -1.57,  0.00,   -1.57, -0.57, -1.57,  0.00] # rad
        max_arm_pos = [ 1.57,  0.57,  1.57,  1.57,    1.57,  0.00,  1.57,  1.57] # rad

    class rewards:
        penalize_curriculum = False
        curriculum_init = 0.3
        penalize_curriculum_sigma = 0.8
        base_height_target = 0.8 # 0.10 rad
        feet_height = 0.025
        target_feet_height = 0.08    # m
        target_knee_swing_pos = 0.8  # rad
        target_hip_swing_pos = -0.9  # rad

        clock_enable = 1 # 1: 使用[sin、cos]时钟信号,  0: 不使用时钟信号
        cycle_time = 1.2                   # sec
        stand_radio = 0.65
        gait_radio = True
        feet_air_time = 0.6
        phase_offset = 0.5
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False
        # tracking reward = exp(error*sigma)
        tracking_sigma_x = 4
        tracking_sigma_y = 6
        tracking_sigma_z = 6
        max_contact_force = 400  # forces above this value are penalized
        max_contact_force_xy = 50  # forces above this value are penalized
        max_feet_vel_z = 2.5
        soft_dof_pos_limit = 0.9
        soft_torque_limit = 0.9
        soft_dof_vel_limit = 0.9
        close_feet_threshold = 0.20

        class scales:
            # survival = 0.5
            #==================== gait style ========================
            feet_swing_under_target = -20
            feet_gait_contact = 0.5
            feet_gait_stance = -1.0
            feet_gait_swing = -2.0
            feet_slide = -0.2
            # feet_too_near = -2.
            feet_x_distance = -2.
            # ==================== vel tracking =======================
            tracking_stuck = -0.5
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            # ==================== base pos ==========================
            torso_gravity = -8.
            torso_ang_vel_xy = -0.5
            # base_gravity = -2.
            # base_lin_vel_z = -0.5
            base_ang_vel_xy = -0.2

            waist_pos = -0.2
            hip_yaw_pos_mask = -2.0
            hip_roll_pos_mask = -2.0
            knee_pos_swing_v1 = -1.0
            ankle_pitch_pos = -0.2
            ankle_roll_pos = -0.2
            # feet_ori_mask = -0.2
            #======================== energy ============================
            feet_contact_forces = -0.01
            action_rate = -0.01
            action_smoothness = -0.01
            dof_torque = -1e-5
            dof_torque_rate = -1e-5
            dof_torque_knee = -3e-5
            dof_vel = -1e-3
            dof_acc = -5e-7
            #======================= reward for safety ===================
            # dof_pos_limits = -5.
            dof_pos_knee_limits = -2.
            # dof_torque_limits = -0.01
            # dof_vel_limits = -0.5
            feet_stumble = -5 # 15
            termination = -200

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 0.25
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        height_offset = 0.9
        clip_observations = 18.
        clip_actions = 18.

    class noise:
        add_noise = True
        noise_level = 1.0    # scales other values

        class noise_scales:
            dof_pos = 0.02
            dof_vel = 1.50
            ang_vel = 0.30
            lin_vel = 0.10
            quat    = 0.02
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim(LeggedRobotCfg.sim):
        dt = 0.005    # 200 Hz
        substeps = 1  # 2
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # 0.1 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

Short_obs = 10 * X3zqAMPCfg.env.num_single_obs # input actor
Long_obs =  10 * X3zqAMPCfg.env.num_single_obs # input state_encoder

class X3zqAMPCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'AmpOnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        noise_std_type = "scalar"  # 'scalar' or 'log'
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        # for ActorCritic_XXX
        class policy_cfg:
            # 长短历史
            num_short_obs = Short_obs
            num_long_obs = Long_obs

            # -- 显示估计 vel
            class estimator_vel:
                name = 'estimator_vel'
                # obs
                input_dim = Long_obs
                hidden_dims = [256, 128, 64]
                # vel + latent
                output_dim = 3 + 9
                activation = 'elu'


    class algorithm(LeggedRobotCfgPPO.algorithm):
        # -- value function
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        # -- surrogate loss
        desired_kl = 0.01
        entropy_coef = 0.005
        gamma = 0.99
        lam = 0.95
        max_grad_norm = 1.0
        # -- training
        learning_rate = 0.001
        num_learning_epochs =  5
        num_mini_batches = 4  # mini batch size = num_envs * num_steps / num_mini_batches
        schedule = 'adaptive'  # adaptive, fixed
        learning_rate_print = False
        amp_enable = True # 混合精度训练
        # -- Amp cfg
        class amp_cfg:
            step_dt = 0.02
            # amp_motion_files = ["/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3/Walk_Lrw/small_walk_065_model_01_15.txt",
            #                     "/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3/Walk_Guo/walk_yaw_065.txt",
            #                     "/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3/Walk_Guo/walk_y_065.txt"]
            # amp_motion_files = ["/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/NormalBoy_000_CatBoy_forward_100.txt",
            #                     "/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/small_walk_065_model_back_100.txt",
            #                     "/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/Take_003_guo_walk_turn_120.txt",
            #                     "/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/Take_004_guo_walk_side_120.txt"]

            # amp_motion_files = ['/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/walk_fb_12s_100hz.txt',
            #                     "/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/Take_003_guo_walk_turn_120.txt",
            #                     "/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/Take_004_guo_walk_side_120.txt"]

            amp_motion_files = [
                # '/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/walk_fb_12s_100hz.txt',
                '/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/walk_forward_12s_100hz.txt',
                '/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/walk_back_12s_100hz.txt',
                '/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/walk_circle_left_12s_100hz.txt',
                '/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/walk_circle_right_12s_100hz.txt',
                "/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/Take_003_guo_walk_turn_120.txt",
                "/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/Take_004_guo_walk_side_120.txt",
            ]

            # amp_motion_files = ["/home/liangzhiyuan/RL/droidup/x2-gym-terrain/humanoid/envs/datasets/x3_zq/from_e3/walk_s1_1.txt"]

            amp_num_preload_transitions = 200000
            amp_replay_buffer_size = 100000
            # amp parameter
            normalizer = True
            # amp_reward_coef = 5.0 * step_dt
            # amp_task_reward_lerp = 0.2
            amp_reward_coef = 0.4
            amp_task_reward_lerp = 0.7
            amp_discr_hidden_dims = [512, 256, 128]
            amp_loss_coef = 1.0
            loss_type = "LSGAN" # "LSGAN", "WGAN", "BCEWithLogits"
            eta_wgan = 0.5 # 0.1 ~ 0.5 is the proper range of selection

        # -- Symmetry Augmentation
        class symmetry_cfg:
            sym_loss = True
            obs_permutation = [
                -0.0001, -1, 2, -3, -4, # cmd
                 11,-12,-13, 14, 15,-16,   5, -6, -7,  8,  9,-10,  -17, -18,  # q
                 25,-26,-27, 28, 29,-30,  19,-20,-21, 22, 23,-24,  -31, -32,  # dq
                 39,-40,-41, 42, 43,-44,  33,-34,-35, 36, 37,-38,  -45, -46,# actions
                -47, 48,-49,-50, 51   # ang_vel + ang
            ]
            act_permutation = [ 6,-7,-8, 9, 10,-11,  0.0001,-1,-2, 3, 4,-5, -12, -13]
            frame_stack = X3zqAMPCfg.env.frame_stack
            sym_coef = 1.0

        # -- est cfg
        class priv_est_cfg:
            obs_recon_loss = False
            obs_cur_idx = X3zqAMPCfg.env.num_single_privileged_obs * (X3zqAMPCfg.env.c_frame_stack- 1)
            obs_cur_dim = Single_obs_dim

            lin_vel_loss = True
            lin_vel_idx = X3zqAMPCfg.env.num_single_privileged_obs * (X3zqAMPCfg.env.c_frame_stack- 1) + X3zqAMPCfg.env.lin_vel_idx
            lin_vel_dim = 3

            feet_height_loss = False
            feet_height_idx = X3zqAMPCfg.env.num_single_privileged_obs * (X3zqAMPCfg.env.c_frame_stack- 1) + X3zqAMPCfg.env.feet_height_idx
            feet_height_dim = Feet_height_obs_dim

            height_recon_loss = False
            height_idx = X3zqAMPCfg.env.num_single_privileged_obs * (X3zqAMPCfg.env.c_frame_stack- 1) + X3zqAMPCfg.env.height_idx
            height_dim = Height_obs_dim

    class runner:
        policy_class_name = 'ActorCriticEST'
        algorithm_class_name = 'AmpPPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 20000  # number of policy updates

        # logging
        save_interval = 500  # check for potential saves every this many iterations
        save_envs = "x3"
        experiment_name = 'x3_zq'
        run_name = 'AMP_x3_zq_50hz'
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        use_wandb = False