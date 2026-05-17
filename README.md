## Features

### 1. Humanoid Robot Training
This repository offers comprehensive guidance and scripts for the training of humanoid robots. Humanoid-Gym features specialized rewards for humanoid robots, simplifying the difficulty of sim-to-real transfer. In this repository, we use RobotEra's XBot-L as a primary example. It can also be used for other robots with minimal adjustments. Our resources cover setup, configuration, and execution. Our goal is to fully prepare the robot for real-world locomotion by providing in-depth training and optimization.


- **Comprehensive Training Guidelines**: We offer thorough walkthroughs for each stage of the training process.
- **Step-by-Step Configuration Instructions**: Our guidance is clear and succinct, ensuring an efficient setup process.
- **Execution Scripts for Easy Deployment**: Utilize our pre-prepared scripts to streamline the training workflow.

### 2. Sim2Sim Support
We also share our sim2sim pipeline, which allows you to transfer trained policies to highly accurate and carefully designed simulated environments. Once you acquire the robot, you can confidently deploy the RL-trained policies in real-world settings.

Our simulator settings, particularly with Mujoco, are finely tuned to closely mimic real-world scenarios. This careful calibration ensures that the performances in both simulated and real-world environments are closely aligned. This improvement makes our simulations more trustworthy and enhances our confidence in their applicability to real-world scenarios.


### 3. Denoising World Model Learning (Coming Soon!)
Denoising World Model Learning(DWL) presents an advanced sim-to-real framework that integrates state estimation and system identification. This dual-method approach ensures the robot's learning and adaptation are both practical and effective in real-world contexts.

- **Enhanced Sim-to-real Adaptability**: Techniques to optimize the robot's transition from simulated to real environments.
- **Improved State Estimation Capabilities**: Advanced tools for precise and reliable state analysis.

### Dexterous Hand Manipulation (Coming Soon!)

## Installation

1. Generate a new Python virtual environment with Python 3.8 using `conda create -n AMP_for_DroidUp python=3.8`.
2. Install PyTorch:
   - `conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia`
3. Install Isaac Gym:
   - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym.
   - `cd isaacgym/python && pip install -e .`
   - Run an example with `cd examples && python 1080_balls_of_solitude.py`.
   - Consult `isaacgym/docs/index.html` for troubleshooting.
4. Install humanoid-gym:
   - Clone this repository.
   - `cd humanoid-gym && pip install -e .`

## Usage Guide

#### Examples

```bash
# This command initiates the PPO algorithm-based training for the humanoid task.
python scripts/train.py --task=x3_zq_amp  --headless --num_envs 4096
python scripts/play.py --task=x3_zq_amp
python scripts/play_amp_motions.py --task=x3_zq_amp

```
#### 1. Default Tasks


- **humanoid_ppo**
   - Purpose: Baseline, PPO policy, Multi-frame low-level control
   - Observation Space: Variable $(47 \times H)$ dimensions, where $H$ is the number of frames
   - $[O_{t-H} ... O_t]$
   - Privileged Information: $73$ dimensions

- **humanoid_dwl (coming soon)**

#### 2. PPO Policy
- **Training Command**: For training the PPO policy, execute:
  ```
  python humanoid/scripts/train.py --task=humanoid_ppo --load_run log_file_path --name run_name
  ```
- **Running a Trained Policy**: To deploy a trained PPO policy, use:
  ```
  python humanoid/scripts/play.py --task=humanoid_ppo --load_run log_file_path --name run_name
  ```
- By default, the latest model of the last run from the experiment folder is loaded. However, other run iterations/models can be selected by adjusting `load_run` and `checkpoint` in the training config.

#### 3. Parameters
- **CPU and GPU Usage**: To run simulations on the CPU, set both `--sim_device=cpu` and `--rl_device=cpu`. For GPU operations, specify `--sim_device=cuda:{0,1,2...}` and `--rl_device={0,1,2...}` accordingly. Please note that `CUDA_VISIBLE_DEVICES` is not applicable, and it's essential to match the `--sim_device` and `--rl_device` settings.
- **Headless Operation**: Include `--headless` for operations without rendering.
- **Rendering Control**: Press 'v' to toggle rendering during training.
- **Policy Location**: Trained policies are saved in `humanoid/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`.

## Code Structure

1. Every environment hinges on an `env` file (`legged_robot.py`) and a `configuration` file (`legged_robot_config.py`). The latter houses two classes: `LeggedRobotCfg` (encompassing all environmental parameters) and `LeggedRobotCfgPPO` (denoting all training parameters).
2. Both `env` and `config` classes use inheritance.
3. Non-zero reward scales specified in `cfg` contribute a function of the corresponding name to the sum-total reward.
4. Tasks must be registered with `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. Registration may occur within `envs/__init__.py`, or outside of this repository.

## Troubleshooting

Observe the following cases:

```bash
# error
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
# solution
# set the correct path
export LD_LIBRARY_PATH="~/miniconda3/envs/your_env/lib:$LD_LIBRARY_PATH" 
# OR
sudo apt install libpython3.8

```

## Citation

Please cite the following if you use this code or parts of it:
```
@article{gu2024humanoid,
  title={Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer},
  author={Gu, Xinyang and Wang, Yen-Jen and Chen, Jianyu},
  journal={arXiv preprint arXiv:2404.05695},
  year={2024}
}
```

## Acknowledgment

The implementation of Humanoid-Gym relies on resources from [legged_gym](https://github.com/leggedrobotics/legged_gym) and [rsl_rl](https://github.com/leggedrobotics/rsl_rl) projects, created by the Robotic Systems Lab. We specifically utilize the `LeggedRobot` implementation from their research to enhance our codebase.

## Any Questions?

If you have further questions, please feel free to contact [support@robotera.com](mailto:support@robotera.com) or create an issue in this repository.

## wandb
### 挂代理，这里用的是clash for windows
```bash
export HTTP_PROXY=http://127.0.0.1:7899
export HTTPS_PROXY=http://127.0.0.1:7899
```
### wandb 项目名称
```bash
entity="2741355724-droid",
project="x2_terrain",
```