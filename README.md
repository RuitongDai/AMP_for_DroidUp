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


## Code Structure

1. Every environment hinges on an `env` file (`legged_robot.py`) and a `configuration` file (`legged_robot_config.py`). The latter houses two classes: `LeggedRobotCfg` (encompassing all environmental parameters) and `LeggedRobotCfgPPO` (denoting all training parameters).
2. Both `env` and `config` classes use inheritance.
3. Non-zero reward scales specified in `cfg` contribute a function of the corresponding name to the sum-total reward.
4. Tasks must be registered with `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. Registration may occur within `envs/__init__.py`, or outside of this repository.


## Usage Guide

#### Examples

```bash
# This command initiates the PPO algorithm-based training for the humanoid task.
python scripts/train.py --task=e1_amp  --headless --num_envs 4096
python scripts/play.py --task=e1_amp
python scripts/play_amp_motion.py --task=e1_amp

```

### Train:
```bash
python scripts/train.py --task=x3_zq_amp  --headless --num_envs 4096
```
- To run on CPU add following arguments: --sim_device=cpu, --rl_device=cpu (sim on CPU and rl on GPU is possible).
- To run headless (no rendering) add --headless.
- Important: To improve performance, once the training starts press v to stop the rendering. You can then enable it later to check the progress.
- The trained policy is saved in issacgym_anymal/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt. Where <experiment_name> and <run_name> are defined in the train config.
- The following command line arguments override the values set in the config files:
  - --task TASK: Task name.
  - --resume: Resume training from a checkpoint
  - --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
  - --run_name RUN_NAME: Name of the run.
  - --load_run LOAD_RUN: Name of the run to load when resume=True. If -1: will load the last run.
  - --checkpoint CHECKPOINT: Saved model checkpoint number. If -1: will load the last checkpoint.
  - --num_envs NUM_ENVS: Number of environments to create.
  - --seed SEED: Random seed.
  - --max_iterations MAX_ITERATIONS: Maximum number of training iterations.
### Play a trained policy:
```bash
python scripts/play.py --task=x3_zq_amp
```
- By default, the loaded policy is the last model of the last run of the experiment folder.
- Other runs/model iteration can be selected by setting load_run and checkpoint in the train config.
### Play amp motions:
```bash
python scripts/play_amp_motions.py --task=x3_zq_amp
```

## Troubleshooting

Observe the following cases:

```bash
# error
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
# solution
# set the correct path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# OR
sudo apt install libpython3.8
```


## 扩展
1. 导入资产,准备数据集
2. 编写特定的motionloader
3. amp_ppo.py里导入新的motionloader
4. 编写cfg和env文件
5. init注册任务


## Acknowledgment

The implementation of Humanoid-Gym relies on resources from [legged_gym](https://github.com/leggedrobotics/legged_gym) and [rsl_rl](https://github.com/leggedrobotics/rsl_rl) projects, created by the Robotic Systems Lab. We specifically utilize the `LeggedRobot` implementation from their research to enhance our codebase.