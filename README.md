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
python scripts/train.py --task=x3_zq_amp  --headless --num_envs 4096
python humanoid/scripts/train.py \
  --task=f4_amp \
  --headless \
  --num_envs=4096 \
  --resume \
  --load_run="2026-10-11_16-15-19_AMP_x3_f4_50hz" \
  --checkpoint=20000
python scripts/play.py --task=x3_zq_amp
python scripts/play_amp_motions.py --task=x3_zq_amp

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

### F4 AMP sim2sim (ONNX / MuJoCo)

在项目根目录运行，脚本不依赖 Isaac Gym，也不依赖 `x3_amp_deploy` 的安装路径：

```bash
# 安装部署依赖（训练环境已有 numpy 和 mujoco）
python -m pip install onnxruntime evdev

# 默认使用手柄；未连接或断开时清零命令并每秒重试连接
python humanoid/scripts/sim2sim_f4.py --onnx /path/to/f4_policy.onnx

# 可显式指定手柄设备
python humanoid/scripts/sim2sim_f4.py --onnx /path/to/f4_policy.onnx \
  --gamepad-device /dev/input/eventN

# 无窗口检查，并记录 10 秒仿真数据
python humanoid/scripts/sim2sim_f4.py --onnx /path/to/f4_policy.onnx \
  --fixed-command --headless --no-realtime --duration 10 --cmd 0.4 0 0 --csv /tmp/f4_sim.csv
```

默认手柄控制，沿用参考部署项目的映射：

| 操作 | 功能 |
| --- | --- |
| 左摇杆上下 | 前后速度 |
| 右摇杆左右 | 横移速度 |
| 按住 LT + 左摇杆左右 | 转向速度 |
| LT + Start | 持续步态 |
| LT + Back | 自动站立/行走 |
| X | 重置机器人与观测历史 |
| Y | 开始/停止 CSV 录制，默认保存到 `logs/f4_sim2sim/` |
| RB | 开关右摇杆推力扰动；开启后右摇杆控制推力，横移命令归零 |

自动搜索名称包含 `360` 或 `pad` 的可读取设备；也可以用 `--gamepad-device` 指定。
手柄断开时清零速度命令及扰动力，并持续尝试重连。默认运行到关闭窗口或按 `Ctrl+C`，
也可以通过 `--duration` 设置时间。与参考一致，X/Y 的底层键值分别映射为
`BTN_NORTH` / `BTN_WEST`。

如需键盘控制，添加 `--keyboard`（先点击 MuJoCo 窗口）：`W/S` 前后速度、`A/D` 横向速度、
`J/L` 转向速度、`Space` 清零命令、`R` 重置、`1` 持续步态、`2` 自动站立/行走。
默认模式为自动站立/行走：命令模长小于 0.05 且实际平面速度不超过 0.5 m/s 时，
时钟观测归零；其余时间按训练侧的 1.1/1.2 秒步态周期运行。

策略输入为 float32 `[1, 780]`（52 维 × 15 帧，从旧到新），输出为 `[1, 14]` 的
`action`，支持导出器附带的 `est` 速度估计。脚本按本项目 `f4_cfg_amp.py` 的默认参数配置：
物理步长 0.005 秒、每 4 步推理一次、动作缩放 0.25、训练 PD 增益与力矩限幅。
若训练时改过配置，应同步修改脚本中的 `Config`。

默认使用 `resources/robots/F4/scene_f4.xml`，运行时将 28 自由度模型的手臂关节固定在零位，
保留其质量和几何，与训练用的 `f4_14dof.urdf` 对齐。关节和电机按名称映射，腰部顺序为
**roll、yaw**。观测使用实际机体角速度和欧拉角；自动站走使用实际速度判断。
MuJoCo 接触模型与 Isaac Gym 不同，最终行走效果需用自己的 F4 ONNX 验证。

```bash
# 验证观测排列、步态、控制频率、力矩限制、重置和 ONNX 接口
python -m unittest discover -s humanoid/scripts -p 'test_sim2sim_f4.py'
```

## Troubleshooting

Observe the following cases:

```bash
# error
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
# solution
# set the correct path
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
# OR
sudo apt install libpython3.8
```

## Acknowledgment

The implementation of Humanoid-Gym relies on resources from [legged_gym](https://github.com/leggedrobotics/legged_gym) and [rsl_rl](https://github.com/leggedrobotics/rsl_rl) projects, created by the Robotic Systems Lab. We specifically utilize the `LeggedRobot` implementation from their research to enhance our codebase.
