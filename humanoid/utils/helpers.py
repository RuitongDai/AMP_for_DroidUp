# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
import shutil

def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        # TODO sort by date to handle change of month
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint
        if args.use_wandb:
            cfg_train.runner.use_wandb = args.use_wandb

    return env_cfg, cfg_train


def get_args():
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "x2_vision",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--resume",
            "action": "store_true",
            "default": False,
            "help": "Resume training from a checkpoint",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--run_name",
            "type": str,
            "help": "Name of the run. Overrides config file if provided.",
        },
        {
            "name": "--load_run",
            "type": str,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": False,
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--num_envs",
            "type": int,
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--seed",
            "type": int,
            "help": "Random seed. Overrides config file if provided.",
        },
        {
            "name": "--max_iterations",
            "type": int,
            "help": "Maximum number of training iterations. Overrides config file if provided.",
        },
        {
            "name": "--use_wandb",
            "action": "store_true",
            "default": False,
            "help": "save learning date",
        },
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy", custom_parameters=custom_parameters
    )

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path, name):
    if hasattr(actor_critic, 'estimator_vel'):
        exporter = PolicyExporterEST(actor_critic)
        exporter.export_onnx(path, name)
    elif hasattr(actor_critic, 'hist_encoder'):
        exporter = PolicyExporterBlindIE(actor_critic)
        exporter.export_onnx(path, name)
    elif hasattr(actor_critic,'depth_encoder') and hasattr(actor_critic,'height_decoder'):
        exporter = PolicyExporterVisionMLP(actor_critic)
        exporter.export_onnx(path, name)
    elif hasattr(actor_critic, 'depth_encoder'):
        exporter = PolicyExporterVision(actor_critic)
        exporter.export_onnx(path, name)
    elif hasattr(actor_critic, 'estimator_lstm'):
        exporter = PolicyExporterEST_LSTM(actor_critic)
        exporter.export(path, name)
    elif hasattr(actor_critic, 'estimator_feat'):
        exporter = PolicyExporterFeat(actor_critic)
        exporter.export_onnx(path, name)
    else:
        os.makedirs(path, exist_ok=True)
        path0 = os.path.join(path, "policy_1.pt")
        model = copy.deepcopy(actor_critic.actor).to("cpu")
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path0)
        path1 = os.path.join(path, name)
        traced_script_module.save(path1)

# def export_policy_as_jit(actor_critic, path, name):
#     os.makedirs(path, exist_ok=True)
#     file_path_pt = os.path.join(path, "policy_1.pt")
#     model = copy.deepcopy(actor_critic.actor).to("cpu")
#     traced_script_module = torch.jit.script(model)
#     traced_script_module.save(file_path_pt)
#     file_path_pt1 = os.path.join(path, name + '.pt')
#     traced_script_module.save(file_path_pt1)

def export_policy_as_onnx(actor_critic, path, name):
    os.makedirs(path, exist_ok=True)
    file_path_onnx = os.path.join(path, "policy_1.onnx")
    file_path_onnx1 = os.path.join(path, name + ".onnx")
    batch_size = 1
    actor_input = torch.rand(batch_size, actor_critic.num_actor_inputs, device='cuda')
    torch.onnx.export(actor_critic.actor, actor_input, file_path_onnx,  # save model as onnx module
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['action','est'],
                      dynamic_axes={
                          'input': {0: 'batch_size'},
                          'actions': {0: 'batch_size'},
                          "est": {0: "batch_size"}
                      }
                      )
    shutil.copy(file_path_onnx, file_path_onnx1)

class PolicyExporterEST(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor).to('cpu')
        self.estimator_vel = copy.deepcopy(actor_critic.estimator_vel).to('cpu')
        self.num_actor_obs = actor_critic.num_actor_obs
        self.num_long_obs = actor_critic.num_long_obs
        self.num_short_obs = actor_critic.num_short_obs

    def forward(self, observations):
        hist_input = observations[:, -self.num_long_obs:]
        encode = self.estimator_vel(hist_input)

        actor_input = torch.cat((observations[:, -self.num_short_obs:], encode), dim=-1)
        actions_mean = self.actor(actor_input)
        est_vel = encode[:, 0:3]
        return actions_mean, est_vel

    def export(self, path, name):
        os.makedirs(path, exist_ok=True)
        path0 = os.path.join(path, "policy_blind_est.pt")
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path0)
        path1 = os.path.join(path, name + '.pt')
        traced_script_module.save(path1)

    def export_onnx(self, path, name):
        os.makedirs(path, exist_ok=True)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        file_path_onnx = os.path.join(path, "policy_blind_est.onnx")
        file_path_onnx1 = os.path.join(path, name + ".onnx")
        batch_size = 1
        actor_input = torch.rand(batch_size, self.num_actor_obs)
        torch.onnx.export(traced_script_module, actor_input, file_path_onnx,  # save model as onnx module
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['action', 'est'],
                          dynamic_axes={
                              'input': {0: 'batch_size'},
                              'actions': {0: 'batch_size'},
                              "est": {0: "batch_size"}
                          }
                          )
        shutil.copy(file_path_onnx, file_path_onnx1)        

class PolicyExporterFeat(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor).to('cpu')
        self.estimator_feat = copy.deepcopy(actor_critic.estimator_feat).to('cpu')
        self.num_actor_obs = actor_critic.num_actor_obs
        self.num_long_obs = actor_critic.num_long_obs
        self.num_short_obs = actor_critic.num_short_obs

    def forward(self, observations):
        hist_input = observations[:, -self.num_long_obs:]
        encode_vel, encode_latent= self.estimator_feat(hist_input)

        actor_input = torch.cat((observations[:, -self.num_short_obs:], encode_vel, encode_latent), dim=-1)
        actions_mean = self.actor(actor_input)
        est_vel = encode_vel
        return actions_mean, est_vel

    def export(self, path, name):
        os.makedirs(path, exist_ok=True)
        path0 = os.path.join(path, "policy_blind_feat.pt")
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path0)
        path1 = os.path.join(path, name + '.pt')
        traced_script_module.save(path1)

    def export_onnx(self, path, name):
        os.makedirs(path, exist_ok=True)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        file_path_onnx = os.path.join(path, "policy_blind_feat.onnx")
        file_path_onnx1 = os.path.join(path, name + ".onnx")
        batch_size = 1
        actor_input = torch.rand(batch_size, self.num_actor_obs)
        torch.onnx.export(traced_script_module, actor_input, file_path_onnx,  # save model as onnx module
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['action', 'est'],
                          dynamic_axes={
                              'input': {0: 'batch_size'},
                              'actions': {0: 'batch_size'},
                              "est": {0: "batch_size"}
                          }
                          )
        shutil.copy(file_path_onnx, file_path_onnx1)

class PolicyExporterEST_LSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator_lstm = copy.deepcopy(actor_critic.estimator_lstm).to('cpu')
        self.num_actor_obs = actor_critic.num_actor_obs
        self.num_long_obs = actor_critic.num_long_obs
        self.num_short_obs = actor_critic.num_short_obs
        self.is_recurrent = actor_critic.is_recurrent
        self.memory_a = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory_a.cpu()
        self.register_buffer(f'hidden_a_state', torch.zeros(self.memory_a.num_layers, 1, self.memory_a.hidden_size))
        self.register_buffer(f'cell_a_state', torch.zeros(self.memory_a.num_layers, 1, self.memory_a.hidden_size))

        self.memory_e = copy.deepcopy(actor_critic.memory_e.rnn)
        self.memory_e.cpu()
        self.register_buffer(f'hidden_e_state', torch.zeros(self.memory_e.num_layers, 1, self.memory_e.hidden_size))
        self.register_buffer(f'cell_e_state', torch.zeros(self.memory_e.num_layers, 1, self.memory_e.hidden_size))

    def forward(self, observations):
        hist_input = observations[..., -self.num_long_obs:]
        input_e, (h, c) = self.memory_e(hist_input.unsqueeze(0), (self.hidden_e_state, self.cell_e_state))
        self.hidden_e_state[:] = h
        self.cell_e_state[:] = c
        encode = self.estimator_lstm(input_e.squeeze(0))

        actor_input = torch.cat((observations[..., -self.num_short_obs:], encode), dim=-1)
        input_a, (h, c) = self.memory_a(actor_input.unsqueeze(0), (self.hidden_a_state, self.cell_a_state))
        self.hidden_a_state[:] = h
        self.cell_a_state[:] = c
        actions_mean = self.actor(input_a.squeeze(0))
        est_vel = encode[..., 0:3]
        return actions_mean, est_vel

    @torch.jit.export
    def reset_memory(self):
        self.hidden_a_state[:] = 0.
        self.cell_a_state[:] = 0.
        self.hidden_e_state[:] = 0.
        self.cell_e_state[:] = 0.

    def export(self, path, name):
        os.makedirs(path, exist_ok=True)
        path0 = os.path.join(path, "policy_lstm_est.pt")
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path0)
        path1 = os.path.join(path, name + '.pt')
        traced_script_module.save(path1)

class PolicyExporterBlindIE(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor).to('cpu')
        self.hist_encoder = copy.deepcopy(actor_critic.hist_encoder).to('cpu')
        self.num_actor_obs = actor_critic.num_actor_obs
        self.num_long_obs = actor_critic.num_long_obs
        self.num_short_obs = actor_critic.num_short_obs

    def forward(self, observations):
        hist_input = observations[:, -self.num_long_obs:]
        encode = self.hist_encoder(hist_input)

        actor_input = torch.cat((observations[:, -self.num_short_obs:], encode), dim=-1)
        actions_mean = self.actor(actor_input)
        est_vel = encode[:, 0:3]
        return actions_mean, est_vel

    def export(self, path, name):
        os.makedirs(path, exist_ok=True)
        path0 = os.path.join(path, "policy_blind_IE.pt")
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path0)
        path1 = os.path.join(path, name + '.pt')
        traced_script_module.save(path1)

    def export_onnx(self, path, name):
        os.makedirs(path, exist_ok=True)
        self.to('cpu')
        file_path_onnx = os.path.join(path, "policy_blind_IE.onnx")
        file_path_onnx1 = os.path.join(path, name + ".onnx")
        batch_size = 1
        obs_input = torch.zeros(batch_size, self.num_actor_obs)
        torch.onnx.export(self,
                          obs_input,
                          file_path_onnx,  # save model as onnx module
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['action', 'est'],
                          dynamic_axes={
                              'input': {0: 'batch_size'},
                              'actions': {0: 'batch_size'},
                              "est": {0: "batch_size"}
                          }
                          )
        shutil.copy(file_path_onnx, file_path_onnx1)

class PolicyExporterVision(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor).to('cpu')
        self.depth_encoder = copy.deepcopy(actor_critic.depth_encoder).to('cpu')
        self.num_actor_obs = actor_critic.num_actor_obs
        self.num_long_obs = actor_critic.num_long_obs
        self.num_short_obs = actor_critic.num_short_obs
        self.depth_dim = actor_critic.depth_dim

    def forward(self, observations, depth):
        mlp_depth = depth.view(depth.shape[0], -1)
        depth_input = torch.cat([observations[:, -self.num_long_obs:], mlp_depth], dim=-1)
        encode, _ = self.depth_encoder(depth_input)

        actor_input = torch.cat((observations[:, -self.num_short_obs:], encode), dim=-1)
        actions_mean = self.actor(actor_input)
        est_vel = encode[:, 0:3]
        return actions_mean, est_vel

    def export(self, path, name):
        os.makedirs(path, exist_ok=True)
        path0 = os.path.join(path, "policy_vision.pt")
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path0)
        path1 = os.path.join(path, name + '.pt')
        traced_script_module.save(path1)

    def export_onnx(self, path, name):
        os.makedirs(path, exist_ok=True)
        self.to('cpu')
        file_path_onnx = os.path.join(path, "policy_vision.onnx")
        file_path_onnx1 = os.path.join(path, name + ".onnx")
        batch_size = 1
        obs_input = torch.zeros(batch_size, self.num_actor_obs)
        depth_input = torch.zeros(batch_size, self.depth_dim[0], self.depth_dim[1])
        torch.onnx.export(self,
                          (obs_input, depth_input),
                          file_path_onnx,  # save model as onnx module
                          do_constant_folding=True,
                          input_names=['obs', 'depth'],
                          output_names=['action', 'est'],
                          dynamic_axes={
                              'obs': {0: 'batch_size'},
                              'depth': {0: 'batch_size'},
                              'actions': {0: 'batch_size'},
                              "est": {0: "batch_size"}
                          }
                          )
        shutil.copy(file_path_onnx, file_path_onnx1)

class PolicyExporterVisionMLP(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor).to('cpu')
        self.depth_encoder = copy.deepcopy(actor_critic.depth_encoder).to('cpu')
        self.num_actor_obs = actor_critic.num_actor_obs
        self.num_long_obs = actor_critic.num_long_obs
        self.num_short_obs = actor_critic.num_short_obs
        self.depth_dim = actor_critic.depth_dim

    def forward(self, observations, depth):
        mlp_depth = depth.view(depth.shape[0], -1)
        depth_input = torch.cat([observations[:, -self.num_long_obs:], mlp_depth], dim=-1)
        encode = self.depth_encoder(depth_input)

        actor_input = torch.cat((observations[:, -self.num_short_obs:], encode), dim=-1)
        actions_mean = self.actor(actor_input)
        est_vel = encode[:, 0:3]
        return actions_mean, est_vel

    def export(self, path, name):
        os.makedirs(path, exist_ok=True)
        path0 = os.path.join(path, "policy_vision_mlp.pt")
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path0)
        path1 = os.path.join(path, name + '.pt')
        traced_script_module.save(path1)

    def export_onnx(self, path, name):
        os.makedirs(path, exist_ok=True)
        self.to('cpu')
        file_path_onnx = os.path.join(path, "policy_vision_mlp.onnx")
        file_path_onnx1 = os.path.join(path, name + ".onnx")
        batch_size = 1
        obs_input = torch.zeros(batch_size, self.num_actor_obs)
        depth_input = torch.zeros(batch_size, self.depth_dim[0], self.depth_dim[1])
        torch.onnx.export(self,
                          (obs_input, depth_input),
                          file_path_onnx,  # save model as onnx module
                          do_constant_folding=True,
                          input_names=['obs', 'depth'],
                          output_names=['action', 'est'],
                          dynamic_axes={
                              'obs': {0: 'batch_size'},
                              'depth': {0: 'batch_size'},
                              'actions': {0: 'batch_size'},
                              "est": {0: "batch_size"}
                          }
                          )
        shutil.copy(file_path_onnx, file_path_onnx1)

def create_point_list(resolution:float=0.01,
                      range_x:tuple=(-0.1, 0.1),
                      range_y:tuple=(-0.05, 0.05),
                      debug=False):

    num_sample_x = int(round((range_x[1] - range_x[0]) / resolution)) + 1
    num_sample_y = int(round((range_y[1] - range_y[0]) / resolution)) + 1
    point_x = np.linspace(range_x[0], range_x[1], num_sample_x)
    point_y = np.linspace(range_y[0], range_y[1], num_sample_y)
    if debug:
        print("point_x_shape:", point_x.shape)
        print("point_y_shape:", point_y.shape)
        print(point_x)
        print(point_y)
    return point_x.tolist(), point_y.tolist()