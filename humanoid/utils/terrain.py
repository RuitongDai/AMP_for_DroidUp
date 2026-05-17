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


import numpy as np

from isaacgym import terrain_utils
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

import pyfqmr
from scipy.ndimage import binary_dilation


def make_difficulty(value, difficulty):
    return value[0] + (value[1] - value[0]) * difficulty


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.terrain_num = np.zeros(7, dtype=np.int16)
        self.terrain_type_name = [None] * self.cfg.num_cols
        # --- 找出 plane 的 type_idx ---
        self.plane_type_idx = []
        self.slope_type_idx = []
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        # if self.type == "trimesh":
        #     self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
        #                                                                                  self.cfg.horizontal_scale,
        #                                                                                  self.cfg.vertical_scale,
        #                                                                                  self.cfg.slope_treshold)
        #     print("Created {} vertices".format(self.vertices.shape[0]))
        #     print("Created {} triangles".format(self.triangles.shape[0]))

        if self.type=="trimesh":
            print("Converting heightmap to trimesh...")
            self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
            half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.horizontal_scale)
            structure = np.ones((half_edge_width*2+1, 1))
            self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)
            if self.cfg.simplify_grid:
                mesh_simplifier = pyfqmr.Simplify()
                mesh_simplifier.setMesh(self.vertices, self.triangles)
                mesh_simplifier.simplify_mesh(target_count = int(0.05*self.triangles.shape[0]), aggressiveness=7, preserve_border=True, verbose=10)

                self.vertices, self.triangles, normals = mesh_simplifier.getMesh()
                self.vertices = self.vertices.astype(np.float32)
                self.triangles = self.triangles.astype(np.uint32)
            print("Created {} vertices".format(self.vertices.shape[0]))
            print("Created {} triangles".format(self.triangles.shape[0]))

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 1.0])
            terrain, _ = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / (self.cfg.num_rows - 1)
                choice = j / self.cfg.num_cols + 0.001

                terrain, name = self.make_terrain(choice, difficulty)
                self.terrain_type_name[j] = name
                self.add_terrain_to_map(terrain, i, j)
        self.plane_type_idx = [i for i, name in enumerate(self.terrain_type_name) if name in ["plane"]]
        self.slope_type_idx = [i for i, name in enumerate(self.terrain_type_name) if name in ["smooth slope", "rough slope", "wave"]]
        print("terrain_type_name:", self.terrain_type_name)
        print("plane_type_idx:", self.plane_type_idx)
        print("slope_type_idx:", self.slope_type_idx)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                                               width=self.width_per_env_pixels,
                                               length=self.width_per_env_pixels,
                                               vertical_scale=self.vertical_scale,
                                               horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=self.width_per_env_pixels,
                                           length=self.width_per_env_pixels,
                                           vertical_scale=self.cfg.vertical_scale,
                                           horizontal_scale=self.cfg.horizontal_scale)

        slope = make_difficulty(self.cfg.slop_range, difficulty)
        step_height = make_difficulty(self.cfg.step_height, difficulty)
        step_height_1 = make_difficulty(self.cfg.step_height_1, difficulty)
        discrete_obstacles_height = make_difficulty(self.cfg.discrete_obstacles_height, difficulty)
        step_stone =  make_difficulty(self.cfg.step_stone, difficulty)
        step_width = self.cfg.step_width
        amplitude = make_difficulty(self.cfg.wave_amplitude, difficulty)
        grid_height = make_difficulty([0.0, 0.1], difficulty)
        type_id = 0
        if choice < self.proportions[0]:
            type_id = 0
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=2.)
        elif choice < self.proportions[1]:
            type_id = 1
            if choice < self.proportions[1] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=2.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.02, max_height=0.02, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            type_id = 3
            if choice < self.proportions[2]:
                type_id = 2
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.02, max_height=0.02, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[4]:
            type_id = 4
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles,
                                                             platform_size=3.)
        elif choice < self.proportions[6]:
            type_id = 6
            if choice < self.proportions[5]:
                type_id = 5
                step_height_1 *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=1.0, step_height=step_height_1, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.02, max_height=0.02, step=0.005, downsampled_scale=0.2)
        elif  choice < self.proportions[7]:
            type_id = 7
            terrain_utils.stepping_stones_terrain(terrain, stone_size=1, stone_distance=1, max_height=0, platform_size=1., depth=-step_stone)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.02, max_height=0.02, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[8]:
            type_id = 8
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.00, max_height=0.00, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[9]:
            type_id = 9
            terrain_utils.wave_terrain(terrain, num_waves=2, amplitude=amplitude)
        else:
            type_id = 10
            random_grid_heightfield(terrain=terrain, grid_height=grid_height, grid_width=1,  platform_size=2.)

        terrain_name = list(self.cfg.terrain_dict.keys())
        return terrain, terrain_name[type_id]

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length / 2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x - x2: center_x + x2, center_y - y2: center_y + y2] = -1000
    terrain.height_field_raw[center_x - x1: center_x + x1, center_y - y1: center_y + y1] = 0


def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def random_grid_heightfield(terrain, grid_height, grid_width,  platform_size=1.):
    grid_width = int(grid_width/ terrain.horizontal_scale)
    # --- 网格数量 ---
    num_boxes_x = int(terrain.width / grid_width)
    num_boxes_y = int(terrain.length / grid_width)
    # --- 初始化 height map ---
    height_field = np.zeros((num_boxes_x, num_boxes_y), dtype=np.float32)

    # --- 生成随机格栅高度 ---
    height_field += np.random.uniform(-grid_height, grid_height, size=height_field.shape)

    # --- 添加中间平台 ---
    platform_size = int(platform_size / terrain.horizontal_scale)
    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0

    # --- 适配到 Isaac Gym height field 分辨率 ---
    hf_w, hf_h = terrain.height_field_raw.shape
    # 缩放随机格栅到 height field 分辨率
    height_field_rescaled = np.kron(
        height_field,
        np.ones((hf_w // num_boxes_x, hf_h // num_boxes_y), dtype=np.float32)
    )
    height_field_rescaled = height_field_rescaled[:hf_w, :hf_h]
    terrain.height_field_raw[:, :] = height_field_rescaled / terrain.vertical_scale
    return terrain

def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles, move_x != 0