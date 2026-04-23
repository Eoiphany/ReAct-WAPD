"""注释
命令:
- `Python -m Autobs.train_ppo -v single -r coverage`

参数含义:
- `-v single`: 使用单站点 PPO 训练环境。
- `-r, --reward_type`: 指定 `coverage`、`spectral_efficiency`、`score` 或兼容别名 `capacity`。
- 环境每个 episode 只放置 1 个站点，动作采用降采样动作网格并带 action mask。
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from PIL import Image
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Box, Dict, Discrete

from Autobs.env.utils import (
    ACTION_SPACE_SIZE,
    DEFAULT_COVERAGE_TARGET,
    DEFAULT_SPECTRAL_EFFICIENCY_TARGET,
    MAP_SIZE,
    calc_action_mask,
    calc_upsampling_loc,
    get_stats,
    load_heuristic_targets,
    load_map_normalized,
    lookup_heuristic_targets,
    resolve_city_map_paths,
    select_reward,
)
from Autobs.paths import DEFAULT_CITY_MAP_PATH, DEFAULT_DATASET_MAP_DIR
from Autobs.pmnet_adapter import infer_surrogate

# BaseEnvironment继承自gym.Env，必须实现至少两个核心函数：
# reset()
# step()
class BaseEnvironment(gym.Env):
    def __init__(self, config: dict) -> None:
        self.version = "single"
        self.reward_type = config.get("reward_type", "coverage")
        self.map_size = config.get("map_size", MAP_SIZE)
        # 默认动作网格边长
        self.action_space_size = config.get("action_space_size", ACTION_SPACE_SIZE)
        self.city_map_path = config.get("city_map_path", str(DEFAULT_DATASET_MAP_DIR))
        self.dataset_limit = config.get("dataset_limit")
        self.dataset_offset = int(config.get("dataset_offset", 0))
        self.dataset_stride = int(config.get("dataset_stride", 1))
        self.heuristic_targets_path = config.get("heuristic_targets_path")
        self.model_path = config.get("model_path")
        self.network_type = config.get("network_type", "pmnet")
        self.default_coverage_target = float(config.get("coverage_target", DEFAULT_COVERAGE_TARGET))
        self.default_spectral_efficiency_target = float(
            config.get("spectral_efficiency_target", DEFAULT_SPECTRAL_EFFICIENCY_TARGET)
        )
        # Autobs/config.yaml 中传入的是256 所以裁剪的功能相当于没有，实际上还学的是整图决策
        self.crop_size = config.get("crop_size", 512)
        self.stride = config.get("stride", 100)

        # 动作空间定义 智能体不是从 32 个动作里选，而是从 1024 个离散动作里选 动作编号范围是 0 ~ action_space_size^2 - 1
        self.action_space = Discrete(self.action_space_size ** 2)
        self.observation_space = Dict(
            {
                # 注意 单站点是一个图片 只有一个地图 和多站点不同
                "observations": Box(low=0.0, high=1.0, shape=(self.map_size ** 2,), dtype=np.float32),
                # 注意 这里的mask并不是RoI区域掩码，只是把所有动作空间中合理的动作输出出来了
                "action_mask": Box(low=0.0, high=1.0, shape=(self.action_space.n,), dtype=np.float32),
            }
        )

        self._city_map_paths = resolve_city_map_paths(
            self.city_map_path,
            DEFAULT_CITY_MAP_PATH,
            dataset_limit=self.dataset_limit,
            dataset_offset=self.dataset_offset,
            dataset_stride=self.dataset_stride,
        )
        self._map_index = -1
        self._current_map_path = None
        self._city_map = None
        self._pixel_map = None
        self._heuristic_targets = load_heuristic_targets(self.heuristic_targets_path)
        self._tx_locs = []
        self._steps = 0

    def _advance_city_map(self) -> None:
        if len(self._city_map_paths) == 1:
            map_path = self._city_map_paths[0]
        else:
            # _map_index初始是-1，需要+1
            self._map_index = (self._map_index + 1) % len(self._city_map_paths)
            map_path = self._city_map_paths[self._map_index]
        if map_path != self._current_map_path:
            self._city_map = load_map_normalized(map_path)
            self._current_map_path = map_path

    def _sample_crop(self) -> np.ndarray:
        height, width = self._city_map.shape
        crops_per_row = (width - self.crop_size) // self.stride + 1
        total_crops = crops_per_row * ((height - self.crop_size) // self.stride + 1)
        crop_id = np.random.randint(0, total_crops)
        row = crop_id // crops_per_row
        col = crop_id % crops_per_row
        top = row * self.stride
        left = col * self.stride
        crop = self._city_map[top : top + self.crop_size, left : left + self.crop_size]
        if crop.shape != (self.map_size, self.map_size):
            image = Image.fromarray((crop * 255).astype(np.uint8))
            crop = np.asarray(image.resize((self.map_size, self.map_size)), dtype=np.float32) / 255.0
        return crop

    # 每次reset到下一个episode后都会更新数据集的图片路径，并且选取其中一个crop
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self._tx_locs = []
        self._steps = 0
        self._advance_city_map()
        self._pixel_map = self._sample_crop()
        obs = np.clip(self._pixel_map.reshape(-1), 0.0, 1.0).astype(np.float32)
        mask = calc_action_mask(self._pixel_map).astype(np.float32)
        # obs是展平后的地图，act_mask是哪些位置可以放站
        return {"observations": obs, "action_mask": mask}, {}

    # 执行一个动作（放一个站），计算奖励，然后结束 episode
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._steps += 1
        self._tx_locs.append(calc_upsampling_loc(int(action), self._pixel_map))
        coverage_target, spectral_efficiency_target = lookup_heuristic_targets(
            self._heuristic_targets,
            self._current_map_path,
            self.default_coverage_target,
            self.default_spectral_efficiency_target,
        )

        # evaluate
        _pathgain_db, metrics = get_stats(
            self._pixel_map,
            self._tx_locs,
            pmnet=lambda inputs: infer_surrogate(inputs, model_path=self.model_path, network_type=self.network_type),
            coverage_target=coverage_target,
            spectral_efficiency_target=spectral_efficiency_target,
        )
        reward = select_reward(metrics, self.reward_type)
        # 注意只是把obs整合成一维的了，_pixel_map没有动，reshape是保存副本
        obs = np.clip(self._pixel_map.reshape(-1), 0.0, 1.0).astype(np.float32)
        mask = calc_action_mask(self._pixel_map).astype(np.float32)
        info = {
            "accumulated_reward": reward,
            "n_bs": 1,
            "steps": self._steps,
            "coverage_target": coverage_target,
            "spectral_efficiency_target": spectral_efficiency_target,
            **metrics,
        }
        # 下一时刻obs，奖励，terminated正常结束，truncated外部中断，附加信息字典
        return {"observations": obs, "action_mask": mask}, reward, True, False, info
