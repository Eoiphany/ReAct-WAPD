"""注释
命令:
- `Python -m Autobs.train_ppo -v multi`

参数含义:
- `-v, --version`: 训练多站点环境时，回调会统计每轮部署的累积奖励。
- 本文件只保留 PPO 训练日志聚合实际需要的回调逻辑。
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import gymnasium as gym
import numpy as np
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module import RLModule
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.single_agent_episode import SingleAgentEpisode as Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import EpisodeType, PolicyID


class AsyncActionCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        episode: Union[EpisodeType, Episode, EpisodeV2],
        worker: Optional["EnvRunner"] = None,
        env_runner: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        env: Optional[gym.Env] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        rl_module: Optional[RLModule] = None,
        env_index: int,
        **kwargs,
    ):
        if hasattr(episode, "hist_data"):
            episode.hist_data["reward_per_round"] = []

    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeType, Episode, EpisodeV2],
        worker: Optional["EnvRunner"] = None,
        env_runner: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        env: Optional[gym.Env] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        rl_module: Optional[RLModule] = None,
        env_index: int,
        **kwargs,
    ):
        if not hasattr(episode, "last_info_for"):
            return
        last_info = episode.last_info_for()
        accumulated_reward = last_info.get("accumulated_reward")
        n_bs = last_info.get("n_bs")
        steps = last_info.get("steps")
        if accumulated_reward is None or n_bs is None or steps is None:
            return
        reward_per_round = accumulated_reward / np.ceil(steps)
        episode.custom_metrics["reward_per_round"] = reward_per_round
        if hasattr(episode, "hist_data"):
            episode.hist_data["reward_per_round"].append(reward_per_round)

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        sampler_results = result.get("sampler_results")
        if not sampler_results:
            return
        num_episodes = sampler_results.get("episodes_this_iter", 0)
        reward_per_round = sampler_results.get("hist_stats", {}).get("reward_per_round", [])[-num_episodes:]
        if reward_per_round:
            result["custom_metrics"]["reward_per_round_std"] = float(np.std(reward_per_round))
            result["custom_metrics"]["reward_per_round_mean"] = float(np.mean(reward_per_round))

