"""注释
命令:
- `python -m Autobs.pretrain_policy --version single`

参数含义:
- `-v, --version`: 训练环境版本，不影响 action masking 的核心逻辑。
- 本文件负责把环境输出的 `action_mask` 融入 PPO 策略 logits，仅保留训练过程需要的最小实现。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.spaces import Box, Dict, Discrete
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR, ENCODER_OUT
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorStructType

from Autobs.utils import ACTION_SPACE_SIZE, MAP_SIZE


def build_single_site_spaces(map_size: int = MAP_SIZE, action_space_size: int = ACTION_SPACE_SIZE):
    """注释
    功能: 构建单站点初始化策略模块所需的观测空间与动作空间。
    输入: `map_size`为归一化地图边长，`action_space_size`为动作网格边长。
    输出: `(observation_space, action_space)`元组。
    示例: `obs_space, act_space = build_single_site_spaces()`。
    时间: 2026-04-27。
    """
    action_space = Discrete(int(action_space_size) ** 2)
    observation_space = Dict(
        {
            "observations": Box(low=0.0, high=1.0, shape=(int(map_size) ** 2,), dtype=np.float32),
            "action_mask": Box(low=0.0, high=1.0, shape=(action_space.n,), dtype=np.float32),
        }
    )
    return observation_space, action_space


class _CNNActorCriticEncoder(th.nn.Module):
    def __init__(self, observation_space, model_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        if observation_space is None or not getattr(observation_space, "shape", None):
            raise ValueError("Observation space with a concrete shape is required for CNN encoder")

        model_config = model_config or {}
        flat_dim = int(observation_space.shape[0])
        self.in_channels, self.map_size = self._infer_channels_and_map_size(flat_dim)
        channels = list(model_config.get("cnn_channels", [32, 64, 128, 128]))
        blocks: list[th.nn.Module] = []
        in_ch = self.in_channels
        for out_ch in channels:
            blocks.extend(
                [
                    th.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    th.nn.BatchNorm2d(out_ch),
                    th.nn.ReLU(inplace=True),
                ]
            )
            in_ch = out_ch
        self.backbone = th.nn.Sequential(*blocks)
        self.actor_out_channels = in_ch

    @staticmethod
    def _infer_channels_and_map_size(flat_dim: int) -> tuple[int, int]:
        for channels in (1, 2, 3, 4):
            if flat_dim % channels != 0:
                continue
            side = int(round((flat_dim / channels) ** 0.5))
            if side * side * channels == flat_dim:
                return channels, side
        raise ValueError(f"Cannot infer CNN input shape from flattened dimension: {flat_dim}")

    def _reshape_obs(self, obs: th.Tensor) -> th.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return obs.view(obs.shape[0], self.in_channels, self.map_size, self.map_size)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        obs = batch[Columns.OBS]
        if not isinstance(obs, th.Tensor):
            obs = th.as_tensor(obs, dtype=th.float32)
        obs = obs.float()
        spatial_features = self.backbone(self._reshape_obs(obs))
        return {ENCODER_OUT: {ACTOR: spatial_features}}


class _HeatmapPolicyHead(th.nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, action_side: int) -> None:
        super().__init__()
        self.action_side = int(action_side)
        conv_hidden = max(int(hidden_dim // 2), 32)
        dropout = 0.15
        self.net = th.nn.Sequential(
            th.nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            th.nn.ReLU(inplace=True),
            th.nn.Dropout2d(p=dropout),
            th.nn.Conv2d(hidden_dim, conv_hidden, kernel_size=3, padding=1),
            th.nn.ReLU(inplace=True),
            th.nn.Dropout2d(p=dropout),
            th.nn.Upsample(size=(self.action_side, self.action_side), mode="bilinear", align_corners=False),
            # 通道数为1，输出一个动作打分图，后续会被 flatten 成动作分数向量
            th.nn.Conv2d(conv_hidden, 1, kernel_size=1),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x).flatten(start_dim=1)


class ActionMaskPolicyModule(DefaultPPOTorchRLModule):
    def __init__(
        self,
        observation_space=None,
        action_space=None,
        inference_only=None,
        learner_only: bool = False,
        model_config=None,
        catalog_class=None,
        **kwargs,
    ):
        model_cfg = kwargs.pop("model_config_dict", None) or model_config or {}
        if isinstance(observation_space, gym.spaces.Dict):
            observation_space = observation_space["observations"]
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_cfg,
            catalog_class=catalog_class,
        )

    def setup(self):
        """注释
        功能: 初始化当前Autobs监督学习主线所需的actor编码器与heatmap策略头。
        输入: 无，依赖模块自身的`observation_space`、`action_space`与`model_config`。
        输出: 无，副作用是挂载`encoder`与`pi`网络模块。
        示例: `module.setup()`会完成单站点动作策略网络初始化。
        时间: 2026-04-28。
        """
        self.encoder = _CNNActorCriticEncoder(self.observation_space, self.model_config)
        head_hidden_dim = int(self.model_config.get("head_hidden_dim", 256))
        action_side = int(round(int(self.action_space.n) ** 0.5))
        if action_side * action_side != int(self.action_space.n):
            raise ValueError(f"Action space size must be a perfect square, got {self.action_space.n}")
        self.pi = _HeatmapPolicyHead(self.encoder.actor_out_channels, head_hidden_dim, action_side)

    def _forward_inference(self, batch: TensorStructType, **kwargs) -> Mapping[str, Any]:
        """注释
        功能: 执行推理阶段的actor前向传播，并对非法动作施加mask。
        输入: `batch`为RLModule批量输入，`kwargs`为保留扩展参数。
        输出: 含`SampleBatch.ACTION_DIST_INPUTS`的logits字典。
        示例: `outputs = module._forward_inference(batch)`。
        时间: 2026-04-28。
        """
        return mask_forward_fn(self._forward_actor, batch, **kwargs)

    def _forward_exploration(self, batch: TensorStructType, **kwargs) -> Mapping[str, Any]:
        """注释
        功能: 执行探索阶段的actor前向传播，并对非法动作施加mask。
        输入: `batch`为RLModule批量输入，`kwargs`为保留扩展参数。
        输出: 含`SampleBatch.ACTION_DIST_INPUTS`的logits字典。
        示例: `outputs = module._forward_exploration(batch)`。
        时间: 2026-04-28。
        """
        return mask_forward_fn(self._forward_actor, batch, **kwargs)

    def _forward_train(self, batch: TensorStructType, **kwargs) -> Mapping[str, Any]:
        """注释
        功能: 执行监督训练阶段的actor前向传播，并对非法动作施加mask。
        输入: `batch`为RLModule批量输入，`kwargs`为保留扩展参数。
        输出: 含`SampleBatch.ACTION_DIST_INPUTS`的logits字典。
        示例: `outputs = module._forward_train(batch)`。
        时间: 2026-04-28。
        """
        return mask_forward_fn(self._forward_actor, batch, **kwargs)

    def compute_values(self, batch, **kwargs):
        """注释
        功能: 显式阻止当前监督学习版策略模块进入value function计算链路。
        输入: `batch`为可能来自RLlib的批量观测字典，`kwargs`为保留扩展参数。
        输出: 无；总是抛出`RuntimeError`提示当前模块不支持value估计。
        示例: `module.compute_values(batch)`会直接报错。
        时间: 2026-04-28。
        """
        raise RuntimeError("ActionMaskPolicyModule no longer provides value-function computation in Autobs")

    def _forward_actor(self, batch: TensorStructType, **kwargs) -> Mapping[str, Any]:
        """注释
        功能: 执行共享actor主干与heatmap策略头前向传播，只生成动作logits。
        输入: `batch`为已剥离动作掩码后的观测批量，`kwargs`为保留扩展参数。
        输出: 含`SampleBatch.ACTION_DIST_INPUTS`的原始动作logits字典。
        示例: `outputs = module._forward_actor(batch)`。
        时间: 2026-04-28。
        """
        del kwargs
        encoder_outs = self.encoder(batch)
        return {SampleBatch.ACTION_DIST_INPUTS: self.pi(encoder_outs[ENCODER_OUT][ACTOR])}

# {
#     "observations": ...,
#     "action_mask": [1, 1, 0, 0, 1, 0, 1, ...]
# }
# 环境告诉模型：当前哪些动作能选，哪些不能选
# 模型先照常算出所有动作的分数
# 然后把“不能选”的动作分数改成极小
# 这样PPO在采样和训练时，就等于只在合法动作里做决策
# BaseEnvironment类在reset()/step()里返回的observation dict
def mask_forward_fn(forward_fn, batch, **kwargs):
    #     batch = {
    #     "obs": {      # SampleBatch.OBS == "obs"
    #         "observations": ...,
    #         "action_mask": ...
    #     },
    #     ...
    # }
    obs_batch = batch[SampleBatch.OBS]
    action_mask = None
    local_batch = dict(batch)
    if isinstance(obs_batch, Mapping) and "observations" in obs_batch:
        local_batch[SampleBatch.OBS] = obs_batch["observations"]
        action_mask = obs_batch.get("action_mask")

    outputs = forward_fn(local_batch, **kwargs)
    logits = outputs[SampleBatch.ACTION_DIST_INPUTS]
    logits = th.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=FLOAT_MIN)

    if action_mask is not None:
        action_mask = th.as_tensor(action_mask, dtype=logits.dtype, device=logits.device)
        action_mask = th.nan_to_num(action_mask, nan=0.0, posinf=0.0, neginf=0.0)
        legal_mask = action_mask > 0.0
        has_legal_action = th.any(legal_mask, dim=-1, keepdim=True)
        legal_mask = th.where(has_legal_action, legal_mask, th.ones_like(legal_mask, dtype=th.bool))
        inf_mask = th.where(legal_mask, th.zeros_like(logits), th.full_like(logits, FLOAT_MIN))
        logits = logits + inf_mask

    outputs[SampleBatch.ACTION_DIST_INPUTS] = logits
    return outputs
