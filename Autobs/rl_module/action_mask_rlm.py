"""注释
命令:
- `Python -m Autobs.train_ppo -v single`

参数含义:
- `-v, --version`: 训练环境版本，不影响 action masking 的核心逻辑。
- 本文件负责把环境输出的 `action_mask` 融入 PPO 策略 logits，仅保留训练过程需要的最小实现。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import torch as th
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorStructType


class _CNNActorCriticEncoder(th.nn.Module):
    def __init__(self, observation_space, model_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        if observation_space is None or not getattr(observation_space, "shape", None):
            raise ValueError("Observation space with a concrete shape is required for CNN encoder")

        model_config = model_config or {}
        flat_dim = int(observation_space.shape[0])
        self.in_channels, self.map_size = self._infer_channels_and_map_size(flat_dim)
        channels = list(model_config.get("cnn_channels", [32, 64, 128, 128]))
        feature_dim = int(model_config.get("cnn_feature_dim", 256))
        self.feature_dim = feature_dim

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
        self.pool = th.nn.AdaptiveAvgPool2d((4, 4))
        self.proj = th.nn.Sequential(
            th.nn.Flatten(),
            th.nn.Linear(in_ch * 4 * 4, feature_dim),
            th.nn.ReLU(inplace=True),
        )

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
        features = self.proj(self.pool(self.backbone(self._reshape_obs(obs))))
        return {ENCODER_OUT: {ACTOR: features, CRITIC: features}}


class _MLPHead(th.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = th.nn.Module()
        self.net.mlp = th.nn.Sequential(
            th.nn.Linear(in_dim, hidden_dim),
            th.nn.ReLU(inplace=True),
            th.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net.mlp(x)


class PPOActionMaskRLM(DefaultPPOTorchRLModule):
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
        self.encoder = _CNNActorCriticEncoder(self.observation_space, self.model_config)
        # 两个 head 现在共享同一个 CNN encoder 输出特征，然后各自过一个小 MLP 头
        """
        也就是说，现在 actor/critic 的结构是：
        共享 CNN backbone
        共享投影层得到 feature_dim=256
        actor 一个两层 MLP head
        critic 一个两层 MLP head
        """
        feature_dim = int(getattr(self.encoder, "feature_dim", self.model_config.get("cnn_feature_dim", 256)))
        head_hidden_dim = int(self.model_config.get("head_hidden_dim", feature_dim))
        self.pi = _MLPHead(feature_dim, head_hidden_dim, int(self.action_space.n))
        self.vf = _MLPHead(feature_dim, head_hidden_dim, 1)

    def _forward_inference(self, batch: TensorStructType, **kwargs) -> Mapping[str, Any]:
        return mask_forward_fn(super()._forward_inference, batch, **kwargs)

    def _forward_exploration(self, batch: TensorStructType, **kwargs) -> Mapping[str, Any]:
        return mask_forward_fn(super()._forward_exploration, batch, **kwargs)

    def _forward_train(self, batch: TensorStructType, **kwargs) -> Mapping[str, Any]:
        return mask_forward_fn(super()._forward_train, batch, **kwargs)

    def compute_values(self, batch, **kwargs):
        obs = batch.get(SampleBatch.OBS)
        if isinstance(obs, dict) and "observations" in obs:
            batch = dict(batch)
            batch[SampleBatch.OBS] = obs["observations"]
        return super().compute_values(batch, **kwargs)

# {
#     "observations": ...,
#     "action_mask": [1, 1, 0, 0, 1, 0, 1, ...]
# }
# 环境告诉模型：当前哪些动作能选，哪些不能选
# 模型先照常算出所有动作的分数
# 然后把“不能选”的动作分数改成极小
# 这样 PPO 在采样和训练时，就等于只在合法动作里做决策
# BaseEnvironment类在 reset() / step() 里返回的 observation dict
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
