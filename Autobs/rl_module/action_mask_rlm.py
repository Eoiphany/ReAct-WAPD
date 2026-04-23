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
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorStructType


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
