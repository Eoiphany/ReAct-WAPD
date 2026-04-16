"""注释
命令:
- `Python -m Autobs.train_ppo -v single`

参数含义:
- `-v, --version`: 训练环境版本，不影响 action masking 的核心逻辑。
- 本文件负责把环境输出的 `action_mask` 融入 PPO 策略 logits，仅保留训练过程需要的最小实现。
"""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import torch as th
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorStructType


class PPOActionMaskRLM(DefaultPPOTorchRLModule):
    def __init__(self, config: RLModuleConfig = None, observation_space=None, action_space=None, **kwargs):
        catalog_class = kwargs.pop("catalog_class", None)
        model_cfg = kwargs.get("model_config_dict") or kwargs.get("model_config") or {}
        if config is not None:
            observation_space = config.observation_space
            action_space = config.action_space
            model_cfg = config.model_config_dict or model_cfg
            catalog_class = config.catalog_class or catalog_class
        if isinstance(observation_space, gym.spaces.Dict):
            observation_space = observation_space["observations"]
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
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
    action_mask = batch[SampleBatch.OBS]["action_mask"]
    batch[SampleBatch.OBS] = batch[SampleBatch.OBS]["observations"]
    outputs = forward_fn(batch, **kwargs)
    logits = outputs[SampleBatch.ACTION_DIST_INPUTS]
    inf_mask = th.clamp(th.log(action_mask), min=FLOAT_MIN)
    # log -> 设置下界 -> 输出+掩码下界值（仅非掩码是0）
    outputs[SampleBatch.ACTION_DIST_INPUTS] = logits + inf_mask
    return outputs

