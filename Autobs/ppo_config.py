"""注释
命令:
- `python -m Autobs.train_ppo \
   -v single \
   -r coverage \
   --iters 10`

参数含义:
- `-v, --version`: 选择单站点或多站点环境。
- `-r, --reward_type`: 选择覆盖率或容量奖励。
- `--iters`: PPO 训练迭代次数。
- 本文件负责读取 `Autobs/config.yaml` 并构造 PPO 算法实例。
"""

from __future__ import annotations

from copy import deepcopy

import yaml

from Autobs.paths import CONFIG_PATH


def load_project_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dict_update(old_dict: dict, new_dict: dict) -> dict:
    updated = deepcopy(old_dict)
    updated.update(new_dict)
    return updated

# yaml.safe_load() 这里把lr读成了字符串 '1e-5'，不是 float
def _coerce_scalar_or_schedule(value):
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            return value
    return value


def get_ppo_config(base_environment, env_config_overrides: dict | None = None):
    import torch
    from ray.rllib.algorithms import PPOConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    from Autobs.env.callbacks import AsyncActionCallbacks
    # 在 PPO 原始策略输出上，强行把非法动作概率压成 0
    from Autobs.rl_module.action_mask_rlm import PPOActionMaskRLM

    config = load_project_config()
    # 从config的yaml文件中读取env的配置
    env_config = dict_update(config.get("env", {}), {"algo_name": "ppo"})
    if env_config_overrides:
        env_config = dict_update(env_config, env_config_overrides)
    train_config = config.get("train", {})
    lr = _coerce_scalar_or_schedule(train_config.get("lr", 1e-5))

    ppo_config = (
        PPOConfig()
        .environment(env=base_environment, env_config=env_config, disable_env_checking=True)
        # 单线程 每次采样50步
        .env_runners(
            num_env_runners=0,
            rollout_fragment_length=50,
            sample_timeout_s=120,
        )
        .framework("torch")
        .resources(num_gpus=torch.cuda.device_count())
        .training(
            gamma=train_config.get("gamma", 0.99),
            lr=lr,
            train_batch_size=train_config.get("train_batch_size", 2048),
            minibatch_size=train_config.get("sgd_minibatch_size", 256),
            # 每个iteration里，同一批train_batch_size样本被反复学习多少遍
            num_epochs=train_config.get("num_epochs", 5),
            clip_param=train_config.get("clip_param", 0.1),
        )
        .callbacks(AsyncActionCallbacks)
        .debugging(log_sys_usage=False)
    )
    # 智能体从action_space_size中选1个动作执行，但是这action_space_size个动作不一定全是合法的
    # 就比如动作按理说应该在RoI区域选择，如果动作生成在了非RoI区域，就不合法了
    # 这时候就需要动作掩码来屏蔽掉这些非法动作，PPOActionMaskRLM就是专门为这个目的设计的
    if not env_config.get("no_masking", True):
        ppo_config = ppo_config.rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=PPOActionMaskRLM,
                # 四层128维全连接+RELU激活
                model_config=train_config.get("model", {}),
            ),
        )

    return ppo_config.build_algo()
