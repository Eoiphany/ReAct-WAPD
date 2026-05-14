"""注释
命令:
python -m Autobs.pretrain_policy \
   --dataset /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/outputs/action_rank.npz \
   --version single \
   --epochs 1000 \
   --batch-size 32 \
   --lr 1e-4 \
   --val-ratio 0.1 \
   --geom-augment \
   --target-temperature 0.5 \
   --output-dir /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/pretrained_policy

参数含义:
- `--dataset`: 由 `build_action_rank_dataset.py` 生成的 `.npz` 数据集。
- `--version`: 预训练对应的环境版本，默认 `single`。
- `--epochs`: 监督训练轮数。
- `--batch-size`: 每个监督 batch 的地图样本数。
- `--lr`: 预训练学习率。
- `--val-ratio`: 验证集比例；若样本数足够，会从数据集中划出一部分只用于监控 `val_loss / val_top1`。
- `--patience`: 基于验证集 `loss` 的早停 patience；小于等于 0 时关闭早停。
- `--geom-augment`: 训练阶段开启几何数据增强；会对地图观测和动作标签同步做翻转/旋转。
- `--target-temperature`: 将动作分数转成 soft target 时的 softmax 温度。
- `--device`: 预训练使用的 torch 设备。
- `--output-dir`: 输出目录，会保存 `best_module_state.pt` 与训练摘要 JSON。

脚本逻辑说明:
本脚本复用同一套动作掩码策略模块结构，对离线动作排序数据集做监督预训练。
监督目标不是 one-hot 最优动作，而是由动作分数 softmax 得到的软分布，从而保留
“同图合法动作的相对好坏”。训练完成后输出可直接通过
后续策略学习脚本通过模块状态方式载入的策略权重。若启用验证集，则最佳权重按
`val_loss` 选择，而不是按训练集 `loss` 选择；同时会额外汇报 `val_top5 / val_top10`，
便于观察排序质量是否超过单纯的 `top1` 命中率。
"""

from __future__ import annotations

import argparse
import json
import math
import yaml
from pathlib import Path
from typing import Any

import numpy as np
import torch


def build_parser() -> argparse.ArgumentParser:
    """注释
    功能: 构建监督预训练脚本的命令行参数解析器。
    输入: 无。
    输出: `argparse.ArgumentParser`，包含数据集路径、训练超参数、增强与输出配置。
    示例: `parser = build_parser()`。
    时间: 2026-04-27。
    """
    parser = argparse.ArgumentParser(description="Supervised pretraining for action-mask policy from legal-action ranking dataset")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--version", default="single", choices=["single"])
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--patience", default=100, type=int)
    parser.add_argument("--geom-augment", action="store_true")
    parser.add_argument("--target-temperature", default=0.2, type=float)
    parser.add_argument("--pairwise-coeff", default=0.5, type=float)
    parser.add_argument("--reward-coeff", default=0.2, type=float)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    return parser


def scores_to_target_probs(action_scores: np.ndarray, action_masks: np.ndarray, temperature: float) -> np.ndarray:
    """注释
    功能: 将逐动作监督分数转换为软标签概率分布，并自动屏蔽非法动作。
    输入: `action_scores` 为逐动作分数，`action_masks` 为合法动作掩码，`temperature` 为 softmax 温度。
    输出: 与动作维度同形状的 soft target 概率数组。
    示例: `target_probs = scores_to_target_probs(action_scores, action_masks, 0.5)`。
    时间: 2026-04-27。
    """
    if action_scores.shape != action_masks.shape:
        raise ValueError("action_scores and action_masks must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    valid = action_masks > 0.0
    # 合法动作除以温度系数，不合法动作设为 -inf
    scaled = np.where(valid, action_scores / temperature, -np.inf)
    # isfinite只包含正无穷
    finite_max = np.max(np.where(np.isfinite(scaled), scaled, -1e30), axis=1, keepdims=True)
    exp = np.where(valid, np.exp(scaled - finite_max), 0.0)
    denom = np.sum(exp, axis=1, keepdims=True)
    fallback = np.where(valid, 1.0, 0.0)
    # 最大值为1或者合法的动作数
    fallback /= np.maximum(np.sum(fallback, axis=1, keepdims=True), 1.0)
    return np.where(denom > 0.0, exp / denom, fallback).astype(np.float32)


def _resolve_device(device_name: str) -> torch.device:
    """注释
    功能: 将字符串设备名解析为实际 `torch.device`，并在 `auto` 模式下自动选择可用设备。
    输入: `device_name` 为 `auto/cpu/cuda/mps` 之一。
    输出: 对应的 `torch.device` 对象。
    示例: `device = _resolve_device("auto")`。
    时间: 2026-04-27。
    """
    if device_name == "auto":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def load_project_config() -> dict:
    """注释
    功能: 从`config.yaml`读取预训练所需项目配置。
    输入: 无。
    输出: 配置字典，若文件为空则返回空字典。
    示例: `cfg = load_project_config()`。
    时间: 2026-04-26。
    """
    from Autobs.paths import CONFIG_PATH

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def split_train_val_indices(num_samples: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """注释
    功能: 将样本索引拆分为训练集与验证集索引。
    输入: `num_samples` 为样本数，`val_ratio` 为验证集比例，`seed` 为随机种子。
    输出: `(train_indices, val_indices)` 两个不重叠索引数组。
    示例: `train_ids, val_ids = split_train_val_indices(100, 0.1, 42)`。
    时间: 2026-04-27。
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if val_ratio <= 0.0 or num_samples == 1:
        return np.arange(num_samples, dtype=np.int64), np.empty((0,), dtype=np.int64)

    val_count = int(round(num_samples * val_ratio))
    val_count = max(1, val_count)
    val_count = min(num_samples - 1, val_count)

    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(indices)
    val_indices = np.sort(indices[:val_count])
    train_indices = np.sort(indices[val_count:])
    return train_indices, val_indices


def pick_best_epoch(best_epoch: dict[str, Any] | None, current_epoch: dict[str, Any]) -> dict[str, Any]:
    """注释
    功能: 按验证 `top10`、验证 `loss` 和训练 `loss` 的优先级选择当前最优 epoch。
    输入: 历史最优记录 `best_epoch` 与当前 epoch 记录 `current_epoch`。
    输出: 更新后的最优 epoch 字典。
    示例: `best_epoch = pick_best_epoch(best_epoch, row)`。
    时间: 2026-04-27。
    """
    if best_epoch is None:
        return dict(current_epoch)

    best_val_top10 = best_epoch.get("val_top10")
    current_val_top10 = current_epoch.get("val_top10")
    if best_val_top10 is not None and current_val_top10 is not None:
        if float(current_val_top10) > float(best_val_top10):
            return dict(current_epoch)
        if float(current_val_top10) < float(best_val_top10):
            return dict(best_epoch)

    best_val_loss = best_epoch.get("val_loss")
    current_val_loss = current_epoch.get("val_loss")
    if best_val_loss is not None and current_val_loss is not None:
        if float(current_val_loss) < float(best_val_loss):
            return dict(current_epoch)
        return dict(best_epoch)

    if float(current_epoch["train_loss"]) < float(best_epoch["train_loss"]):
        return dict(current_epoch)
    return dict(best_epoch)


def compute_topk_metrics(logits: torch.Tensor, best_actions: torch.Tensor, ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    """注释
    功能: 计算策略 logits 相对于最优动作标签的 top-k 命中率。
    输入: `logits` 为 `[batch, actions]` 张量，`best_actions` 为最优动作索引，`ks` 为需要统计的 k 值集合。
    输出: 形如 `{"top1": ..., "top5": ..., "top10": ...}` 的指标字典。
    示例: `metrics = compute_topk_metrics(logits, best_actions, ks=(1, 5, 10))`。
    时间: 2026-04-27。
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, actions]")
    if best_actions.ndim != 1:
        raise ValueError("best_actions must have shape [batch]")
    metrics: dict[str, float] = {}
    num_actions = logits.shape[1]
    for k in ks:
        topk = min(int(k), int(num_actions))
        top_indices = torch.topk(logits, k=topk, dim=-1).indices
        hits = (top_indices == best_actions.unsqueeze(-1)).any(dim=-1).float().mean().item()
        metrics[f"top{k}"] = float(hits)
    return metrics


def normalize_legal_action_scores(action_scores: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
    """注释
    功能: 仅在合法动作子集内对动作分数做逐样本标准化，便于 reward 型损失稳定训练。
    输入: `action_scores` 为逐动作分数张量，`action_masks` 为合法动作掩码张量。
    输出: 与输入同形状的合法动作标准化分数张量。
    示例: `normalized_scores = normalize_legal_action_scores(action_scores, action_masks)`。
    时间: 2026-04-27。
    """
    legal = action_masks > 0.0
    # 非法动作强制为0 从action_scores中提取合法的动作得分为masked_scores
    masked_scores = torch.where(legal, action_scores, torch.zeros_like(action_scores))
    legal_counts = legal.sum(dim=-1, keepdim=True).clamp(min=1)
    mean = masked_scores.sum(dim=-1, keepdim=True) / legal_counts
    # 合法位置的分数减去合法动作的平均分，非法位置直接置0
    centered = torch.where(legal, action_scores - mean, torch.zeros_like(action_scores))
    # 方差
    variance = (centered.pow(2).sum(dim=-1, keepdim=True) / legal_counts).clamp(min=1e-6)
    # 标准化
    normalized = centered / variance.sqrt()
    return torch.where(legal, normalized, torch.zeros_like(normalized))


def expected_reward_loss(logits: torch.Tensor, action_scores: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
    """注释
    功能: 最大化策略在合法动作集上的期望标准化 reward。
    输入: 当前策略 `logits`、监督动作分数 `action_scores` 与合法动作掩码 `action_masks`。
    输出: 取负后的期望 reward 损失标量。
    示例: `loss = expected_reward_loss(logits, action_scores, action_masks)`。
    时间: 2026-04-27。
    """
    # score标准化只在合法动作上进行，非法动作直接置0，score-mean除std，得到normalized_scores 都是监督数据
    normalized_scores = normalize_legal_action_scores(action_scores, action_masks)
    probs = torch.softmax(logits, dim=-1)
    expected_reward = (probs * normalized_scores).sum(dim=-1)
    return -expected_reward.mean()


def pairwise_ranking_loss(logits: torch.Tensor, action_scores: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
    """注释
    功能: 构造“最优动作应压过 hardest negative”的 pairwise 排序损失。
    输入: 当前策略 `logits`、监督动作分数 `action_scores` 与合法动作掩码 `action_masks`。
    输出: 标量 pairwise 排序损失。
    示例: `loss = pairwise_ranking_loss(logits, action_scores, action_masks)`。
    时间: 2026-04-27。
    """
    legal = action_masks > 0.0
    # 拿到最佳动作score的索引
    best_actions = torch.argmax(torch.where(legal, action_scores, torch.full_like(action_scores, -1e30)), dim=-1)
    # 模型学到的策略未归一化的原始分数
    best_logits = logits.gather(1, best_actions.unsqueeze(-1)).squeeze(-1)
    # 监督下已知最佳动作的分数
    best_scores = action_scores.gather(1, best_actions.unsqueeze(-1)).squeeze(-1)

    # 构造one-hot掩码排除最优动作
    best_one_hot = torch.nn.functional.one_hot(best_actions, num_classes=logits.shape[-1]).bool()
    competitor_logits = logits.masked_fill(best_one_hot, -1e30)
    # 在剩余动作中选取logit值最高的动作作为最难负样本
    hard_negative_actions = torch.argmax(competitor_logits, dim=-1)
    hard_negative_logits = logits.gather(1, hard_negative_actions.unsqueeze(-1)).squeeze(-1)
    hard_negative_scores = action_scores.gather(1, hard_negative_actions.unsqueeze(-1)).squeeze(-1)

    margin_weight = (best_scores - hard_negative_scores).clamp(min=0.0)
    # \mathcal{L}_{pair}^{\left(1\right)}=\left(1+\max{\left(0,z_\theta\left(a^+\mid m\right)-z_\theta\left(a^-\mid m\right)\right)}\right)softplus\left[-\left(z_\theta\left(a^+\mid m\right)-z_\theta\left(a^-\mid m\right)\right)\right]
    pairwise = torch.nn.functional.softplus(-(best_logits - hard_negative_logits))
    return (pairwise * (1.0 + margin_weight)).mean()


def apply_geometric_transform(
    observation: np.ndarray,
    action_mask: np.ndarray,
    target_probs: np.ndarray,
    best_action: int,
    transform_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """注释
    功能: 对单个样本同步执行几何变换，保证观测、动作掩码、目标分布与最优动作标签保持一致。
    输入: `observation`、`action_mask`、`target_probs`、`best_action` 和变换名 `transform_name`。
    输出: `(transformed_obs, transformed_mask, transformed_probs, transformed_best_action)`。
    示例: `obs_i, mask_i, probs_i, best_i = apply_geometric_transform(...)`。
    时间: 2026-04-27。
    """
    map_side = int(round(math.sqrt(observation.size)))
    action_side = int(round(math.sqrt(action_mask.size)))
    if map_side * map_side != observation.size:
        raise ValueError("observation cannot be reshaped into a square map")
    if action_side * action_side != action_mask.size:
        raise ValueError("action arrays cannot be reshaped into a square action grid")

    obs_grid = observation.reshape(map_side, map_side)
    mask_grid = action_mask.reshape(action_side, action_side)
    probs_grid = target_probs.reshape(action_side, action_side)
    best_row, best_col = divmod(int(best_action), action_side)

    if transform_name == "identity":
        transformed_obs = obs_grid
        transformed_mask = mask_grid
        transformed_probs = probs_grid
        new_row, new_col = best_row, best_col
    elif transform_name == "flip_lr":
        transformed_obs = np.fliplr(obs_grid)
        transformed_mask = np.fliplr(mask_grid)
        transformed_probs = np.fliplr(probs_grid)
        new_row, new_col = best_row, action_side - 1 - best_col
    elif transform_name == "flip_ud":
        transformed_obs = np.flipud(obs_grid)
        transformed_mask = np.flipud(mask_grid)
        transformed_probs = np.flipud(probs_grid)
        new_row, new_col = action_side - 1 - best_row, best_col
    elif transform_name == "rot90":
        transformed_obs = np.rot90(obs_grid, k=1)
        transformed_mask = np.rot90(mask_grid, k=1)
        transformed_probs = np.rot90(probs_grid, k=1)
        new_row, new_col = action_side - 1 - best_col, best_row
    elif transform_name == "rot180":
        transformed_obs = np.rot90(obs_grid, k=2)
        transformed_mask = np.rot90(mask_grid, k=2)
        transformed_probs = np.rot90(probs_grid, k=2)
        new_row, new_col = action_side - 1 - best_row, action_side - 1 - best_col
    elif transform_name == "rot270":
        transformed_obs = np.rot90(obs_grid, k=3)
        transformed_mask = np.rot90(mask_grid, k=3)
        transformed_probs = np.rot90(probs_grid, k=3)
        new_row, new_col = best_col, action_side - 1 - best_row
    else:
        raise ValueError(f"Unsupported transform_name: {transform_name}")

    return (
        transformed_obs.reshape(-1).astype(np.float32),
        transformed_mask.reshape(-1).astype(np.float32),
        transformed_probs.reshape(-1).astype(np.float32),
        int(new_row * action_side + new_col),
    )


def transform_action_scores(
    action_scores: np.ndarray,
    action_mask: np.ndarray,
    transform_name: str,
) -> np.ndarray:
    """注释
    功能: 将逐动作分数按照动作网格的几何变换规则同步重排。
    输入: `action_scores` 为逐动作分数，`action_mask` 用于确定动作网格边长，`transform_name` 为变换名。
    输出: 变换后的逐动作分数数组。
    示例: `scores_i = transform_action_scores(action_scores, action_mask, "rot90")`。
    时间: 2026-04-27。
    """
    action_side = int(round(math.sqrt(action_mask.size)))
    if action_side * action_side != action_mask.size:
        raise ValueError("action arrays cannot be reshaped into a square action grid")
    scores_grid = action_scores.reshape(action_side, action_side)
    if transform_name == "identity":
        transformed_scores = scores_grid
    elif transform_name == "flip_lr":
        transformed_scores = np.fliplr(scores_grid)
    elif transform_name == "flip_ud":
        transformed_scores = np.flipud(scores_grid)
    elif transform_name == "rot90":
        transformed_scores = np.rot90(scores_grid, k=1)
    elif transform_name == "rot180":
        transformed_scores = np.rot90(scores_grid, k=2)
    elif transform_name == "rot270":
        transformed_scores = np.rot90(scores_grid, k=3)
    else:
        raise ValueError(f"Unsupported transform_name: {transform_name}")
    return transformed_scores.reshape(-1).astype(np.float32)

def apply_random_geometric_augmentation(
    observations: np.ndarray,
    action_masks: np.ndarray,
    target_probs: np.ndarray,
    action_scores: np.ndarray,
    best_actions: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """注释
    功能: 对一个 batch 的监督样本随机做几何增强，并同步更新所有动作相关标签。
    输入: 批量 `observations`、`action_masks`、`target_probs`、`action_scores`、`best_actions` 和随机数生成器。
    输出: 增强后的五个批量数组，顺序与输入保持一致。
    示例: `aug_obs, aug_masks, aug_probs, aug_scores, aug_best = apply_random_geometric_augmentation(...)`。
    时间: 2026-04-27。
    """
    transforms = ("identity", "flip_lr", "flip_ud", "rot90", "rot180", "rot270")
    aug_obs = np.empty_like(observations, dtype=np.float32)
    aug_masks = np.empty_like(action_masks, dtype=np.float32)
    aug_probs = np.empty_like(target_probs, dtype=np.float32)
    aug_scores = np.empty_like(action_scores, dtype=np.float32)
    aug_best = np.empty_like(best_actions, dtype=np.int64)

    for idx in range(observations.shape[0]):
        transform_name = transforms[int(rng.integers(0, len(transforms)))]
        obs_i, mask_i, probs_i, best_i = apply_geometric_transform(
            observations[idx],
            action_masks[idx],
            target_probs[idx],
            int(best_actions[idx]),
            transform_name,
        )
        aug_obs[idx] = obs_i
        aug_masks[idx] = mask_i
        aug_probs[idx] = probs_i
        aug_scores[idx] = transform_action_scores(action_scores[idx], action_masks[idx], transform_name)
        aug_best[idx] = best_i
    return aug_obs, aug_masks, aug_probs, aug_scores, aug_best


def load_rank_dataset(dataset_path: str | Path, temperature: float) -> dict[str, np.ndarray]:
    """注释
    功能: 读取动作排序监督数据集，并派生 soft target 与最优动作标签。
    输入: `dataset_path` 为 `.npz` 数据集路径，`temperature` 为 soft target 温度。
    输出: 包含观测、动作掩码、动作分数、soft target 与最优动作索引的数组字典。
    示例: `dataset = load_rank_dataset("outputs/action_rank_dataset.npz", 0.5)`。
    时间: 2026-04-27。
    """
    path = Path(dataset_path).expanduser().resolve()
    payload = np.load(path, allow_pickle=False)
    observations = np.asarray(payload["observations"], dtype=np.float32)
    action_masks = np.asarray(payload["action_masks"], dtype=np.float32)
    action_scores = np.asarray(payload["action_scores"], dtype=np.float32)
    target_probs = scores_to_target_probs(action_scores, action_masks, temperature=temperature)
    best_actions = np.argmax(np.where(action_masks > 0.0, action_scores, -1e30), axis=1).astype(np.int64)
    return {
        "observations": observations,
        "action_masks": action_masks,
        "action_scores": action_scores,
        "target_probs": target_probs,
        "best_actions": best_actions,
    }


def build_policy_module(version: str, device: torch.device):
    """注释
    功能: 构建用于监督预训练的动作掩码策略模块，并按项目配置初始化其观测/动作空间。
    输入: `version` 为策略版本，`device` 为目标 torch 设备。
    输出: 已经 `.to(device)` 且处于 train 模式的策略模块实例。
    示例: `module = build_policy_module("single", device)`。
    时间: 2026-04-27。
    """
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    from Autobs.action_mask_rlm import ActionMaskPolicyModule, build_single_site_spaces

    config = load_project_config()
    env_config = dict(config.get("env", {}))
    model_config = dict(config.get("train", {}).get("model", {}))
    if version != "single":
        raise ValueError("Only single-site policy learning is supported in the current Autobs pipeline")
    observation_space, action_space = build_single_site_spaces(
        map_size=int(env_config.get("map_size", 256)),
        action_space_size=int(env_config.get("action_space_size", 32)),
    )
    spec = RLModuleSpec(
        module_class=ActionMaskPolicyModule,
        observation_space=observation_space,
        action_space=action_space,
        model_config=model_config,
    )
    module = spec.build()
    module.to(device)
    module.train()
    return module


def iter_minibatches(num_samples: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """注释
    功能: 打乱样本索引并按 batch 大小切分成小批次索引列表。
    输入: `num_samples` 为样本数，`batch_size` 为批大小，`rng` 为随机数生成器。
    输出: 每个元素都是一个 batch 索引数组的列表。
    示例: `batch_indices = iter_minibatches(128, 32, rng)`。
    时间: 2026-04-27。
    """
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, num_samples, batch_size)]


def forward_masked_logits(module, observations: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
    """注释
    功能: 将观测与动作掩码打包为 RLModule 期望的输入格式，并返回已完成 masking 的动作 logits。
    输入: 策略模块 `module`、批量观测 `observations`、批量动作掩码 `action_masks`。
    输出: `[batch, actions]` 形状的 masked logits 张量。
    示例: `logits = forward_masked_logits(module, observations, action_masks)`。
    时间: 2026-04-27。
    """
    from ray.rllib.policy.sample_batch import SampleBatch

    batch = {
        SampleBatch.OBS: {
            "observations": observations,
            "action_mask": action_masks,
        }
    }
    # _forward_train 内部会调用 module.forward()，并自动处理 action_mask 来屏蔽 logits
    outputs = module._forward_train(batch)
    return outputs[SampleBatch.ACTION_DIST_INPUTS]


def evaluate_split(
    module,
    observations: torch.Tensor,
    action_masks: torch.Tensor,
    target_probs: torch.Tensor,
    best_actions: torch.Tensor,
    indices: np.ndarray,
) -> dict[str, float]:
    """注释
    功能: 在验证集切片上评估监督预训练模型的 listwise 损失与 top-k 命中率。
    输入: 策略模块、全量张量与待评估索引数组 `indices`。
    输出: `loss`、`top1`、`top5`、`top10` 指标字典。
    示例: `metrics = evaluate_split(module, observations, action_masks, target_probs, best_actions, val_indices)`。
    时间: 2026-04-27。
    """
    if indices.size == 0:
        return {"loss": float("nan"), "top1": float("nan"), "top5": float("nan"), "top10": float("nan")}

    with torch.no_grad():
        obs_batch = observations[indices]
        mask_batch = action_masks[indices]
        target_batch = target_probs[indices]
        best_batch = best_actions[indices]
        logits = forward_masked_logits(module, obs_batch, mask_batch)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(target_batch * log_probs).sum(dim=-1).mean()
        topk_metrics = compute_topk_metrics(logits, best_batch, ks=(1, 5, 10))
    return {"loss": float(loss.item()), **topk_metrics}


def pretrain_policy(args: argparse.Namespace) -> dict[str, Any]:
    """注释
    功能: 执行动作排序监督预训练，保存最佳模块状态并输出训练摘要。
    输入: `args` 为命令行解析后的训练配置。
    输出: 包含最佳 epoch、验证指标、超参数和权重路径的摘要字典。
    示例: `summary = pretrain_policy(args)`。
    时间: 2026-04-27。
    """
    device = _resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    dataset = load_rank_dataset(args.dataset, temperature=args.target_temperature)
    module = build_policy_module(args.version, device=device)
    optimizer = torch.optim.Adam(module.parameters(), lr=args.lr, weight_decay=args.weight_decay)

#     | 名称            | 含义      |
# | ------------- | ------- |
# | observations  | 状态      |
# | action_masks  | 哪些动作合法  |
# | target_probs  | 理想概率分布  |
# | action_scores | 每个动作的评分 |
# | best_actions  | 最优动作标签  |

    observations_np = np.asarray(dataset["observations"], dtype=np.float32)
    action_masks_np = np.asarray(dataset["action_masks"], dtype=np.float32)
    target_probs_np = np.asarray(dataset["target_probs"], dtype=np.float32)
    action_scores_np = np.asarray(dataset["action_scores"], dtype=np.float32)
    best_actions_np = np.asarray(dataset["best_actions"], dtype=np.int64)

    observations = torch.from_numpy(observations_np).to(device)
    action_masks = torch.from_numpy(action_masks_np).to(device)
    target_probs = torch.from_numpy(target_probs_np).to(device)
    best_actions = torch.from_numpy(best_actions_np).to(device)
    train_indices, val_indices = split_train_val_indices(observations.shape[0], args.val_ratio, args.seed)

    history: list[dict[str, float]] = []
    best_epoch: dict[str, Any] | None = None
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        # 打乱train样本索引并按 batch 大小切分成小批次索引列表
        batch_indices = iter_minibatches(train_indices.size, args.batch_size, rng)
        epoch_loss = 0.0
        epoch_top1 = 0.0
        seen = 0

        # batch_indices=[[],[],...] 每个元素是一个batch的索引数组batch_ids
        for batch_ids in batch_indices:
            split_ids = train_indices[batch_ids]
            obs_batch_np = observations_np[split_ids]
            mask_batch_np = action_masks_np[split_ids]
            target_batch_np = target_probs_np[split_ids]
            score_batch_np = action_scores_np[split_ids]
            best_batch_np = best_actions_np[split_ids]

            if args.geom_augment:
                obs_batch_np, mask_batch_np, target_batch_np, score_batch_np, best_batch_np = apply_random_geometric_augmentation(
                    obs_batch_np,
                    mask_batch_np,
                    target_batch_np,
                    score_batch_np,
                    best_batch_np,
                    rng,
                )

            obs_batch = torch.from_numpy(obs_batch_np).to(device)
            mask_batch = torch.from_numpy(mask_batch_np).to(device)
            target_batch = torch.from_numpy(target_batch_np).to(device)
            score_batch = torch.from_numpy(score_batch_np).to(device)
            best_batch = torch.from_numpy(best_batch_np).to(device)

            optimizer.zero_grad(set_to_none=True)
            # 前向传播得到屏蔽后的 logits 模型认为每个动作有多好
            logits = forward_masked_logits(module, obs_batch, mask_batch)
            # logπθ​(a∣s)
            log_probs = torch.log_softmax(logits, dim=-1)
            # target_probs_np 监督提供的每个动作的理想概率分布，target_batch是它的tensor版本
            listwise_loss = -(target_batch * log_probs).sum(dim=-1).mean()
            pairwise_loss = pairwise_ranking_loss(logits, score_batch, mask_batch)
            # 让模型更倾向选择高分动作
            reward_loss = expected_reward_loss(logits, score_batch, mask_batch)
            loss = listwise_loss + args.pairwise_coeff * pairwise_loss + args.reward_coeff * reward_loss
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=-1)
            batch_size = len(batch_ids)
            epoch_loss += float(loss.item()) * batch_size
            epoch_top1 += float((preds == best_batch).float().sum().item())
            seen += batch_size

        train_loss = epoch_loss / max(seen, 1)
        train_top1 = epoch_top1 / max(seen, 1)
        val_metrics = evaluate_split(
            module,
            observations,
            action_masks,
            target_probs,
            best_actions,
            val_indices,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "val_loss": None if np.isnan(val_metrics["loss"]) else val_metrics["loss"],
            "val_top1": None if np.isnan(val_metrics["top1"]) else val_metrics["top1"],
            "val_top5": None if np.isnan(val_metrics["top5"]) else val_metrics["top5"],
            "val_top10": None if np.isnan(val_metrics["top10"]) else val_metrics["top10"],
        }
        history.append(row)

        if row["val_loss"] is None:
            print(f"epoch={epoch} train_loss={train_loss:.6f} train_top1={train_top1:.4f}")
        else:
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} train_top1={train_top1:.4f} "
                f"val_loss={val_metrics['loss']:.6f} val_top1={val_metrics['top1']:.4f} "
                f"val_top5={val_metrics['top5']:.4f} val_top10={val_metrics['top10']:.4f}"
            )

        updated_best = pick_best_epoch(best_epoch, row)
        if best_epoch is None or updated_best["epoch"] != best_epoch["epoch"]:
            best_epoch = updated_best
            best_state = {key: value.detach().cpu() for key, value in module.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if val_indices.size > 0 and args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"early_stop_epoch={epoch} patience={args.patience}")
            break

    if best_state is None:
        best_state = {key: value.detach().cpu() for key, value in module.state_dict().items()}
    if best_epoch is None:
        best_epoch = {"epoch": 0, "train_loss": math.inf, "train_top1": 0.0, "val_loss": None, "val_top1": None}

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "best_module_state.pt"
    summary_path = output_dir / "pretrain_summary.json"
    torch.save({"state_dict": best_state}, state_path)
    summary = {
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "val_ratio": float(args.val_ratio),
        "patience": int(args.patience),
        "geom_augment": bool(args.geom_augment),
        "target_temperature": float(args.target_temperature),
        "pairwise_coeff": float(args.pairwise_coeff),
        "reward_coeff": float(args.reward_coeff),
        "device": str(device),
        "num_samples": int(observations.shape[0]),
        "train_size": int(train_indices.size),
        "val_size": int(val_indices.size),
        "best_epoch": int(best_epoch["epoch"]),
        "best_train_loss": float(best_epoch["train_loss"]),
        "best_train_top1": float(best_epoch["train_top1"]),
        "best_val_loss": None if best_epoch["val_loss"] is None else float(best_epoch["val_loss"]),
        "best_val_top1": None if best_epoch["val_top1"] is None else float(best_epoch["val_top1"]),
        "best_val_top5": None if best_epoch.get("val_top5") is None else float(best_epoch["val_top5"]),
        "best_val_top10": None if best_epoch.get("val_top10") is None else float(best_epoch["val_top10"]),
        "history": history,
        "module_state_path": str(state_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    """注释
    功能: 解析命令行并执行监督预训练主流程。
    输入: `argv` 为可选命令行参数列表。
    输出: 无；运行结束后打印摘要 JSON。
    示例: `main()`。
    时间: 2026-04-27。
    """
    args = build_parser().parse_args(argv)
    summary = pretrain_policy(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
