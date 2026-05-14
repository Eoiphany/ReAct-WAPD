"""注释
命令:
python -m Autobs.train_constrained_rerank \
  --dataset /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/outputs/action_rank_dataset.npz \
  --version single \
  --init-module-state /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/outputs/pretrained_policy/best_module_state.pt \
  --epochs 200 \
  --batch-size 32 \
  --lr 5e-5 \
  --top-n 32 \
  --coverage-margin 0.0 \
  --coverage-penalty 8.0 \
  --infeasible-bias 0.5 \
  --kl-coeff 1.0 \
  --pairwise-coeff 0.25 \
  --reward-coeff 1.0 \
  --reward-temperature 0.3 \
  --device mps \
  --output-dir /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/outputs/rerank_policy

参数含义:
- `--dataset`: 由 `build_action_rank_dataset.py` 生成的 `.npz` 数据集，且其中必须包含 `action_coverages` 与 `action_spectral_efficiencies`。
- `--version`: 当前仅支持 `single`。
- `--init-module-state`: 第一阶段监督预训练得到的策略权重，用作二阶段初始化与 KL anchor。
- `--epochs / --batch-size / --lr / --weight-decay`: 二阶段微调的训练超参数。
- `--val-ratio / --patience`: 验证集比例与早停轮数。
- `--geom-augment`: 是否对观测与动作标签同步做几何增强。
- `--top-n`: 只在 reference policy 的 topN 合法候选上做 rerank 微调。
- `--coverage-margin`: coverage 门槛的松弛量，实际门槛为 `coverage_target - coverage_margin`。
- `--coverage-penalty`: coverage 不达标时的相对罚强度。
- `--infeasible-bias`: coverage 不达标时额外减去的常数罚项，避免模型把轻微违约但高 SE 的动作顶上去。
- `--kl-coeff`: 当前策略对 reference policy 的 KL 锚定强度。
- `--pairwise-coeff`: 候选内 pairwise 排序损失权重。
- `--reward-coeff`: 候选内期望 reward 损失权重。
- `--reward-temperature`: 将约束 reward 转成 listwise soft target 时的 softmax 温度。
- `--device`: 训练设备。
- `--output-dir`: 输出目录，会保存 `best_module_state.pt` 和 `rerank_summary.json`。

脚本逻辑说明:
本脚本处理的是“单步初始化”问题，而不是时序 DRL。它以第一阶段预训练策略为 reference，
先取 reference 的 topN 合法动作作为候选集，再在候选集内部用“coverage 达标后尽量抬高 SE”的
约束 reward 做微调。损失由四部分组成：候选集上的 listwise 软标签、pairwise 排序、期望 reward，
以及对 reference policy 的 KL 锚定。这样做的目标不是重新学习 legality，而是在保持 coverage
优势不被冲掉的前提下，把高 SE 的候选动作往前推。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Autobs.pretrain_policy import (
    _resolve_device,
    apply_geometric_transform,
    build_policy_module,
    compute_topk_metrics,
    forward_masked_logits,
    iter_minibatches,
    normalize_legal_action_scores,
    split_train_val_indices,
    transform_action_scores,
)
from Autobs.utils import apply_module_state, load_module_state

NEGATIVE_SENTINEL = -1e9


def build_parser() -> argparse.ArgumentParser:
    """注释
    功能: 构建单步约束式 rerank 微调脚本的命令行参数解析器。
    输入: 无。
    输出: `argparse.ArgumentParser`，包含数据集、初始化权重、约束 reward 与训练超参数。
    示例: `parser = build_parser()`。
    时间: 2026-04-27。
    """
    parser = argparse.ArgumentParser(description="Single-step constrained rerank finetuning on top of pretrained initialization policy")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--version", default="single", choices=["single"])
    parser.add_argument("--init-module-state", required=True, type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--patience", default=40, type=int)
    parser.add_argument("--geom-augment", action="store_true")
    parser.add_argument("--top-n", default=32, type=int)
    parser.add_argument("--coverage-margin", default=0.0, type=float)
    parser.add_argument("--coverage-penalty", default=8.0, type=float)
    parser.add_argument("--infeasible-bias", default=0.5, type=float)
    parser.add_argument("--kl-coeff", default=1.0, type=float)
    parser.add_argument("--pairwise-coeff", default=0.25, type=float)
    parser.add_argument("--reward-coeff", default=1.0, type=float)
    parser.add_argument("--reward-temperature", default=0.3, type=float)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    return parser


def _make_module_agent(module):
    """注释
    功能: 将裸 `RLModule` 包装成兼容 `apply_module_state()` 的最小 agent 接口。
    输入: `module` 为待加载权重的策略模块。
    输出: 仅暴露 `get_module()` 的轻量包装对象。
    示例: `apply_module_state(_make_module_agent(module), state_dict)`。
    时间: 2026-04-27。
    """
    class _ModuleAgent:
        def __init__(self, inner_module):
            self._inner_module = inner_module

        def get_module(self):
            return self._inner_module

    return _ModuleAgent(module)


def load_rerank_dataset(dataset_path: str | Path) -> dict[str, np.ndarray]:
    """注释
    功能: 读取单步 rerank 微调所需的数据集，并强校验新增加的 radio 指标字段。
    输入: `dataset_path` 为 `.npz` 数据集路径。
    输出: 包含观测、动作掩码、分数、逐动作 coverage、逐动作 SE 与目标门槛的数组字典。
    示例: `dataset = load_rerank_dataset("outputs/action_rank_dataset.npz")`。
    时间: 2026-04-27。
    """
    path = Path(dataset_path).expanduser().resolve()
    payload = np.load(path, allow_pickle=False)
    required_keys = (
        "observations",
        "action_masks",
        "action_scores",
        "action_coverages",
        "action_spectral_efficiencies",
        "coverage_targets",
        "spectral_efficiency_targets",
    )
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(
            f"Dataset {path} is missing required arrays: {missing}. Please rebuild it with Autobs.build_action_rank_dataset."
        )
    return {
        "observations": np.asarray(payload["observations"], dtype=np.float32),
        "action_masks": np.asarray(payload["action_masks"], dtype=np.float32),
        "action_scores": np.asarray(payload["action_scores"], dtype=np.float32),
        "action_coverages": np.asarray(payload["action_coverages"], dtype=np.float32),
        "action_spectral_efficiencies": np.asarray(payload["action_spectral_efficiencies"], dtype=np.float32),
        "coverage_targets": np.asarray(payload["coverage_targets"], dtype=np.float32),
        "spectral_efficiency_targets": np.asarray(payload["spectral_efficiency_targets"], dtype=np.float32),
    }


def apply_random_geometric_augmentation_rerank(
    observations: np.ndarray,
    action_masks: np.ndarray,
    action_scores: np.ndarray,
    action_coverages: np.ndarray,
    action_ses: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """注释
    功能: 对 rerank 二阶段样本做几何增强，并同步变换观测、动作掩码、动作分数、动作 coverage 与动作 SE。
    输入: 一批 `observations`、`action_masks`、`action_scores`、`action_coverages`、`action_ses` 以及随机数生成器。
    输出: 与输入同形状的增强后数组元组，保证监督标签与动作网格空间对齐。
    示例: `aug_obs, aug_masks, aug_scores, aug_cov, aug_se = apply_random_geometric_augmentation_rerank(...)`。
    时间: 2026-04-27。
    """
    transforms = ("identity", "flip_lr", "flip_ud", "rot90", "rot180", "rot270")
    aug_obs = np.empty_like(observations, dtype=np.float32)
    aug_masks = np.empty_like(action_masks, dtype=np.float32)
    aug_scores = np.empty_like(action_scores, dtype=np.float32)
    aug_coverages = np.empty_like(action_coverages, dtype=np.float32)
    aug_ses = np.empty_like(action_ses, dtype=np.float32)

    best_actions = np.argmax(np.where(action_masks > 0.0, action_scores, NEGATIVE_SENTINEL), axis=1).astype(np.int64)
    dummy_probs = np.where(action_masks > 0.0, 1.0, 0.0).astype(np.float32)

    for idx in range(observations.shape[0]):
        transform_name = transforms[int(rng.integers(0, len(transforms)))]
        obs_i, mask_i, _probs_i, _best_i = apply_geometric_transform(
            observations[idx],
            action_masks[idx],
            dummy_probs[idx],
            int(best_actions[idx]),
            transform_name,
        )
        aug_obs[idx] = obs_i
        aug_masks[idx] = mask_i
        aug_scores[idx] = transform_action_scores(action_scores[idx], action_masks[idx], transform_name)
        aug_coverages[idx] = transform_action_scores(action_coverages[idx], action_masks[idx], transform_name)
        aug_ses[idx] = transform_action_scores(action_ses[idx], action_masks[idx], transform_name)
    return aug_obs, aug_masks, aug_scores, aug_coverages, aug_ses


def build_reward_values(
    *,
    action_coverages: torch.Tensor,
    action_ses: torch.Tensor,
    action_masks: torch.Tensor,
    coverage_targets: torch.Tensor,
    coverage_margin: float,
    coverage_penalty: float,
    infeasible_bias: float,
) -> torch.Tensor:
    """注释
    功能: 构造“coverage 达标后尽量抬 SE”的逐动作约束 reward。
    输入: 每个动作的 `coverage`、`SE`、合法掩码、样本级 coverage 目标以及罚项超参数。
    输出: 与动作维度一致的 reward 张量；非法动作写为极小值，便于后续屏蔽。
    示例: `reward_values = build_reward_values(...)`。
    时间: 2026-04-27。
    """
    legal = action_masks > 0.0
    se_norm = normalize_legal_action_scores(action_ses, action_masks)
    coverage_floor = torch.clamp(coverage_targets.unsqueeze(-1) - float(coverage_margin), min=0.0, max=1.0)
    coverage_gap = torch.clamp(coverage_floor - action_coverages, min=0.0)
    coverage_gap = coverage_gap / torch.clamp(coverage_floor, min=1e-6)
    infeasible = (coverage_gap > 0.0).float()
    reward = se_norm - float(coverage_penalty) * coverage_gap - float(infeasible_bias) * infeasible
    return torch.where(legal, reward, torch.full_like(reward, NEGATIVE_SENTINEL))


def build_candidate_mask(reference_logits: torch.Tensor, action_masks: torch.Tensor, top_n: int) -> torch.Tensor:
    """注释
    功能: 基于第一阶段 reference policy 的 logits，截取 topN 合法候选动作作为二阶段 rerank 子空间。
    输入: `reference_logits`、合法动作掩码 `action_masks` 和候选数 `top_n`。
    输出: 布尔型候选掩码；若 topN 异常为空，则回退到全部合法动作。
    示例: `candidate_mask = build_candidate_mask(reference_logits, action_masks, top_n=32)`。
    时间: 2026-04-27。
    """
    legal = action_masks > 0.0
    masked_logits = reference_logits.masked_fill(~legal, -1e30)
    batch_size, action_dim = masked_logits.shape
    candidate_mask = torch.zeros_like(legal)
    candidate_k = max(1, min(int(top_n), int(action_dim)))
    top_indices = torch.topk(masked_logits, k=candidate_k, dim=-1).indices
    candidate_mask.scatter_(1, top_indices, True)
    candidate_mask &= legal

    no_candidate = ~candidate_mask.any(dim=-1, keepdim=True)
    candidate_mask = torch.where(no_candidate, legal, candidate_mask)
    return candidate_mask


def masked_soft_target(reward_values: torch.Tensor, candidate_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    """注释
    功能: 将候选集内的约束 reward 转为 listwise 软目标分布。
    输入: 逐动作 `reward_values`、候选掩码 `candidate_mask` 与 softmax 温度 `temperature`。
    输出: 仅在候选集上归一化的 soft target 概率分布。
    示例: `soft_targets = masked_soft_target(reward_values, candidate_mask, 0.3)`。
    时间: 2026-04-27。
    """
    if temperature <= 0.0:
        raise ValueError("reward_temperature must be positive")
    scaled = reward_values / float(temperature)
    scaled = scaled.masked_fill(~candidate_mask, -1e30)
    probs = torch.softmax(scaled, dim=-1)
    probs = torch.where(candidate_mask, probs, torch.zeros_like(probs))
    denom = probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return probs / denom


def pairwise_rerank_loss(logits: torch.Tensor, reward_values: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    """注释
    功能: 在候选集内部构造 hard negative 的 pairwise 排序损失，强化“更优约束 reward 的动作 logits 更高”。
    输入: 当前策略 `logits`、约束 reward 和候选掩码。
    输出: 标量 pairwise 损失。
    示例: `loss = pairwise_rerank_loss(logits, reward_values, candidate_mask)`。
    时间: 2026-04-27。
    """
    masked_rewards = reward_values.masked_fill(~candidate_mask, NEGATIVE_SENTINEL)
    best_actions = torch.argmax(masked_rewards, dim=-1)
    best_logits = logits.gather(1, best_actions.unsqueeze(-1)).squeeze(-1)
    best_rewards = masked_rewards.gather(1, best_actions.unsqueeze(-1)).squeeze(-1)

    best_one_hot = torch.nn.functional.one_hot(best_actions, num_classes=logits.shape[-1]).bool()
    competitor_logits = logits.masked_fill(best_one_hot | (~candidate_mask), -1e30)
    hard_negative_actions = torch.argmax(competitor_logits, dim=-1)
    hard_negative_logits = logits.gather(1, hard_negative_actions.unsqueeze(-1)).squeeze(-1)
    hard_negative_rewards = masked_rewards.gather(1, hard_negative_actions.unsqueeze(-1)).squeeze(-1)

    margin_weight = (best_rewards - hard_negative_rewards).clamp(min=0.0)
    pairwise = torch.nn.functional.softplus(-(best_logits - hard_negative_logits))
    return (pairwise * (1.0 + margin_weight)).mean()


def expected_reward_loss(logits: torch.Tensor, reward_values: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    """注释
    功能: 最大化候选集上的策略期望 reward，使概率质量向 coverage 达标且 SE 高的动作聚集。
    输入: 当前策略 `logits`、约束 reward 和候选掩码。
    输出: 取负后的期望 reward 损失，供最小化训练。
    示例: `loss = expected_reward_loss(logits, reward_values, candidate_mask)`。
    时间: 2026-04-27。
    """
    # 非候选动作设为极小值
    masked_logits = logits.masked_fill(~candidate_mask, -1e30)
    # 策略概率分布 只在候选动作上归一化
    probs = torch.softmax(masked_logits, dim=-1)
    probs = torch.where(candidate_mask, probs, torch.zeros_like(probs))
    denom = probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    probs = probs / denom
    expected_reward = (probs * reward_values.masked_fill(~candidate_mask, 0.0)).sum(dim=-1)
    return -expected_reward.mean()


def kl_anchor_loss(logits: torch.Tensor, reference_logits: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    """注释
    功能: 约束当前策略不要过快偏离第一阶段 reference policy，避免 coverage 优势被二阶段洗掉。
    输入: 当前策略 `logits`、reference policy 的 `reference_logits` 和候选掩码。
    输出: 候选集上的 KL 锚定损失。
    示例: `kl = kl_anchor_loss(logits, reference_logits, candidate_mask)`。
    时间: 2026-04-27。
    """
    current_log_probs = torch.log_softmax(logits.masked_fill(~candidate_mask, -1e30), dim=-1)
    reference_probs = torch.softmax(reference_logits.masked_fill(~candidate_mask, -1e30), dim=-1)
    reference_probs = torch.where(candidate_mask, reference_probs, torch.zeros_like(reference_probs))
    denom = reference_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    reference_probs = reference_probs / denom
    return torch.nn.functional.kl_div(current_log_probs, reference_probs, reduction="batchmean")


def constrained_best_actions(reward_values: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    """注释
    功能: 在候选集内部选出约束 reward 最大的动作，作为二阶段验证时的目标动作。
    输入: 逐动作 `reward_values` 与候选掩码 `candidate_mask`。
    输出: 每个样本的最优动作索引张量。
    示例: `best_actions = constrained_best_actions(reward_values, candidate_mask)`。
    时间: 2026-04-27。
    """
    return torch.argmax(reward_values.masked_fill(~candidate_mask, NEGATIVE_SENTINEL), dim=-1)


def evaluate_split(
    *,
    module,
    reference_module,
    observations: torch.Tensor,
    action_masks: torch.Tensor,
    action_coverages: torch.Tensor,
    action_ses: torch.Tensor,
    coverage_targets: torch.Tensor,
    indices: np.ndarray,
    top_n: int,
    coverage_margin: float,
    coverage_penalty: float,
    infeasible_bias: float,
    reward_temperature: float,
) -> dict[str, float]:
    """注释
    功能: 在验证集上评估二阶段 rerank 模型的 listwise 损失、平均 reward 与 top-k 命中率。
    输入: 当前策略、reference 策略、验证数据张量与约束 reward 超参数。
    输出: `loss`、`reward_mean`、`top1`、`top5`、`top10` 指标字典。
    示例: `metrics = evaluate_split(...)`。
    时间: 2026-04-27。
    """
    if indices.size == 0:
        return {"loss": float("nan"), "reward_mean": float("nan"), "top1": float("nan"), "top5": float("nan"), "top10": float("nan")}

    with torch.no_grad():
        obs_batch = observations[indices]
        mask_batch = action_masks[indices]
        cov_batch = action_coverages[indices]
        se_batch = action_ses[indices]
        cov_target_batch = coverage_targets[indices]

        logits = forward_masked_logits(module, obs_batch, mask_batch)
        reference_logits = forward_masked_logits(reference_module, obs_batch, mask_batch)
        candidate_mask = build_candidate_mask(reference_logits, mask_batch, top_n=top_n)
        reward_values = build_reward_values(
            action_coverages=cov_batch,
            action_ses=se_batch,
            action_masks=mask_batch,
            coverage_targets=cov_target_batch,
            coverage_margin=coverage_margin,
            coverage_penalty=coverage_penalty,
            infeasible_bias=infeasible_bias,
        )
        soft_targets = masked_soft_target(reward_values, candidate_mask, reward_temperature)
        masked_logits = logits.masked_fill(~candidate_mask, -1e30)
        listwise_loss = -(soft_targets * torch.log_softmax(masked_logits, dim=-1)).sum(dim=-1).mean()
        reward_mean = -(expected_reward_loss(masked_logits, reward_values, candidate_mask)).item()
        best_actions = constrained_best_actions(reward_values, candidate_mask)
        topk_metrics = compute_topk_metrics(masked_logits, best_actions, ks=(1, 5, 10))
    return {"loss": float(listwise_loss.item()), "reward_mean": float(reward_mean), **topk_metrics}


def pick_best_epoch(best_epoch: dict[str, Any] | None, current_epoch: dict[str, Any]) -> dict[str, Any]:
    """注释
    功能: 根据验证 reward、验证 top10 与训练损失选择当前最优 epoch。
    输入: 历史最优 epoch 记录 `best_epoch` 与当前 epoch 记录 `current_epoch`。
    输出: 更新后的最优 epoch 字典。
    示例: `best_epoch = pick_best_epoch(best_epoch, row)`。
    时间: 2026-04-27。
    """
    if best_epoch is None:
        return dict(current_epoch)
    best_reward = best_epoch.get("val_reward_mean")
    current_reward = current_epoch.get("val_reward_mean")
    if best_reward is not None and current_reward is not None:
        if float(current_reward) > float(best_reward):
            return dict(current_epoch)
        if float(current_reward) < float(best_reward):
            return dict(best_epoch)
    best_top10 = best_epoch.get("val_top10")
    current_top10 = current_epoch.get("val_top10")
    if best_top10 is not None and current_top10 is not None:
        if float(current_top10) > float(best_top10):
            return dict(current_epoch)
        if float(current_top10) < float(best_top10):
            return dict(best_epoch)
    if float(current_epoch["train_loss"]) < float(best_epoch["train_loss"]):
        return dict(current_epoch)
    return dict(best_epoch)


def train_constrained_rerank(args: argparse.Namespace) -> dict[str, Any]:
    """注释
    功能: 执行单步约束式 rerank 二阶段微调训练，并导出最佳模块权重与训练摘要。
    输入: `args` 为命令行解析后的训练配置。
    输出: 包含最佳 epoch、验证指标、超参数和权重路径的摘要字典。
    示例: `summary = train_constrained_rerank(args)`。
    时间: 2026-04-27。
    """
    device = _resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    dataset = load_rerank_dataset(args.dataset)
    observations_np = np.asarray(dataset["observations"], dtype=np.float32)
    action_masks_np = np.asarray(dataset["action_masks"], dtype=np.float32)
    action_scores_np = np.asarray(dataset["action_scores"], dtype=np.float32)
    action_coverages_np = np.asarray(dataset["action_coverages"], dtype=np.float32)
    action_ses_np = np.asarray(dataset["action_spectral_efficiencies"], dtype=np.float32)
    coverage_targets_np = np.asarray(dataset["coverage_targets"], dtype=np.float32)

    module = build_policy_module(args.version, device=device)
    reference_module = build_policy_module(args.version, device=device)
    module_state = load_module_state(args.init_module_state)
    apply_module_state(_make_module_agent(module), module_state)
    apply_module_state(_make_module_agent(reference_module), module_state)
    reference_module.eval()
    for parameter in reference_module.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(module.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    observations = torch.from_numpy(observations_np).to(device)
    action_masks = torch.from_numpy(action_masks_np).to(device)
    action_coverages = torch.from_numpy(action_coverages_np).to(device)
    action_ses = torch.from_numpy(action_ses_np).to(device)
    coverage_targets = torch.from_numpy(coverage_targets_np).to(device)

    train_indices, val_indices = split_train_val_indices(observations.shape[0], args.val_ratio, args.seed)

    history: list[dict[str, float | None]] = []
    best_epoch: dict[str, Any] | None = None
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        batch_indices = iter_minibatches(train_indices.size, args.batch_size, rng)
        epoch_loss = 0.0
        epoch_reward = 0.0
        seen = 0

        for batch_ids in batch_indices:
            split_ids = train_indices[batch_ids]
            obs_batch_np = observations_np[split_ids]
            mask_batch_np = action_masks_np[split_ids]
            score_batch_np = action_scores_np[split_ids]
            cov_batch_np = action_coverages_np[split_ids]
            se_batch_np = action_ses_np[split_ids]

            if args.geom_augment:
                obs_batch_np, mask_batch_np, score_batch_np, cov_batch_np, se_batch_np = apply_random_geometric_augmentation_rerank(
                    obs_batch_np,
                    mask_batch_np,
                    score_batch_np,
                    cov_batch_np,
                    se_batch_np,
                    rng,
                )

            obs_batch = torch.from_numpy(obs_batch_np).to(device)
            mask_batch = torch.from_numpy(mask_batch_np).to(device)
            cov_batch = torch.from_numpy(cov_batch_np).to(device)
            se_batch = torch.from_numpy(se_batch_np).to(device)
            cov_target_batch = coverage_targets[split_ids]

            optimizer.zero_grad(set_to_none=True)
            logits = forward_masked_logits(module, obs_batch, mask_batch)
            with torch.no_grad():
                reference_logits = forward_masked_logits(reference_module, obs_batch, mask_batch)
            candidate_mask = build_candidate_mask(reference_logits, mask_batch, top_n=args.top_n)
            reward_values = build_reward_values(
                action_coverages=cov_batch,
                action_ses=se_batch,
                action_masks=mask_batch,
                coverage_targets=cov_target_batch,
                coverage_margin=args.coverage_margin,
                coverage_penalty=args.coverage_penalty,
                infeasible_bias=args.infeasible_bias,
            )
            soft_targets = masked_soft_target(reward_values, candidate_mask, args.reward_temperature)
            masked_logits = logits.masked_fill(~candidate_mask, -1e30)
            listwise_loss = -(soft_targets * torch.log_softmax(masked_logits, dim=-1)).sum(dim=-1).mean()
            pairwise_loss = pairwise_rerank_loss(masked_logits, reward_values, candidate_mask)
            reward_loss = expected_reward_loss(masked_logits, reward_values, candidate_mask)
            kl_loss = kl_anchor_loss(masked_logits, reference_logits, candidate_mask)
            loss = listwise_loss + args.pairwise_coeff * pairwise_loss + args.reward_coeff * reward_loss + args.kl_coeff * kl_loss
            loss.backward()
            optimizer.step()

            batch_size = len(batch_ids)
            epoch_loss += float(loss.item()) * batch_size
            epoch_reward += float((-reward_loss).item()) * batch_size
            seen += batch_size

        train_loss = epoch_loss / max(seen, 1)
        train_reward_mean = epoch_reward / max(seen, 1)
        val_metrics = evaluate_split(
            module=module,
            reference_module=reference_module,
            observations=observations,
            action_masks=action_masks,
            action_coverages=action_coverages,
            action_ses=action_ses,
            coverage_targets=coverage_targets,
            indices=val_indices,
            top_n=args.top_n,
            coverage_margin=args.coverage_margin,
            coverage_penalty=args.coverage_penalty,
            infeasible_bias=args.infeasible_bias,
            reward_temperature=args.reward_temperature,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_reward_mean": train_reward_mean,
            "val_loss": None if np.isnan(val_metrics["loss"]) else val_metrics["loss"],
            "val_reward_mean": None if np.isnan(val_metrics["reward_mean"]) else val_metrics["reward_mean"],
            "val_top1": None if np.isnan(val_metrics["top1"]) else val_metrics["top1"],
            "val_top5": None if np.isnan(val_metrics["top5"]) else val_metrics["top5"],
            "val_top10": None if np.isnan(val_metrics["top10"]) else val_metrics["top10"],
        }
        history.append(row)

        if row["val_loss"] is None:
            print(f"epoch={epoch} train_loss={train_loss:.6f} train_reward_mean={train_reward_mean:.4f}")
        else:
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} train_reward_mean={train_reward_mean:.4f} "
                f"val_loss={val_metrics['loss']:.6f} val_reward_mean={val_metrics['reward_mean']:.4f} "
                f"val_top1={val_metrics['top1']:.4f} val_top5={val_metrics['top5']:.4f} val_top10={val_metrics['top10']:.4f}"
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
        best_epoch = {"epoch": 0, "train_loss": math.inf, "train_reward_mean": -math.inf, "val_reward_mean": None}

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "best_module_state.pt"
    summary_path = output_dir / "rerank_summary.json"
    torch.save({"state_dict": best_state}, state_path)
    summary = {
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "init_module_state": str(Path(args.init_module_state).expanduser().resolve()),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "val_ratio": float(args.val_ratio),
        "patience": int(args.patience),
        "geom_augment": bool(args.geom_augment),
        "top_n": int(args.top_n),
        "coverage_margin": float(args.coverage_margin),
        "coverage_penalty": float(args.coverage_penalty),
        "infeasible_bias": float(args.infeasible_bias),
        "kl_coeff": float(args.kl_coeff),
        "pairwise_coeff": float(args.pairwise_coeff),
        "reward_coeff": float(args.reward_coeff),
        "reward_temperature": float(args.reward_temperature),
        "device": str(device),
        "num_samples": int(observations.shape[0]),
        "train_size": int(train_indices.size),
        "val_size": int(val_indices.size),
        "best_epoch": int(best_epoch["epoch"]),
        "best_train_loss": float(best_epoch["train_loss"]),
        "best_train_reward_mean": float(best_epoch["train_reward_mean"]),
        "best_val_loss": None if best_epoch.get("val_loss") is None else float(best_epoch["val_loss"]),
        "best_val_reward_mean": None if best_epoch.get("val_reward_mean") is None else float(best_epoch["val_reward_mean"]),
        "best_val_top1": None if best_epoch.get("val_top1") is None else float(best_epoch["val_top1"]),
        "best_val_top5": None if best_epoch.get("val_top5") is None else float(best_epoch["val_top5"]),
        "best_val_top10": None if best_epoch.get("val_top10") is None else float(best_epoch["val_top10"]),
        "history": history,
        "module_state_path": str(state_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    """注释
    功能: 解析命令行并运行单步约束式 rerank 微调主流程。
    输入: `argv` 为可选命令行参数列表。
    输出: 无；训练完成后打印摘要 JSON。
    示例: `main()`。
    时间: 2026-04-27。
    """
    args = build_parser().parse_args(argv)
    summary = train_constrained_rerank(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
