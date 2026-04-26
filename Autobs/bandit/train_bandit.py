"""注释
命令:
python -m Autobs.bandit.train_bandit \
  --dataset /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/outputs/action_rank.npz \
  --version single \
  --init-module-state /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/pretrained_policy/best_module_state.pt \
  --epochs 2000 \
  --batch-size 32 \
  --lr 5e-5 \
  --kl-coeff 0.1 \
  --entropy-coeff 0.01 \
  --val-ratio 0.1 \
  --output-dir /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/bandit_policy

参数含义:
- `--dataset`: 第一阶段动作排序监督数据集 `.npz`。
- `--version`: 与第一阶段一致的策略版本，通常为 `single`。
- `--init-module-state`: 第一阶段预训练得到的策略模块权重。
- `--epochs`: 第二阶段 bandit 微调轮数。
- `--batch-size`: 每个 batch 的地图样本数。
- `--lr`: 第二阶段学习率。
- `--kl-coeff`: 保持贴近第一阶段软标签分布的 KL/交叉熵约束权重。
- `--entropy-coeff`: 熵奖励系数，避免策略过早塌缩。
- `--val-ratio`: 验证集比例。
- `--patience`: 基于 `val_reward_mean` 的早停 patience。
- `--output-dir`: 输出目录，保存 `best_module_state.pt` 与摘要 JSON。

脚本逻辑说明:
本脚本在第一阶段 CNN 预训练策略基础上进行第二阶段 contextual bandit 微调。
每个 batch 中，策略从当前动作分布里采样一个合法动作，直接使用离线 `action_scores`
中的即时 reward 作为反馈，优化目标为：
- policy gradient bandit loss；
- 对第一阶段软标签分布的 KL/交叉熵约束；
- 熵正则。
验证集用贪心 `argmax` 动作评估 `val_reward_mean`，并以其选择最佳权重。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Autobs.pretrain.pretrain_policy import (
    _resolve_device,
    build_policy_module,
    compute_topk_metrics,
    forward_masked_logits,
    load_rank_dataset,
    split_train_val_indices,
)
from Autobs.train_ppo import apply_module_state, load_module_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Second-stage contextual bandit finetuning for initialization policy")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--version", default="single", choices=["single", "multi"])
    parser.add_argument("--init-module-state", required=True, type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--kl-coeff", default=0.1, type=float)
    parser.add_argument("--entropy-coeff", default=0.01, type=float)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--patience", default=1000, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    return parser


def pick_best_epoch(best_epoch: dict[str, Any] | None, current_epoch: dict[str, Any]) -> dict[str, Any]:
    if best_epoch is None:
        return dict(current_epoch)
    if float(current_epoch["val_reward_mean"]) > float(best_epoch["val_reward_mean"]):
        return dict(current_epoch)
    return dict(best_epoch)


def _make_agent_like(module):
    class _AgentLike:
        def __init__(self, inner):
            self._inner = inner

        def get_module(self):
            return self._inner

    return _AgentLike(module)


def iter_minibatches(num_samples: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, num_samples, batch_size)]


def _masked_mean(scores: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(masks.sum(dim=-1), min=1.0)
    return (scores * masks).sum(dim=-1) / denom


def evaluate_val_split(
    module,
    observations: torch.Tensor,
    action_masks: torch.Tensor,
    action_scores: torch.Tensor,
    target_probs: torch.Tensor,
    best_actions: torch.Tensor,
    indices: np.ndarray,
) -> dict[str, float]:
    if indices.size == 0:
        return {
            "val_reward_mean": float("nan"),
            "val_top1": float("nan"),
            "val_top5": float("nan"),
            "val_top10": float("nan"),
        }

    with torch.no_grad():
        obs_batch = observations[indices]
        mask_batch = action_masks[indices]
        scores_batch = action_scores[indices]
        best_batch = best_actions[indices]
        logits = forward_masked_logits(module, obs_batch, mask_batch)
        greedy_actions = torch.argmax(logits, dim=-1)
        rewards = scores_batch.gather(1, greedy_actions.unsqueeze(-1)).squeeze(-1)
        topk = compute_topk_metrics(logits, best_batch, ks=(1, 5, 10))
    return {
        "val_reward_mean": float(rewards.mean().item()),
        "val_top1": float(topk["top1"]),
        "val_top5": float(topk["top5"]),
        "val_top10": float(topk["top10"]),
    }


def train_bandit(args: argparse.Namespace) -> dict[str, Any]:
    device = _resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    dataset = load_rank_dataset(args.dataset, temperature=0.5)
    module = build_policy_module(args.version, device=device)
    init_state = load_module_state(args.init_module_state)
    apply_module_state(_make_agent_like(module), init_state)
    optimizer = torch.optim.Adam(module.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    observations = torch.from_numpy(np.asarray(dataset["observations"], dtype=np.float32)).to(device)
    action_masks = torch.from_numpy(np.asarray(dataset["action_masks"], dtype=np.float32)).to(device)
    action_scores = torch.from_numpy(np.asarray(dataset["action_scores"], dtype=np.float32)).to(device)
    target_probs = torch.from_numpy(np.asarray(dataset["target_probs"], dtype=np.float32)).to(device)
    best_actions = torch.from_numpy(np.asarray(dataset["best_actions"], dtype=np.int64)).to(device)
    train_indices, val_indices = split_train_val_indices(observations.shape[0], args.val_ratio, args.seed)

    history: list[dict[str, float]] = []
    best_epoch: dict[str, Any] | None = None
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        module.train()
        epoch_loss = 0.0
        epoch_reward = 0.0
        seen = 0
        for batch_ids in iter_minibatches(train_indices.size, args.batch_size, rng):
            split_ids = train_indices[batch_ids]
            obs_batch = observations[split_ids]
            mask_batch = action_masks[split_ids]
            score_batch = action_scores[split_ids]
            target_batch = target_probs[split_ids]

            optimizer.zero_grad(set_to_none=True)
            logits = forward_masked_logits(module, obs_batch, mask_batch)
            dist = torch.distributions.Categorical(logits=logits)
            sampled_actions = dist.sample()
            sampled_rewards = score_batch.gather(1, sampled_actions.unsqueeze(-1)).squeeze(-1)
            baseline = _masked_mean(score_batch, mask_batch)
            advantage = sampled_rewards - baseline
            policy_loss = -(advantage.detach() * dist.log_prob(sampled_actions)).mean()

            log_probs = torch.log_softmax(logits, dim=-1)
            kl_like_loss = -(target_batch * log_probs).sum(dim=-1).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + args.kl_coeff * kl_like_loss - args.entropy_coeff * entropy
            loss.backward()
            optimizer.step()

            batch_size = len(batch_ids)
            epoch_loss += float(loss.item()) * batch_size
            epoch_reward += float(sampled_rewards.mean().item()) * batch_size
            seen += batch_size

        train_loss = epoch_loss / max(seen, 1)
        train_reward_mean = epoch_reward / max(seen, 1)

        module.eval()
        val_metrics = evaluate_val_split(
            module,
            observations,
            action_masks,
            action_scores,
            target_probs,
            best_actions,
            val_indices,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_reward_mean": train_reward_mean,
            **val_metrics,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} train_reward_mean={train_reward_mean:.4f} "
            f"val_reward_mean={val_metrics['val_reward_mean']:.4f} "
            f"val_top1={val_metrics['val_top1']:.4f} "
            f"val_top5={val_metrics['val_top5']:.4f} "
            f"val_top10={val_metrics['val_top10']:.4f}"
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
        best_epoch = {
            "epoch": 0,
            "train_loss": float("inf"),
            "train_reward_mean": float("nan"),
            "val_reward_mean": float("nan"),
            "val_top1": float("nan"),
            "val_top5": float("nan"),
            "val_top10": float("nan"),
        }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "best_module_state.pt"
    summary_path = output_dir / "bandit_summary.json"
    torch.save({"state_dict": best_state}, state_path)
    summary = {
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "init_module_state": str(Path(args.init_module_state).expanduser().resolve()),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "kl_coeff": float(args.kl_coeff),
        "entropy_coeff": float(args.entropy_coeff),
        "val_ratio": float(args.val_ratio),
        "patience": int(args.patience),
        "device": str(device),
        "num_samples": int(observations.shape[0]),
        "train_size": int(train_indices.size),
        "val_size": int(val_indices.size),
        "best_epoch": int(best_epoch["epoch"]),
        "best_train_loss": float(best_epoch["train_loss"]),
        "best_train_reward_mean": float(best_epoch["train_reward_mean"]),
        "best_val_reward_mean": float(best_epoch["val_reward_mean"]),
        "best_val_top1": float(best_epoch["val_top1"]),
        "best_val_top5": float(best_epoch["val_top5"]),
        "best_val_top10": float(best_epoch["val_top10"]),
        "history": history,
        "module_state_path": str(state_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = train_bandit(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
