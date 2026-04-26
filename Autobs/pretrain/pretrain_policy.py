"""注释
命令:
python -m Autobs.pretrain.pretrain_policy \
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
本脚本直接复用 PPO 的同一套策略模块结构，对离线动作排序数据集做监督预训练。
监督目标不是 one-hot 最优动作，而是由动作分数 softmax 得到的软分布，从而保留
“同图合法动作的相对好坏”。训练完成后输出可直接通过
`train_ppo.py --module-state-path ...` 载入的模块权重。若启用验证集，则最佳权重按
`val_loss` 选择，而不是按训练集 `loss` 选择；同时会额外汇报 `val_top5 / val_top10`，
便于观察排序质量是否超过单纯的 `top1` 命中率。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supervised pretraining for PPO policy from legal-action ranking dataset")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--version", default="single", choices=["single", "multi"])
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--patience", default=1000, type=int)
    parser.add_argument("--geom-augment", action="store_true")
    parser.add_argument("--target-temperature", default=0.2, type=float)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    return parser


def scores_to_target_probs(action_scores: np.ndarray, action_masks: np.ndarray, temperature: float) -> np.ndarray:
    if action_scores.shape != action_masks.shape:
        raise ValueError("action_scores and action_masks must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    valid = action_masks > 0.0
    scaled = np.where(valid, action_scores / temperature, -np.inf)
    finite_max = np.max(np.where(np.isfinite(scaled), scaled, -1e30), axis=1, keepdims=True)
    exp = np.where(valid, np.exp(scaled - finite_max), 0.0)
    denom = np.sum(exp, axis=1, keepdims=True)
    fallback = np.where(valid, 1.0, 0.0)
    fallback /= np.maximum(np.sum(fallback, axis=1, keepdims=True), 1.0)
    return np.where(denom > 0.0, exp / denom, fallback).astype(np.float32)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def split_train_val_indices(num_samples: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
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
    if best_epoch is None:
        return dict(current_epoch)

    best_key = "val_loss" if best_epoch.get("val_loss") is not None else "train_loss"
    current_key = "val_loss" if current_epoch.get("val_loss") is not None else "train_loss"
    if current_key != best_key:
        if current_key == "val_loss":
            return dict(current_epoch)
        return dict(best_epoch)
    if float(current_epoch[current_key]) < float(best_epoch[best_key]):
        return dict(current_epoch)
    return dict(best_epoch)


def compute_topk_metrics(logits: torch.Tensor, best_actions: torch.Tensor, ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
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


def apply_geometric_transform(
    observation: np.ndarray,
    action_mask: np.ndarray,
    target_probs: np.ndarray,
    best_action: int,
    transform_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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

# 把观测样本做变换
def apply_random_geometric_augmentation(
    observations: np.ndarray,
    action_masks: np.ndarray,
    target_probs: np.ndarray,
    best_actions: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transforms = ("identity", "flip_lr", "flip_ud", "rot90", "rot180", "rot270")
    aug_obs = np.empty_like(observations, dtype=np.float32)
    aug_masks = np.empty_like(action_masks, dtype=np.float32)
    aug_probs = np.empty_like(target_probs, dtype=np.float32)
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
        aug_best[idx] = best_i
    return aug_obs, aug_masks, aug_probs, aug_best


def load_rank_dataset(dataset_path: str | Path, temperature: float) -> dict[str, np.ndarray]:
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
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    from Autobs.ppo_config import load_project_config
    from Autobs.rl_module.action_mask_rlm import PPOActionMaskRLM
    from Autobs.train_ppo import load_environment_class

    config = load_project_config()
    env_config = dict(config.get("env", {}))
    model_config = dict(config.get("train", {}).get("model", {}))
    environment_class = load_environment_class(version)
    env = environment_class(env_config)
    spec = RLModuleSpec(
        module_class=PPOActionMaskRLM,
        observation_space=env.observation_space,
        action_space=env.action_space,
        model_config=model_config,
    )
    module = spec.build()
    module.to(device)
    module.train()
    return module


def iter_minibatches(num_samples: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, num_samples, batch_size)]


def forward_masked_logits(module, observations: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
    from ray.rllib.policy.sample_batch import SampleBatch

    batch = {
        SampleBatch.OBS: {
            "observations": observations,
            "action_mask": action_masks,
        }
    }
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
    device = _resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    dataset = load_rank_dataset(args.dataset, temperature=args.target_temperature)
    module = build_policy_module(args.version, device=device)
    optimizer = torch.optim.Adam(module.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    observations_np = np.asarray(dataset["observations"], dtype=np.float32)
    action_masks_np = np.asarray(dataset["action_masks"], dtype=np.float32)
    target_probs_np = np.asarray(dataset["target_probs"], dtype=np.float32)
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
        batch_indices = iter_minibatches(train_indices.size, args.batch_size, rng)
        epoch_loss = 0.0
        epoch_top1 = 0.0
        seen = 0

        for batch_ids in batch_indices:
            split_ids = train_indices[batch_ids]
            obs_batch_np = observations_np[split_ids]
            mask_batch_np = action_masks_np[split_ids]
            target_batch_np = target_probs_np[split_ids]
            best_batch_np = best_actions_np[split_ids]

            if args.geom_augment:
                obs_batch_np, mask_batch_np, target_batch_np, best_batch_np = apply_random_geometric_augmentation(
                    obs_batch_np,
                    mask_batch_np,
                    target_batch_np,
                    best_batch_np,
                    rng,
                )

            obs_batch = torch.from_numpy(obs_batch_np).to(device)
            mask_batch = torch.from_numpy(mask_batch_np).to(device)
            target_batch = torch.from_numpy(target_batch_np).to(device)
            best_batch = torch.from_numpy(best_batch_np).to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = forward_masked_logits(module, obs_batch, mask_batch)
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = -(target_batch * log_probs).sum(dim=-1).mean()
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
    args = build_parser().parse_args(argv)
    summary = pretrain_policy(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
