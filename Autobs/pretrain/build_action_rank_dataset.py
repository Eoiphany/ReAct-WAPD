"""注释
命令:
python -m Autobs.pretrain.build_action_rank_dataset \
   --dataset-path /Users/epiphanyer/Desktop/coding/paper_experiment/dataset/png/buildingsWHeight \
   --dataset-limit 512 \
   --model-path /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/checkpoints/rmnet_radiomap3dseer.pt \
   --network-type rmnet \
   --heuristic-targets-path /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/outputs/heuristic_targets.json \
   --output-path /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/outputs/action_rank_dataset.npz

参数含义:
- `--dataset-path`: 输入地图目录、单图路径或逗号分隔路径列表。
- `--dataset-limit / --dataset-offset / --dataset-stride`: 数据子集控制，决定生成多少张图的监督样本。
- `--model-path`: surrogate 权重路径，用于对每个合法动作打分。
- `--network-type`: surrogate 模型类型，需与 `--model-path` 匹配。
- `--heuristic-targets-path`: 可选的场景目标文件；若提供，则按场景 target 计算 `score/score_v2`。
- `--reward-key`: 用哪一个指标作为动作排序分数，默认 `score_v2`。
- `--output-path`: 输出记录所有轨迹的 `.npz` 路径；会同时写一个同名 `.json` 元数据文件，仅记录[1/512] 0.png legal=508 best_action=690 score_v2=1.0989。

脚本逻辑说明:
本脚本对每张图枚举当前动作空间下的全部合法 action，使用与训练一致的 surrogate 和 reward
计算每个动作的单站点评分，生成动作排序监督数据集。输出中保存：
- `observations`: 每张图对应的单步观测向量；
- `action_masks`: 当前动作空间下的合法动作掩码；
- `action_scores`: 每个 action 的监督分数，非法动作写入极小值；
- 元数据 JSON: 地图路径、最佳动作、目标值等摘要。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from Autobs.compare_initialization_methods import LocalSurrogatePredictor
from Autobs.env.utils import (
    DEFAULT_COVERAGE_TARGET,
    DEFAULT_SPECTRAL_EFFICIENCY_TARGET,
    calc_action_mask,
    calc_upsampling_loc,
    get_stats,
    load_heuristic_targets,
    load_map_normalized,
    lookup_heuristic_targets,
    resolve_city_map_paths,
)
from Autobs.paths import DEFAULT_DATASET_MAP_DIR, PACKAGE_ROOT


DEFAULT_OUTPUT_PATH = PACKAGE_ROOT / "outputs" / "action_rank_dataset.npz"
NEGATIVE_SENTINEL = -1e9
REWARD_KEY_CHOICES = ("score_v2", "score", "coverage", "spectral_efficiency", "channel_capacity")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build per-map legal-action ranking dataset for policy pretraining")
    parser.add_argument("--dataset-path", "--city-map-path", dest="city_map_path", default=str(DEFAULT_DATASET_MAP_DIR))
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--dataset-offset", type=int, default=0)
    parser.add_argument("--dataset-stride", type=int, default=1)
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--network-type", default="pmnet", choices=["pmnet", "pmnet_v3", "rmnet", "rmnet_v3"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--heuristic-targets-path", default=None)
    parser.add_argument("--reward-key", default="score_v2", choices=REWARD_KEY_CHOICES)
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), type=str)
    return parser


def collect_map_paths(args: argparse.Namespace) -> list[Path]:
    paths = resolve_city_map_paths(
        args.city_map_path,
        DEFAULT_DATASET_MAP_DIR,
        dataset_limit=args.dataset_limit,
        dataset_offset=args.dataset_offset,
        dataset_stride=args.dataset_stride,
    )
    return [Path(path).expanduser().resolve() for path in paths]


def score_legal_actions(
    pixel_map: np.ndarray,
    predictor: LocalSurrogatePredictor,
    coverage_target: float,
    spectral_efficiency_target: float,
    reward_key: str,
) -> tuple[np.ndarray, int, dict[str, float]]:
    action_mask = calc_action_mask(pixel_map).astype(np.float32)
    legal_actions = np.flatnonzero(action_mask > 0.0)
    scores = np.full(action_mask.shape, NEGATIVE_SENTINEL, dtype=np.float32)
    best_metrics: dict[str, float] | None = None
    best_action = -1

    for action in legal_actions.astype(int).tolist():
        tx_loc = calc_upsampling_loc(action, pixel_map)
        _pathgain_db, metrics = get_stats(
            pixel_map,
            [tx_loc],
            pmnet=predictor,
            coverage_target=coverage_target,
            spectral_efficiency_target=spectral_efficiency_target,
        )
        value = float(metrics[reward_key])
        scores[action] = value
        if best_metrics is None or value > float(best_metrics[reward_key]):
            best_metrics = metrics
            best_action = int(action)

    if best_metrics is None:
        raise ValueError("No legal actions found for this map")

    return scores, best_action, {key: float(value) for key, value in best_metrics.items()}


def build_dataset(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    map_paths = collect_map_paths(args)
    predictor = LocalSurrogatePredictor(args.model_path, args.network_type, args.device)
    heuristic_targets = load_heuristic_targets(args.heuristic_targets_path)

    observations: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    action_scores: list[np.ndarray] = []
    coverage_targets: list[float] = []
    se_targets: list[float] = []
    metadata_samples: list[dict[str, Any]] = []

    for map_index, map_path in enumerate(map_paths, start=1):
        pixel_map = load_map_normalized(map_path)
        observation = np.clip(pixel_map.reshape(-1), 0.0, 1.0).astype(np.float32)
        action_mask = calc_action_mask(pixel_map).astype(np.float32)
        coverage_target, spectral_efficiency_target = lookup_heuristic_targets(
            heuristic_targets,
            map_path,
            DEFAULT_COVERAGE_TARGET,
            DEFAULT_SPECTRAL_EFFICIENCY_TARGET,
        )
        scores, best_action, best_metrics = score_legal_actions(
            pixel_map=pixel_map,
            predictor=predictor,
            coverage_target=coverage_target,
            spectral_efficiency_target=spectral_efficiency_target,
            reward_key=args.reward_key,
        )
        best_tx_row, best_tx_col = calc_upsampling_loc(best_action, pixel_map)

        observations.append(observation)
        action_masks.append(action_mask)
        action_scores.append(scores)
        coverage_targets.append(float(coverage_target))
        se_targets.append(float(spectral_efficiency_target))
        metadata_samples.append(
            {
                "image": str(map_path),
                "best_action": int(best_action),
                "best_tx_row": int(best_tx_row),
                "best_tx_col": int(best_tx_col),
                "coverage_target": float(coverage_target),
                "spectral_efficiency_target": float(spectral_efficiency_target),
                "best_metrics": best_metrics,
                "legal_action_count": int(np.count_nonzero(action_mask > 0.0)),
            }
        )
        print(
            f"[{map_index}/{len(map_paths)}] {map_path.name} "
            f"legal={int(np.count_nonzero(action_mask > 0.0))} "
            f"best_action={best_action} "
            f"{args.reward_key}={best_metrics[args.reward_key]:.4f}"
        )

    arrays = {
        "observations": np.stack(observations, axis=0).astype(np.float16),
        "action_masks": np.stack(action_masks, axis=0).astype(np.float16),
        "action_scores": np.stack(action_scores, axis=0).astype(np.float32),
        "coverage_targets": np.asarray(coverage_targets, dtype=np.float32),
        "spectral_efficiency_targets": np.asarray(se_targets, dtype=np.float32),
    }
    metadata = {
        "reward_key": args.reward_key,
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "network_type": args.network_type,
        "device": args.device,
        "samples": metadata_samples,
    }
    return arrays, metadata


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    arrays, metadata = build_dataset(args)

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".json")
    np.savez_compressed(output_path, **arrays)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"dataset": str(output_path), "metadata": str(metadata_path), "num_samples": len(metadata["samples"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
