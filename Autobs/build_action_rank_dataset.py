"""注释
命令:

python -m Autobs.build_action_rank_dataset \
  --dataset-path /Users/epiphanyer/Desktop/coding/paper_experiment/dataset/png/buildingsWHeight \
  --dataset-limit 512 \
  --model-path /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/checkpoints/rmnet_radiomap3dseer.pt \
  --network-type rmnet \
  --heuristic-targets-path /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/outputs/heuristic_targets.json \
  --output-path /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/outputs/action_rank.npz

参数含义:
- `--dataset-path`: 输入地图目录、单图路径或逗号分隔路径列表。
- `--dataset-limit / --dataset-offset / --dataset-stride`: 数据子集控制，决定生成多少张图的监督样本。
- `--model-path`: surrogate 权重路径，用于对每个合法动作打分。
- `--network-type`: surrogate 模型类型，需与 `--model-path` 匹配。
- `--heuristic-targets-path`: 可选的场景目标文件；若提供，则按场景 target 计算统一后的 `score`。
- `--reward-key`: 用哪一个指标作为动作排序分数；当前仅保留统一后的 `score`。
- `--output-path`: 输出记录所有轨迹的 `.npz` 路径；会同时写一个同名 `.json` 元数据文件，仅记录如
  `[1/512] 0.png legal=508 best_action=690 score=1.0989` 这类摘要。

脚本逻辑说明:
本脚本对每张图枚举当前动作空间下的全部合法 action，使用与训练一致的 surrogate 和 reward
计算每个动作的单站点评分，生成动作排序监督数据集。输出中保存：
- `observations`: 每张图对应的单步观测向量；
- `action_masks`: 当前动作空间下的合法动作掩码；
- `action_scores`: 每个 action 的监督分数，非法动作写入极小值；
- `action_coverages`: 每个 action 的覆盖率标签，非法动作写入 0；
- `action_spectral_efficiencies`: 每个 action 的平均频谱效率标签，非法动作写入 0；
- 元数据 JSON: 地图路径、最佳动作、目标值等摘要。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from Autobs.compare_initialization_methods import LocalSurrogatePredictor
from Autobs.utils import (
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
REWARD_KEY_CHOICES = ("score",)


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
    parser.add_argument("--reward-key", default="score", choices=REWARD_KEY_CHOICES)
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), type=str)
    return parser


def collect_map_paths(args: argparse.Namespace) -> list[Path]:
    """注释
    功能: 根据命令行输入解析并收集本次要构建监督数据的地图路径。
    输入: `args` 为命令行参数，包含数据源路径与子集控制参数。
    输出: 规范化后的地图路径列表。
    示例: `map_paths = collect_map_paths(args)`。
    时间: 2026-04-27。
    """
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict[str, float]]:
    """注释
    功能: 枚举单张图上的全部合法动作，计算每个动作的统一 score、coverage 与平均频谱效率。
    输入: `pixel_map`、代理预测器 `predictor`、样本级 coverage/SE 目标以及 `reward_key`。
    输出: 逐动作 `scores`、`coverages`、`spectral_efficiencies`、最优动作索引和最优动作指标字典。
    示例: `scores, covs, ses, best_action, best_metrics = score_legal_actions(...)`。
    时间: 2026-04-27。
    """
    action_mask = calc_action_mask(pixel_map).astype(np.float32)
    legal_actions = np.flatnonzero(action_mask > 0.0)
    scores = np.full(action_mask.shape, NEGATIVE_SENTINEL, dtype=np.float32)
    coverages = np.zeros(action_mask.shape, dtype=np.float32)
    spectral_efficiencies = np.zeros(action_mask.shape, dtype=np.float32)
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
        coverages[action] = float(metrics["coverage"])
        spectral_efficiencies[action] = float(metrics["spectral_efficiency"])
        if best_metrics is None or value > float(best_metrics[reward_key]):
            best_metrics = metrics
            best_action = int(action)

    if best_metrics is None:
        raise ValueError("No legal actions found for this map")

    return (
        scores,
        coverages,
        spectral_efficiencies,
        best_action,
        {key: float(value) for key, value in best_metrics.items()},
    )


def build_dataset(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """注释
    功能: 针对整批地图构建动作排序监督数据集，并同时落地逐动作 coverage/SE 标签供二阶段 rerank 使用。
    输入: `args` 为命令行参数，包含地图集合、surrogate 权重和输出设置。
    输出: `(arrays, metadata)`，前者是 `.npz` 数组字典，后者是摘要元数据字典。
    示例: `arrays, metadata = build_dataset(args)`。
    时间: 2026-04-27。
    """
    map_paths = collect_map_paths(args)
    predictor = LocalSurrogatePredictor(args.model_path, args.network_type, args.device)
    heuristic_targets = load_heuristic_targets(args.heuristic_targets_path)

    observations: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    action_scores: list[np.ndarray] = []
    action_coverages: list[np.ndarray] = []
    action_ses: list[np.ndarray] = []
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
        scores, coverages, spectral_efficiencies, best_action, best_metrics = score_legal_actions(
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
        action_coverages.append(coverages)
        action_ses.append(spectral_efficiencies)
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
            f"score={best_metrics['score']:.4f} "
            f"coverage={best_metrics['coverage']:.4f} "
            f"se={best_metrics['spectral_efficiency']:.4f} "
            f"capacity={best_metrics['channel_capacity']:.4f}"
        )

    arrays = {
        "observations": np.stack(observations, axis=0).astype(np.float16),
        "action_masks": np.stack(action_masks, axis=0).astype(np.float16),
        "action_scores": np.stack(action_scores, axis=0).astype(np.float32),
        "action_coverages": np.stack(action_coverages, axis=0).astype(np.float32),
        "action_spectral_efficiencies": np.stack(action_ses, axis=0).astype(np.float32),
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
    """注释
    功能: 解析命令行、构建动作排序数据集并写出 `.npz` 与 `.json` 元数据文件。
    输入: `argv` 为可选命令行参数列表。
    输出: 无；完成后打印输出文件路径与样本数。
    示例: `main()`。
    时间: 2026-04-27。
    """
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
