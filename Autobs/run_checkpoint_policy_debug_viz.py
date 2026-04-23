"""注释
命令示例:
python -m Autobs.run_checkpoint_policy_debug_viz \
  --image /Users/epiphanyer/Desktop/coding/test/dataset/png/buildingsWHeight/8.png \
  --checkpoint /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/checkpoints_zonghe \
  --model-path /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/runs/pmnet_radiomap3dseer/16_0.0001_0.5_10/pmnet_radiomap3dseer_best.pt \
  --network-type pmnet \
  --output-dir /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/outputs/policy_debug_viz

参数含义:
- --image: 单张灰度高度图路径。
- --checkpoint: PPO checkpoint 目录。
- --model-path: surrogate 权重路径。
- --network-type: surrogate 模型类型。
- --device: surrogate 推理设备。
- --output-dir: 输出目录。
- --top-k: 摘要里保留前 K 个动作候选。

脚本逻辑说明:
本脚本在单张图上同时计算 PPO 选点和 surrogate 视角下的最优合法动作点。它会枚举全部合法动作，
对每个动作计算 coverage / spectral efficiency / score，并输出 PPO 选点在全部合法动作中的排名，
再保存原图+点位对比图和动作网格 score 热力图。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from Autobs.compare_initialization_methods import (
    LocalSurrogatePredictor,
    add_position_heights,
    evaluate_layout,
)
from Autobs.env.utils import (
    DEFAULT_COVERAGE_TARGET,
    DEFAULT_COVERAGE_THRESHOLD_DB,
    DEFAULT_NOISE_COEFFICIENT_DB,
    DEFAULT_SPECTRAL_EFFICIENCY_TARGET,
    DEFAULT_W1,
    DEFAULT_W2,
    calc_action_mask,
    calc_upsampling_loc,
    load_map_normalized,
)
from Autobs.paths import CHECKPOINT_DIR, DEFAULT_RMNET_WEIGHTS, PACKAGE_ROOT
from Autobs.run_checkpoint_rmnet_viz import (
    build_policy_observation,
    compute_checkpoint_action,
    load_checkpoint_agent,
)


DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs" / "policy_debug_viz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose PPO action quality on one image using surrogate rankings")
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR), type=str)
    parser.add_argument("--model-path", default=str(DEFAULT_RMNET_WEIGHTS), type=str)
    parser.add_argument("--network-type", default="rmnet", choices=["pmnet", "pmnet_v3", "rmnet", "rmnet_v3"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), type=str)
    parser.add_argument("--top-k", default=10, type=int)
    return parser


def rank_action(candidates: list[dict[str, Any]], action: int, metric: str) -> int | None:
    ranked = sorted(candidates, key=lambda item: (-float(item[metric]), int(item["action"])))
    for idx, item in enumerate(ranked, start=1):
        if int(item["action"]) == int(action):
            return idx
    return None


def build_score_grid(candidates: list[dict[str, Any]], side: int = 32) -> np.ndarray:
    grid = np.full((side, side), np.nan, dtype=np.float32)
    for item in candidates:
        row, col = divmod(int(item["action"]), side)
        grid[row, col] = float(item["score"])
    return grid


def save_debug_artifacts(summary: dict[str, Any], candidates: list[dict[str, Any]], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "policy_debug_summary.json"
    candidates_csv = output_dir / "policy_debug_candidates.csv"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with candidates_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0].keys()) if candidates else ["action"])
        writer.writeheader()
        writer.writerows(candidates)
    return {
        "summary_json": str(summary_json),
        "candidates_csv": str(candidates_csv),
    }


def render_policy_debug_visualization(
    pixel_map: np.ndarray,
    score_grid: np.ndarray,
    summary: dict[str, Any],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".mplconfig"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax = axes[0]
    ax.imshow(pixel_map, cmap="gray", vmin=0.0, vmax=1.0)
    ppo_x, ppo_y, _ = summary["ppo_positions_xyz"][0]
    best_x, best_y, _ = summary["best_positions_xyz"][0]
    ax.scatter([ppo_x], [ppo_y], c="#d62828", marker="x", s=64, linewidths=1.8, label="PPO")
    ax.scatter([best_x], [best_y], c="#2a9d8f", marker="+", s=72, linewidths=1.8, label="Best")
    ax.set_title("PPO vs Surrogate Best")
    ax.axis("off")
    ax.legend(loc="lower right")

    ax = axes[1]
    image = ax.imshow(score_grid, cmap="viridis")
    ppo_row, ppo_col = divmod(int(summary["ppo_action"]), score_grid.shape[1])
    best_row, best_col = divmod(int(summary["best_action"]), score_grid.shape[1])
    ax.scatter([ppo_col], [ppo_row], c="#d62828", marker="x", s=64, linewidths=1.8)
    ax.scatter([best_col], [best_row], c="#ffffff", marker="+", s=72, linewidths=1.8)
    ax.set_title(
        f"Score Grid\nppo={summary['ppo_score']:.4f} best={summary['best_score']:.4f}"
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def evaluate_legal_actions(
    pixel_map: np.ndarray,
    predictor: LocalSurrogatePredictor,
    legal_actions: np.ndarray,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for action in legal_actions.astype(int).tolist():
        tx_loc = calc_upsampling_loc(action, pixel_map)
        metrics = evaluate_layout(
            pixel_map,
            [tx_loc],
            predictor,
            coverage_target=DEFAULT_COVERAGE_TARGET,
            spectral_efficiency_target=DEFAULT_SPECTRAL_EFFICIENCY_TARGET,
            w1=DEFAULT_W1,
            w2=DEFAULT_W2,
            coverage_threshold_db=DEFAULT_COVERAGE_THRESHOLD_DB,
            noise_coefficient_db=DEFAULT_NOISE_COEFFICIENT_DB,
        )
        positions_xyz = add_position_heights(pixel_map, [[tx_loc[1], tx_loc[0]]])
        candidates.append(
            {
                "action": action,
                "coverage": float(metrics["coverage"]),
                "spectral_efficiency": float(metrics["spectral_efficiency"]),
                "channel_capacity_mbps": float(metrics["channel_capacity_mbps"]),
                "score": float(metrics["score"]),
                "positions_xyz": positions_xyz[0] if positions_xyz else None,
            }
        )
    return candidates


def diagnose_policy(
    image_path: str | Path,
    checkpoint_path: str | Path,
    model_path: str | Path,
    network_type: str,
    device_name: str,
    top_k: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    pixel_map = load_map_normalized(image_path)
    observation = build_policy_observation(pixel_map)
    legal_actions = np.flatnonzero(observation["action_mask"] > 0.0)
    if legal_actions.size == 0:
        raise ValueError("No legal actions on this map")

    predictor = LocalSurrogatePredictor(str(model_path), network_type, device_name)
    agent = load_checkpoint_agent(checkpoint_path)
    try:
        ppo_action = int(compute_checkpoint_action(agent, observation, explore=False))
    finally:
        stop = getattr(agent, "stop", None)
        if callable(stop):
            stop()

    candidates = evaluate_legal_actions(pixel_map, predictor, legal_actions)
    ranked = sorted(candidates, key=lambda item: (-float(item["score"]), int(item["action"])))
    best = ranked[0]
    ppo_rank_by_score = rank_action(candidates, ppo_action, "score")
    ppo_item = next(item for item in candidates if int(item["action"]) == ppo_action)

    summary = {
        "image": str(Path(image_path).expanduser().resolve()),
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "model_path": str(Path(model_path).expanduser().resolve()),
        "network_type": network_type,
        "legal_action_count": int(legal_actions.size),
        "ppo_action": int(ppo_action),
        "ppo_rank_by_score": ppo_rank_by_score,
        "ppo_score": float(ppo_item["score"]),
        "ppo_coverage": float(ppo_item["coverage"]),
        "ppo_spectral_efficiency": float(ppo_item["spectral_efficiency"]),
        "ppo_positions_xyz": [ppo_item["positions_xyz"]],
        "best_action": int(best["action"]),
        "best_score": float(best["score"]),
        "best_coverage": float(best["coverage"]),
        "best_spectral_efficiency": float(best["spectral_efficiency"]),
        "best_positions_xyz": [best["positions_xyz"]],
        "top_k_by_score": ranked[: max(1, top_k)],
    }

    output_dir = Path(output_dir).expanduser().resolve()
    score_grid = build_score_grid(candidates)
    artifacts = save_debug_artifacts(summary, candidates, output_dir)
    figure_path = output_dir / "policy_debug.png"
    render_policy_debug_visualization(pixel_map, score_grid, summary, figure_path)
    summary["artifacts"] = {
        **artifacts,
        "figure_png": str(figure_path),
    }
    Path(artifacts["summary_json"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = diagnose_policy(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        model_path=args.model_path,
        network_type=args.network_type,
        device_name=args.device,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
