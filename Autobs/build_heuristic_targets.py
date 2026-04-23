"""注释
命令示例:
python -m Autobs.build_heuristic_targets \
  --dataset-path /Users/epiphanyer/Desktop/coding/paper_experiment/dataset/png/buildingsWHeight \
  --dataset-limit 512 \
  --methods run_ga run_pso \
  --k-max 1 \
  --model-path /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/runs/pmnet_radiomap3dseer/16_0.0001_0.5_10/pmnet_radiomap3dseer_best.pt \
  --network-type pmnet \
  --output-path /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/outputs/heuristic_targets.json

参数含义:
- --dataset-path: 训练集目录或单图路径。
- --dataset-limit / --dataset-offset / --dataset-stride: 数据集子集控制。
- --methods: 用于生成目标的启发式方法，默认 `run_ga run_pso`。
- --target-aggregator: 多个启发式结果如何聚合，默认 `min`。
- 其余参数与 `compare_initialization_methods.py` 的启发式评估保持一致。

脚本逻辑说明:
本脚本离线跑指定启发式方法，按每张图生成 `coverage_target / spectral_efficiency_target`。
当多个方法同时成功时，默认取各指标的最小值作为训练目标，避免把门槛设得过高。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from Autobs.compare_initialization_methods import (
    NETWORK_CHOICES,
    _resolve_device,
    normalize_method_name,
    run_heuristic_method,
)
from Autobs.env.utils import resolve_city_map_paths
from Autobs.paths import DEFAULT_DATASET_MAP_DIR, PACKAGE_ROOT


DEFAULT_OUTPUT_PATH = PACKAGE_ROOT / "outputs" / "heuristic_targets.json"
DEFAULT_METHODS = ("run_ga", "run_pso")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline-build per-map heuristic coverage and spectral-efficiency targets")
    parser.add_argument("--dataset-path", "--city-map-path", dest="city_map_path", default=str(DEFAULT_DATASET_MAP_DIR))
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--dataset-offset", type=int, default=0)
    parser.add_argument("--dataset-stride", type=int, default=1)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--target-aggregator", choices=["min", "mean", "max"], default="min")
    parser.add_argument("--k-max", default=1, type=int)
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--network-type", default="pmnet", choices=NETWORK_CHOICES)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), type=str)
    parser.add_argument("--coverage-target", default=0.9, type=float)
    parser.add_argument("--spectral-efficiency-target", default=3.5, type=float)
    parser.add_argument("--max-evals", default=200, type=int)
    parser.add_argument("--d-min", default=12.0, type=float)
    parser.add_argument("--repair-max-tries", default=100, type=int)
    parser.add_argument("--w1", default=1.0, type=float)
    parser.add_argument("--w2", default=1.0, type=float)
    parser.add_argument("--coverage-threshold-db", default=-117.0, type=float)
    parser.add_argument("--noise-coefficient-db", default=10.0, type=float)
    parser.add_argument("--sa-initial-temp", default=1.0, type=float)
    parser.add_argument("--sa-cooling-rate", default=0.995, type=float)
    parser.add_argument("--sa-gaussian-sigma", default=6.0, type=float)
    parser.add_argument("--greedy-candidate-stride", default=8, type=int)
    parser.add_argument("--greedy-candidate-limit", default=5000, type=int)
    parser.add_argument("--ga-population-size", default=24, type=int)
    parser.add_argument("--ga-elite-size", default=4, type=int)
    parser.add_argument("--ga-tournament-size", default=3, type=int)
    parser.add_argument("--ga-mutation-rate", default=0.3, type=float)
    parser.add_argument("--ga-gaussian-sigma", default=6.0, type=float)
    parser.add_argument("--pso-swarm-size", default=20, type=int)
    parser.add_argument("--pso-inertia", default=0.7, type=float)
    parser.add_argument("--pso-c1", default=1.4, type=float)
    parser.add_argument("--pso-c2", default=1.4, type=float)
    parser.add_argument("--pso-velocity-clamp", default=8.0, type=float)
    parser.add_argument("--bruteforce-candidate-stride", default=16, type=int)
    parser.add_argument("--bruteforce-candidate-limit", default=80, type=int)
    return parser


def aggregate_targets(records: list[dict[str, Any]], aggregator: str = "min") -> dict[str, float]:
    if not records:
        raise ValueError("No heuristic records provided for aggregation")
    coverage_values = [float(item["coverage"]) for item in records]
    se_values = [float(item["spectral_efficiency"]) for item in records]
    if aggregator == "min":
        coverage_target = min(coverage_values)
        spectral_efficiency_target = min(se_values)
    elif aggregator == "max":
        coverage_target = max(coverage_values)
        spectral_efficiency_target = max(se_values)
    elif aggregator == "mean":
        coverage_target = sum(coverage_values) / len(coverage_values)
        spectral_efficiency_target = sum(se_values) / len(se_values)
    else:
        raise ValueError(f"Unsupported aggregator: {aggregator}")
    return {
        "coverage_target": float(coverage_target),
        "spectral_efficiency_target": float(spectral_efficiency_target),
    }


def collect_map_paths(args: argparse.Namespace) -> list[Path]:
    paths = resolve_city_map_paths(
        args.city_map_path,
        DEFAULT_DATASET_MAP_DIR,
        dataset_limit=args.dataset_limit,
        dataset_offset=args.dataset_offset,
        dataset_stride=args.dataset_stride,
    )
    return [Path(path).expanduser().resolve() for path in paths]


def build_targets(args: argparse.Namespace) -> dict[str, Any]:
    map_paths = collect_map_paths(args)
    output_root_dir = Path(args.output_path).expanduser().resolve().parent / "heuristic_target_runs"
    output_root_dir.mkdir(parents=True, exist_ok=True)
    targets_by_image: dict[str, Any] = {}

    for map_index, map_path in enumerate(map_paths, start=1):
        print(f"[{map_index}/{len(map_paths)}] {map_path.name}")
        method_records: list[dict[str, Any]] = []
        for method in [normalize_method_name(method_name) for method_name in args.methods]:
            record = run_heuristic_method(method, map_path, output_root_dir, args)
            method_records.append(record)
            print(
                f"  {method}: coverage={record['coverage']:.4f} "
                f"se={record['spectral_efficiency']:.4f}"
            )
        aggregated = aggregate_targets(method_records, aggregator=args.target_aggregator)
        print(
            f"  targets: coverage_target={aggregated['coverage_target']:.4f} "
            f"spectral_efficiency_target={aggregated['spectral_efficiency_target']:.4f}"
        )
        targets_by_image[str(map_path)] = {
            **aggregated,
            "methods": method_records,
        }

    return {
        "city_map_path": str(Path(args.city_map_path).expanduser().resolve()),
        "resolved_map_count": len(map_paths),
        "methods": [normalize_method_name(method_name) for method_name in args.methods],
        "target_aggregator": args.target_aggregator,
        "targets_by_image": targets_by_image,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.device = str(_resolve_device(args.device))
    print(
        f"Building heuristic targets: dataset={Path(args.city_map_path).expanduser().resolve()} "
        f"methods={[normalize_method_name(method_name) for method_name in args.methods]} "
        f"aggregator={args.target_aggregator}"
    )
    payload = build_targets(args)
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "resolved_map_count": payload["resolved_map_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
