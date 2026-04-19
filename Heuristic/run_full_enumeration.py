"""
命令示例:
python run_full_enumeration.py \
  --height-map data/b.png \
  --k-max 3 \
  --coverage-target 0.9 \
  --spectral-efficiency-target 3.5 \
  --device mps \
  --model-path /Users/epiphanyer/Desktop/coding/paper_experiment/Heuristic/model/RMNet.pt \
  --network-type rmnet \
  --output-dir outputs/full_enumeration

算法核心思路:
不再从全图抽样候选点，而是直接取 placement mask 上的全部合法像素。
然后按站点数 k=1..k_max 逐层枚举所有唯一站点组合，逐个调用代理模型求
coverage、spectral_efficiency、channel_capacity_mbps 和 score。脚本会返回最早满足 coverage / spectral_efficiency 双目标
的站点数与代表布局，同时统计每个 k 下可达到的最佳 score、coverage、spectral_efficiency。

参数说明:
--height-map: 输入灰度高度图，0 视为可统计的地面 RoI。
--k-max: 要枚举的最大站点数量。
--coverage-target: 目标覆盖率，用于判断某个 k 是否首次可行。
--spectral-efficiency-target: 目标平均频谱效率，用于判断某个 k 是否首次可行。
--model-path: 代理模型权重路径。
--network-type: 代理模型类型。
--output-dir: 输出目录。
--device: 推理设备，例如 cpu/cuda/mps。
--w1: 覆盖率不足惩罚权重。
--w2: 频谱效率不足惩罚权重。
--coverage-threshold-db: 最强单站接收功率覆盖阈值，单位 dBm。
--noise-coefficient-db: 接收机噪声系数，单位 dB。总噪声功率由热噪声密度、带宽和该系数在代码中计算。

脚本逻辑说明:
这个脚本把 placement mask 上的所有合法像素都视为候选点，再按 k=1 到 k_max 逐层枚举唯一组合。
每个组合都通过代理模型得到站点级预测图，并据此计算覆盖率、平均频谱效率、平均信道容量和总 score。
其中覆盖率按最强单站接收功率是否过阈值计算，同时输出服务站与干扰站形成的
`log2(1 + SINR)` 和 `CHANNEL_BANDWIDTH_HZ * log2(1 + SINR)`。它适合做小规模精确对照实验。
"""

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

import run_sa as core


@dataclass
class EnumeratedState:
    positions: np.ndarray
    metrics: core.EvalResult


@dataclass
class KSearchSummary:
    k: int
    combination_count: int
    best_score: EnumeratedState
    best_coverage: EnumeratedState
    best_capacity: EnumeratedState
    feasible_found: bool
    best_feasible: EnumeratedState | None


@dataclass
class EnumerationResult:
    per_k: Dict[int, KSearchSummary]
    first_feasible_k: int | None
    first_feasible: EnumeratedState | None
    global_best_score: EnumeratedState
    eval_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact base-station enumeration over all legal pixels")
    parser.add_argument("--height-map", required=True, type=str, help="Path to grayscale height map")
    parser.add_argument("--k-max", required=True, type=int, help="Maximum number of sites to enumerate")
    parser.add_argument("--coverage-target", required=True, type=float, help="Target coverage ratio")
    parser.add_argument(
        "--spectral-efficiency-target",
        required=True,
        type=float,
        help="Target average spectral efficiency",
    )
    parser.add_argument("--model-path", required=True, type=str, help="Checkpoint path")
    parser.add_argument("--network-type", required=True, type=str, help="Network type")
    parser.add_argument("--output-dir", required=True, type=str, help="Output directory")
    parser.add_argument("--device", default="mps", type=str, help="Torch device, e.g. cpu/cuda/mps")
    parser.add_argument("--w1", default=1.0, type=float, help="Penalty weight for coverage shortfall")
    parser.add_argument("--w2", default=1.0, type=float, help="Penalty weight for spectral-efficiency shortfall")
    parser.add_argument(
        "--coverage-threshold-db",
        default=core.DEFAULT_COVERAGE_THRESHOLD_DB,
        type=float,
        help="Coverage threshold on strongest per-pixel received power in dBm",
    )
    parser.add_argument(
        "--noise-coefficient-db",
        default=core.NOISE_COEFFICIENT_DB,
        type=float,
        help="Receiver noise coefficient in dB; total noise power is computed in code",
    )
    return parser.parse_args()


class FullEnumerationOptimizer:
    def __init__(
        self,
        predictor: object,
        height_map: np.ndarray,
        k_max: int,
        coverage_target: float,
        capacity_target: float,
        w1: float,
        w2: float,
        coverage_threshold_db: float,
        noise_coefficient_db: float,
    ) -> None:
        self.predictor = predictor
        self.height_map = height_map
        self.k_max = k_max
        self.coverage_target = coverage_target
        self.capacity_target = capacity_target
        self.w1 = w1
        self.w2 = w2
        self.coverage_threshold_db = coverage_threshold_db
        self.noise_coefficient_db = noise_coefficient_db
        self.placement_mask = core.build_placement_mask(height_map)
        self.roi_mask = core.build_roi_mask(height_map)
        self.eval_count = 0
        self.history: List[Dict[str, float | int | str]] = []

    def _evaluate_positions(self, positions: np.ndarray) -> core.EvalResult:
        self.eval_count += 1
        site_pathgain_db = self.predictor.predict_site_maps(self.height_map, positions)
        pathgain_db = np.max(site_pathgain_db, axis=0).astype(np.float32)
        rx_power_dbm = core.TX_POWER_DBM + site_pathgain_db.astype(np.float64)
        summed_rx_power_mw = np.sum(np.power(10.0, rx_power_dbm / 10.0), axis=0).astype(np.float32)
        covered_site_counts = np.count_nonzero(rx_power_dbm >= self.coverage_threshold_db, axis=0).astype(np.int16)
        return core.evaluate(
            pathgain_db=pathgain_db,
            summed_rx_power_mw=summed_rx_power_mw,
            roi_mask=self.roi_mask,
            coverage_target=self.coverage_target,
            spectral_efficiency_target=self.capacity_target,
            w1=self.w1,
            w2=self.w2,
            coverage_threshold_db=self.coverage_threshold_db,
            noise_coefficient_db=self.noise_coefficient_db,
            covered_site_counts=covered_site_counts,
        )

    def _candidates(self) -> np.ndarray:
        ys, xs = np.where(self.placement_mask)
        if len(xs) == 0:
            raise ValueError("No legal placement pixels available for exhaustive enumeration")
        return np.stack([xs, ys], axis=1).astype(np.float32)

    def optimize(self) -> EnumerationResult:
        candidates = self._candidates()
        if self.k_max < 1:
            raise ValueError("k_max must be at least 1")
        if self.k_max > len(candidates):
            raise ValueError(
                f"k_max={self.k_max} exceeds the number of legal placement pixels ({len(candidates)})"
            )

        per_k: Dict[int, KSearchSummary] = {}
        first_feasible_k: int | None = None
        first_feasible: EnumeratedState | None = None
        global_best_score: EnumeratedState | None = None

        for k in range(1, self.k_max + 1):
            best_score: EnumeratedState | None = None
            best_coverage: EnumeratedState | None = None
            best_capacity: EnumeratedState | None = None
            best_feasible: EnumeratedState | None = None
            combination_count = 0

            for combo in itertools.combinations(range(len(candidates)), k):
                positions = candidates[list(combo)].copy()
                metrics = self._evaluate_positions(positions)
                state = EnumeratedState(positions=positions, metrics=metrics)
                combination_count += 1

                if self._is_better_state(state, best_score, "score"):
                    best_score = state
                if self._is_better_state(state, best_coverage, "coverage"):
                    best_coverage = state
                if self._is_better_state(state, best_capacity, "capacity"):
                    best_capacity = state
                if self._meets_targets(metrics) and self._is_better_state(state, best_feasible, "score"):
                    best_feasible = state

            if best_score is None or best_coverage is None or best_capacity is None:
                raise RuntimeError(f"No combinations evaluated for k={k}")

            summary = KSearchSummary(
                k=k,
                combination_count=combination_count,
                best_score=best_score,
                best_coverage=best_coverage,
                best_capacity=best_capacity,
                feasible_found=best_feasible is not None,
                best_feasible=best_feasible,
            )
            per_k[k] = summary

            if first_feasible_k is None and best_feasible is not None:
                first_feasible_k = k
                first_feasible = best_feasible

            if self._is_better_state(best_score, global_best_score, "score"):
                global_best_score = best_score

            self.history.append(
                {
                    "k": k,
                    "combination_count": combination_count,
                    "eval_count": self.eval_count,
                    "feasible_found": int(best_feasible is not None),
                    "best_score": best_score.metrics.score,
                    "best_score_coverage": best_score.metrics.coverage,
                    "best_score_spectral_efficiency": best_score.metrics.spectral_efficiency,
                    "best_score_channel_capacity_mbps": best_score.metrics.channel_capacity,
                    "best_score_redundancy_rate": best_score.metrics.redundancy_rate,
                    "best_coverage": best_coverage.metrics.coverage,
                    "best_coverage_score": best_coverage.metrics.score,
                    "best_coverage_redundancy_rate": best_coverage.metrics.redundancy_rate,
                    "best_spectral_efficiency": best_capacity.metrics.spectral_efficiency,
                    "best_channel_capacity_mbps": best_capacity.metrics.channel_capacity,
                    "best_capacity_score": best_capacity.metrics.score,
                    "best_capacity_redundancy_rate": best_capacity.metrics.redundancy_rate,
                    "feasible_score": "" if best_feasible is None else best_feasible.metrics.score,
                    "feasible_coverage": "" if best_feasible is None else best_feasible.metrics.coverage,
                    "feasible_spectral_efficiency": "" if best_feasible is None else best_feasible.metrics.spectral_efficiency,
                    "feasible_channel_capacity_mbps": "" if best_feasible is None else best_feasible.metrics.channel_capacity,
                    "feasible_redundancy_rate": "" if best_feasible is None else best_feasible.metrics.redundancy_rate,
                }
            )

        if global_best_score is None:
            raise RuntimeError("Exhaustive enumeration produced no evaluated layouts")

        return EnumerationResult(
            per_k=per_k,
            first_feasible_k=first_feasible_k,
            first_feasible=first_feasible,
            global_best_score=global_best_score,
            eval_count=self.eval_count,
        )

    def _meets_targets(self, metrics: core.EvalResult) -> bool:
        return metrics.coverage >= self.coverage_target and metrics.spectral_efficiency >= self.capacity_target

    def _is_better_state(
        self,
        candidate: EnumeratedState | None,
        incumbent: EnumeratedState | None,
        key: str,
    ) -> bool:
        if candidate is None:
            return False
        if incumbent is None:
            return True

        candidate_rank = self._metric_rank(candidate.metrics, key)
        incumbent_rank = self._metric_rank(incumbent.metrics, key)
        if candidate_rank != incumbent_rank:
            return candidate_rank > incumbent_rank
        return self._rounded_position_key(candidate.positions) < self._rounded_position_key(incumbent.positions)

    def _metric_rank(self, metrics: core.EvalResult, key: str) -> tuple[float, float, float]:
        if key == "score":
            return (metrics.score, metrics.coverage, metrics.spectral_efficiency)
        if key == "coverage":
            return (metrics.coverage, metrics.spectral_efficiency, metrics.score)
        if key == "capacity":
            return (metrics.spectral_efficiency, metrics.coverage, metrics.score)
        raise ValueError(f"Unsupported ranking key: {key}")

    def _rounded_position_key(self, positions: np.ndarray) -> tuple[int, ...]:
        return tuple(np.rint(positions).astype(int).reshape(-1).tolist())


def _state_to_dict(state: EnumeratedState | None) -> Dict[str, object] | None:
    if state is None:
        return None
    return {
        "score": state.metrics.score,
        "coverage": state.metrics.coverage,
        "spectral_efficiency": state.metrics.spectral_efficiency,
        "channel_capacity_mbps": state.metrics.channel_capacity,
        "redundancy_rate": state.metrics.redundancy_rate,
        "base_score": state.metrics.base_score,
        "penalty": state.metrics.penalty,
        "positions_xy": state.positions.astype(float).tolist(),
        "positions_xy_rounded": np.rint(state.positions).astype(int).tolist(),
    }


def _primary_state(result: EnumerationResult) -> tuple[str, EnumeratedState]:
    if result.first_feasible is not None:
        return "first_feasible", result.first_feasible
    return "global_best_score", result.global_best_score


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    height_map = core.load_height_map(args.height_map)
    predictor = core.RadioMapPredictor(
        model_path=args.model_path,
        network_type=args.network_type,
        device_name=args.device,
    )
    optimizer = FullEnumerationOptimizer(
        predictor=predictor,
        height_map=height_map,
        k_max=args.k_max,
        coverage_target=args.coverage_target,
        capacity_target=args.spectral_efficiency_target,
        w1=args.w1,
        w2=args.w2,
        coverage_threshold_db=args.coverage_threshold_db,
        noise_coefficient_db=args.noise_coefficient_db,
    )
    result = optimizer.optimize()

    primary_kind, primary_state = _primary_state(result)
    best_layout_path = output_dir / "best_layout.npy"
    history_path = output_dir / "history.csv"
    best_metrics_path = output_dir / "best_metrics.json"
    best_prediction_path = output_dir / "best_prediction.png"
    height_map_preview_path = output_dir / "height_map.png"
    site_map_preview_path = output_dir / "site_map.png"

    np.save(best_layout_path, primary_state.positions.astype(np.float32))
    core.save_history_csv(history_path, optimizer.history)
    best_pathgain_db = predictor.predict(height_map, primary_state.positions)
    core.save_pathgain_preview(best_pathgain_db, primary_state.positions, best_prediction_path)
    core.save_height_map_preview(height_map, height_map_preview_path)
    core.save_site_map_preview(height_map, primary_state.positions, site_map_preview_path)

    k_summaries = []
    for k in range(1, args.k_max + 1):
        summary = result.per_k[k]
        k_summaries.append(
            {
                "k": k,
                "combination_count": summary.combination_count,
                "feasible_found": summary.feasible_found,
                "best_score": _state_to_dict(summary.best_score),
                "best_coverage": _state_to_dict(summary.best_coverage),
                "best_capacity": _state_to_dict(summary.best_capacity),
                "best_feasible": _state_to_dict(summary.best_feasible),
            }
        )

    best_metrics = {
        "primary_layout_kind": primary_kind,
        "first_feasible_k": result.first_feasible_k,
        "first_feasible": _state_to_dict(result.first_feasible),
        "global_best_score": _state_to_dict(result.global_best_score),
        "eval_count": result.eval_count,
        "checkpoint": getattr(predictor, "model_path", None),
        "network_type": getattr(predictor, "network_type", None),
        "device": str(getattr(predictor, "device", args.device)),
        "height_map": str(Path(args.height_map).resolve()),
        "k_max": args.k_max,
        "coverage_target": args.coverage_target,
        "spectral_efficiency_target": args.spectral_efficiency_target,
        "coverage_threshold_db": args.coverage_threshold_db,
        "noise_coefficient_db": args.noise_coefficient_db,
        "total_noise_power_dbm": core.compute_total_noise_power_dbm(args.noise_coefficient_db),
        "k_summaries": k_summaries,
    }
    best_metrics_path.write_text(json.dumps(best_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"best_layout_npy: {best_layout_path.resolve()}")
    print(f"best_prediction_png: {best_prediction_path.resolve()}")
    print(f"height_map_png: {height_map_preview_path.resolve()}")
    print(f"site_map_png: {site_map_preview_path.resolve()}")
    print(f"history_csv: {history_path.resolve()}")
    print(f"best_metrics_json: {best_metrics_path.resolve()}")
    print(f"primary_layout_kind: {primary_kind}")
    print(f"first_feasible_k: {result.first_feasible_k}")
    print(f"global_best_score: {result.global_best_score.metrics.score:.6f}")
    print(f"primary_redundancy_rate: {primary_state.metrics.redundancy_rate:.6f}")
    print(f"eval_count: {result.eval_count}")
    print(f"total_runtime_sec: {time.perf_counter() - start_time:.6f}")


if __name__ == "__main__":
    main()
