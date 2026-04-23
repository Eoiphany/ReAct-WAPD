"""
命令示例:
python run_candidate_enumeration.py \
  --height-map data/b.png \
  --k-max 12 \
  --coverage-target 0.9 \
  --spectral-efficiency-target 3.5 \
  --max-evals 200 \
  --device mps \
  --model-path /Users/epiphanyer/Desktop/coding/paper_experiment/Heuristic/model/RMNet.pt \
  --network-type rmnet \
  --output-dir outputs/smoke_test \
  --candidate-limit 2000 \
  --candidate-stride 4

算法核心思路:
这个脚本会先从全图抽样出一批候选点，再枚举其中满足站间距约束的站点组合，
逐个调用代理模型求分，并返回得分最高的组合。它不是“全合法像素上的真穷举”，
而是“候选集上的组合枚举”。候选集较小时结果直接、可解释；
候选集一大，组合数量会迅速膨胀。

参数说明:
--height-map: 输入灰度高度图，0 视为可统计的地面 RoI。
--k-max: 需要同时部署的站点数量。
--coverage-target: 目标覆盖率，用于 score 中的惩罚项。
--spectral-efficiency-target: 目标平均频谱效率，用于 score 中的惩罚项。
--max-evals: 最大代理模型评估次数。
--model-path: 代理模型权重路径。
--network-type: 代理模型类型。
--output-dir: 输出目录。
--device: 推理设备，例如 cpu/cuda/mps。
--d-min: 站点之间的最小距离约束，单位像素。
--w1: 覆盖率不足惩罚权重。
--w2: 频谱效率不足惩罚权重。
--coverage-threshold-db: 最强单站接收功率覆盖阈值，单位 dBm。
--noise-coefficient-db: 接收机噪声系数，单位 dB。总噪声功率由热噪声密度、带宽和该系数在代码中计算。
--candidate-stride: 候选点采样步长。
--candidate-limit: 最大候选点数。

脚本逻辑说明:
这个脚本先在 placement mask 上按步长抽样出候选站点，再在候选集里枚举所有满足间距约束的组合。
每个组合都调用代理模型评估，覆盖率按最强单站接收功率是否过阈值计算，同时输出平均频谱效率
`log2(1 + SINR)` 和平均信道容量 `CHANNEL_BANDWIDTH_HZ * log2(1 + SINR)`。它是“候选集上的组合枚举”，适合做小规模可解释基线。
"""

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

import run_sa as core


class CandidateEnumerationOptimizer:
    def __init__(
        self,
        predictor: core.RadioMapPredictor,
        height_map: np.ndarray,
        k_max: int,
        coverage_target: float,
        capacity_target: float,
        max_evals: int,
        d_min: float,
        w1: float,
        w2: float,
        coverage_threshold_db: float,
        noise_coefficient_db: float,
        candidate_stride: int,
        candidate_limit: int,
    ) -> None:
        self.predictor = predictor
        self.height_map = height_map
        self.k_max = k_max
        self.coverage_target = coverage_target
        self.capacity_target = capacity_target
        self.max_evals = max_evals
        self.d_min = d_min
        self.w1 = w1
        self.w2 = w2
        self.coverage_threshold_db = coverage_threshold_db
        self.noise_coefficient_db = noise_coefficient_db
        self.placement_mask = core.build_placement_mask(height_map)
        self.roi_mask = core.build_roi_mask(height_map)
        self.rng = np.random.default_rng()
        self.candidate_stride = candidate_stride
        self.candidate_limit = candidate_limit
        self.eval_count = 0
        self.history: List[Dict[str, float]] = []

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
        mask = (xs % self.candidate_stride == 0) & (ys % self.candidate_stride == 0)
        candidates = np.stack([xs[mask], ys[mask]], axis=1).astype(np.float32)
        if len(candidates) == 0:
            candidates = np.stack([xs, ys], axis=1).astype(np.float32)
        if len(candidates) > self.candidate_limit:
            idx = self.rng.choice(len(candidates), size=self.candidate_limit, replace=False)
            candidates = candidates[idx]
        return candidates

    def optimize(self) -> core.SAState:
        candidates = self._candidates()
        best: core.SAState | None = None
        checked = 0
        for combo in itertools.combinations(range(len(candidates)), self.k_max):
            if self.eval_count >= self.max_evals:
                break
            positions = candidates[list(combo)].copy()
            valid = True
            for idx in range(len(positions)):
                if core.pairwise_min_distance(positions, idx) < self.d_min:
                    valid = False
                    break
            if not valid:
                continue
            metrics = self._evaluate_positions(positions)
            checked += 1
            state = core.SAState(positions=positions.copy(), metrics=metrics)
            if best is None or state.metrics.score > best.metrics.score:
                best = state
                self.history.append(
                    {
                        "checked_combinations": checked,
                        "eval_count": self.eval_count,
                        "best_score": best.metrics.score,
                        "best_coverage": best.metrics.coverage,
                        "best_spectral_efficiency": best.metrics.spectral_efficiency,
                        "best_channel_capacity_mbps": best.metrics.channel_capacity,
                        "best_redundancy_rate": best.metrics.redundancy_rate,
                    }
                )
        if best is None:
            raise RuntimeError("No valid combination found under current candidate-enumeration settings")
        return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Base-station deployment via candidate-set enumeration on sampled sites")
    parser.add_argument("--height-map", required=True, type=str, help="Path to grayscale height map")
    parser.add_argument("--k-max", required=True, type=int, help="Number of sites in each enumerated layout")
    parser.add_argument("--coverage-target", required=True, type=float, help="Target coverage ratio")
    parser.add_argument(
        "--spectral-efficiency-target",
        required=True,
        type=float,
        help="Target average spectral efficiency",
    )
    parser.add_argument("--max-evals", required=True, type=int, help="Maximum number of radiomap evaluations")
    parser.add_argument("--model-path", required=True, type=str, help="Checkpoint path")
    parser.add_argument("--network-type", required=True, type=str, help="Network type")
    parser.add_argument("--output-dir", required=True, type=str, help="Output directory")
    parser.add_argument("--device", default="mps", type=str, help="Torch device, e.g. cpu/cuda/mps")
    parser.add_argument("--d-min", default=12.0, type=float, help="Minimum pairwise site distance in pixels")
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
    parser.add_argument("--candidate-stride", default=16, type=int, help="Grid stride when sampling candidate sites")
    parser.add_argument("--candidate-limit", default=80, type=int, help="Maximum number of sampled candidate sites")
    return parser.parse_args()


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    predictor = core.RadioMapPredictor(args.model_path, args.network_type, args.device)
    height_map = core.load_height_map(args.height_map)
    optimizer = CandidateEnumerationOptimizer(
        predictor=predictor,
        height_map=height_map,
        k_max=args.k_max,
        coverage_target=args.coverage_target,
        capacity_target=args.spectral_efficiency_target,
        max_evals=args.max_evals,
        d_min=args.d_min,
        w1=args.w1,
        w2=args.w2,
        coverage_threshold_db=args.coverage_threshold_db,
        noise_coefficient_db=args.noise_coefficient_db,
        candidate_stride=args.candidate_stride,
        candidate_limit=args.candidate_limit,
    )
    best = optimizer.optimize()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "best_layout.npy", best.positions.astype(np.float32))
    core.save_history_csv(output_dir / "history.csv", optimizer.history)
    best_pathloss_db = predictor.predict(height_map, best.positions)
    core.save_pathgain_preview(best_pathloss_db, best.positions, output_dir / "best_prediction.png")
    core.save_height_map_preview(height_map, output_dir / "height_map.png")
    core.save_site_map_preview(height_map, best.positions, output_dir / "site_map.png")
    metrics = {
        "best_score": best.metrics.score,
        "coverage": best.metrics.coverage,
        "spectral_efficiency": best.metrics.spectral_efficiency,
        "channel_capacity_mbps": best.metrics.channel_capacity,
        "redundancy_rate": best.metrics.redundancy_rate,
        "base_score": best.metrics.base_score,
        "penalty": best.metrics.penalty,
        "eval_count": optimizer.eval_count,
        "checkpoint": predictor.model_path,
        "network_type": predictor.network_type,
        "device": str(predictor.device),
        "k_max": args.k_max,
        "spectral_efficiency_target": args.spectral_efficiency_target,
        "coverage_threshold_db": optimizer.coverage_threshold_db,
        "noise_coefficient_db": optimizer.noise_coefficient_db,
        "total_noise_power_dbm": core.compute_total_noise_power_dbm(optimizer.noise_coefficient_db),
        "best_positions_xy": best.positions.astype(float).tolist(),
        "total_runtime_sec": time.perf_counter() - start_time,
    }
    (output_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"best_score: {best.metrics.score:.6f}")
    print(f"coverage: {best.metrics.coverage:.6f}")
    print(f"spectral_efficiency: {best.metrics.spectral_efficiency:.6f}")
    print(f"channel_capacity_mbps: {best.metrics.channel_capacity:.6f}")
    print(f"redundancy_rate: {best.metrics.redundancy_rate:.6f}")
    print(f"best_positions_xy: {np.array2string(best.positions, precision=3)}")
    print(f"total_runtime_sec: {time.perf_counter() - start_time:.6f}")


if __name__ == "__main__":
    main()
