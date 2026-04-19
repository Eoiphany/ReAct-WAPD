"""
命令示例:
python run_pso.py \
  --height-map data/b.png \
  --k-max 12 \
  --coverage-target 0.9 \
  --spectral-efficiency-target 3.5 \
  --max-evals 200 \
  --device mps \
  --model-path /Users/epiphanyer/Desktop/coding/paper_experiment/Heuristic/model/RMNet.pt \
  --network-type rmnet \
  --output-dir outputs/smoke_test_pso \
  --swarm-size 20 \
  --inertia 0.7 \
  --c1 1.4 \
  --c2 1.4 \
  --velocity-clamp 8.0

算法核心思路:
粒子群优化(PSO)把每个候选布局看作一个粒子。每个粒子同时维护自己的历史最优位置
和全局最优位置，并根据惯性项、个体学习项、群体学习项更新速度与位置。
每次移动后都会做约束修复并重新评估。它的特点是搜索方向更连续，
通常比纯随机扰动更快向当前最优解附近靠拢。

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
--repair-max-tries: 修复非法站点位置的最大尝试次数。
--w1: 覆盖率不足惩罚权重。
--w2: 频谱效率不足惩罚权重。
--coverage-threshold-db: 最强单站接收功率覆盖阈值，单位 dBm。
--noise-coefficient-db: 接收机噪声系数，单位 dB。总噪声功率由热噪声密度、带宽和该系数在代码中计算。
--swarm-size: 粒子数量。
--inertia: 惯性项。
--c1: 个体学习因子。
--c2: 全局学习因子。
--velocity-clamp: 速度裁剪上限。

脚本逻辑说明:
PSO 把每个候选布局视为一个粒子，每轮同时参考粒子自己的历史最好位置和全局最好位置来更新速度与位置。
每次评估布局时，覆盖率按最强单站接收功率是否过阈值计算；同时把最强站作为服务站，其余站点作为干扰，
输出整个 RoI 的平均频谱效率 `log2(1 + SINR)` 和平均信道容量 `CHANNEL_BANDWIDTH_HZ * log2(1 + SINR)`。它适合在连续空间里做较平滑的群体搜索。
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

import run_sa as core


class ParticleSwarmOptimizer:
    def __init__(
        self,
        predictor: core.RadioMapPredictor,
        height_map: np.ndarray,
        k_max: int,
        coverage_target: float,
        capacity_target: float,
        max_evals: int,
        d_min: float,
        repair_max_tries: int,
        w1: float,
        w2: float,
        coverage_threshold_db: float,
        noise_coefficient_db: float,
        swarm_size: int,
        inertia: float,
        c1: float,
        c2: float,
        velocity_clamp: float,
    ) -> None:
        self.predictor = predictor
        self.height_map = height_map
        self.height, self.width = height_map.shape
        self.k_max = k_max
        self.coverage_target = coverage_target
        self.capacity_target = capacity_target
        self.max_evals = max_evals
        self.d_min = d_min
        self.repair_max_tries = repair_max_tries
        self.w1 = w1
        self.w2 = w2
        self.coverage_threshold_db = coverage_threshold_db
        self.noise_coefficient_db = noise_coefficient_db
        self.placement_mask = core.build_placement_mask(height_map)
        self.roi_mask = core.build_roi_mask(height_map)
        self.rng = np.random.default_rng()
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.velocity_clamp = velocity_clamp
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

    def optimize(self) -> core.SAState:
        positions = []
        velocities = []
        pbest_positions = []
        pbest_metrics = []
        for _ in range(self.swarm_size):
            pos = core.initialize_positions(
                self.k_max, self.placement_mask, self.d_min, self.rng, self.repair_max_tries
            )
            vel = self.rng.normal(0.0, 1.0, size=pos.shape).astype(np.float32)
            metrics = self._evaluate_positions(pos)
            positions.append(pos)
            velocities.append(vel)
            pbest_positions.append(pos.copy())
            pbest_metrics.append(metrics)

        best_idx = int(np.argmax([m.score for m in pbest_metrics]))
        gbest = core.SAState(positions=pbest_positions[best_idx].copy(), metrics=pbest_metrics[best_idx])
        iteration = 0
        self.history.append(self._history_row(iteration, gbest, gbest))

        while self.eval_count < self.max_evals:
            iteration += 1
            current_best = gbest
            for i in range(self.swarm_size):
                r1 = self.rng.random(size=positions[i].shape).astype(np.float32)
                r2 = self.rng.random(size=positions[i].shape).astype(np.float32)
                velocities[i] = (
                    self.inertia * velocities[i]
                    + self.c1 * r1 * (pbest_positions[i] - positions[i])
                    + self.c2 * r2 * (gbest.positions - positions[i])
                )
                velocities[i] = np.clip(velocities[i], -self.velocity_clamp, self.velocity_clamp)
                positions[i] = positions[i] + velocities[i]
                for site_idx in range(self.k_max):
                    core.repair_position(
                        positions[i],
                        site_idx,
                        self.placement_mask,
                        self.d_min,
                        self.rng,
                        self.repair_max_tries,
                    )
                if self.eval_count >= self.max_evals:
                    break
                metrics = self._evaluate_positions(positions[i])
                if metrics.score > pbest_metrics[i].score:
                    pbest_positions[i] = positions[i].copy()
                    pbest_metrics[i] = metrics
                if metrics.score > gbest.metrics.score:
                    gbest = core.SAState(positions=positions[i].copy(), metrics=metrics)
                if metrics.score > current_best.metrics.score:
                    current_best = core.SAState(positions=positions[i].copy(), metrics=metrics)
            self.history.append(self._history_row(iteration, current_best, gbest))
        return gbest

    def _history_row(self, iteration: int, current_best: core.SAState, best: core.SAState) -> Dict[str, float]:
        return {
            "iteration": iteration,
            "eval_count": self.eval_count,
            "current_best_score": current_best.metrics.score,
            "current_best_coverage": current_best.metrics.coverage,
            "current_best_spectral_efficiency": current_best.metrics.spectral_efficiency,
            "current_best_channel_capacity_mbps": current_best.metrics.channel_capacity,
            "current_best_redundancy_rate": current_best.metrics.redundancy_rate,
            "best_score": best.metrics.score,
            "best_coverage": best.metrics.coverage,
            "best_spectral_efficiency": best.metrics.spectral_efficiency,
            "best_channel_capacity_mbps": best.metrics.channel_capacity,
            "best_redundancy_rate": best.metrics.redundancy_rate,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Base-station deployment via particle swarm optimization")
    parser.add_argument("--height-map", required=True, type=str, help="Path to grayscale height map")
    parser.add_argument("--k-max", required=True, type=int, help="Number of sites in each particle layout")
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
    parser.add_argument("--repair-max-tries", default=100, type=int, help="Maximum attempts when repairing one site")
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
    parser.add_argument("--swarm-size", default=20, type=int, help="Number of particles in the swarm")
    parser.add_argument("--inertia", default=0.7, type=float, help="Inertia coefficient in velocity update")
    parser.add_argument("--c1", default=1.4, type=float, help="Cognitive learning coefficient")
    parser.add_argument("--c2", default=1.4, type=float, help="Social learning coefficient")
    parser.add_argument("--velocity-clamp", default=8.0, type=float, help="Absolute clamp on particle velocity")
    return parser.parse_args()


def write_outputs(output_dir: Path, optimizer, predictor, height_map: np.ndarray, best: core.SAState, k_max: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_layout_path = output_dir / "best_layout.npy"
    history_path = output_dir / "history.csv"
    best_metrics_path = output_dir / "best_metrics.json"
    best_prediction_path = output_dir / "best_prediction.png"
    height_map_preview_path = output_dir / "height_map.png"
    site_map_preview_path = output_dir / "site_map.png"

    np.save(best_layout_path, best.positions.astype(np.float32))
    core.save_history_csv(history_path, optimizer.history)
    best_pathloss_db = predictor.predict(height_map, best.positions)
    core.save_pathgain_preview(best_pathloss_db, best.positions, best_prediction_path)
    core.save_height_map_preview(height_map, height_map_preview_path)
    core.save_site_map_preview(height_map, best.positions, site_map_preview_path)

    best_metrics = {
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
        "k_max": k_max,
        "spectral_efficiency_target": optimizer.capacity_target,
        "coverage_threshold_db": optimizer.coverage_threshold_db,
        "noise_coefficient_db": optimizer.noise_coefficient_db,
        "total_noise_power_dbm": core.compute_total_noise_power_dbm(optimizer.noise_coefficient_db),
        "best_positions_xy": best.positions.astype(float).tolist(),
    }
    best_metrics_path.write_text(json.dumps(best_metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    predictor = core.RadioMapPredictor(args.model_path, args.network_type, args.device)
    height_map = core.load_height_map(args.height_map)
    optimizer = ParticleSwarmOptimizer(
        predictor=predictor,
        height_map=height_map,
        k_max=args.k_max,
        coverage_target=args.coverage_target,
        capacity_target=args.spectral_efficiency_target,
        max_evals=args.max_evals,
        d_min=args.d_min,
        repair_max_tries=args.repair_max_tries,
        w1=args.w1,
        w2=args.w2,
        coverage_threshold_db=args.coverage_threshold_db,
        noise_coefficient_db=args.noise_coefficient_db,
        swarm_size=args.swarm_size,
        inertia=args.inertia,
        c1=args.c1,
        c2=args.c2,
        velocity_clamp=args.velocity_clamp,
    )
    best = optimizer.optimize()
    write_outputs(Path(args.output_dir), optimizer, predictor, height_map, best, args.k_max)
    print(f"best_score: {best.metrics.score:.6f}")
    print(f"coverage: {best.metrics.coverage:.6f}")
    print(f"spectral_efficiency: {best.metrics.spectral_efficiency:.6f}")
    print(f"channel_capacity_mbps: {best.metrics.channel_capacity:.6f}")
    print(f"redundancy_rate: {best.metrics.redundancy_rate:.6f}")
    print(f"best_positions_xy: {np.array2string(best.positions, precision=3)}")
    print(f"total_runtime_sec: {time.perf_counter() - start_time:.6f}")


if __name__ == "__main__":
    main()
