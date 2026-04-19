"""
命令示例:
python run_ga.py \
  --height-map data/b.png \
  --k-max 12 \
  --coverage-target 0.9 \
  --spectral-efficiency-target 3.5 \
  --max-evals 200 \
  --device mps \
  --model-path /Users/epiphanyer/Desktop/coding/paper_experiment/Heuristic/model/RMNet.pt \
  --network-type rmnet \
  --output-dir outputs/smoke_test_ga \
  --population-size 24 \
  --elite-size 4 \
  --tournament-size 3 \
  --mutation-rate 0.3 \
  --gaussian-sigma 4

算法核心思路:
遗传算法(GA)把一组站点布局看作一个个体。每一代先保留一部分高分精英，
再通过锦标赛选择父代，执行交叉(crossover)和变异(mutation)生成子代。
子代经过约束修复后再用代理模型打分。算法通过“优者繁殖”的方式，
逐代把种群推向更高 coverage 和 spectral_efficiency 的解。

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
--population-size: 种群大小。
--elite-size: 每代保留的精英个体数。
--tournament-size: 锦标赛选择规模。
--gaussian-sigma: 变异时高斯扰动的标准差，单位像素。
--mutation-rate: 子代发生变异的概率。

脚本逻辑说明:
GA 维护一组候选布局。每一代先按 score 保留精英，再从当前种群中做锦标赛选择，
用交叉与变异生成新布局。每次评估布局时，覆盖率按“每个像素最强单站接收功率是否过阈值”
计算；同时输出平均频谱效率 `log2(1 + SINR)` 和平均信道容量
`CHANNEL_BANDWIDTH_HZ * log2(1 + SINR)`。这样 GA 会持续把种群推向覆盖率和频谱效率都更优的解。
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

import run_sa as core


class GeneticAlgorithmOptimizer:
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
        population_size: int,
        elite_size: int,
        tournament_size: int,
        mutation_rate: float,
        gaussian_sigma: float,
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
        self.population_size = population_size
        self.elite_size = min(elite_size, population_size)
        self.tournament_size = max(2, tournament_size)
        self.mutation_rate = mutation_rate
        self.gaussian_sigma = gaussian_sigma
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

    def _init_population(self) -> List[core.SAState]:
        population: List[core.SAState] = []
        while len(population) < self.population_size and self.eval_count < self.max_evals:
            positions = core.initialize_positions(
                k_max=self.k_max,
                valid_mask=self.placement_mask,
                d_min=self.d_min,
                rng=self.rng,
                repair_max_tries=self.repair_max_tries,
            )
            population.append(core.SAState(positions=positions, metrics=self._evaluate_positions(positions)))
        return population

    def _tournament_select(self, population: List[core.SAState]) -> core.SAState:
        idxs = self.rng.integers(0, len(population), size=self.tournament_size)
        candidates = [population[int(i)] for i in idxs]
        return max(candidates, key=lambda item: item.metrics.score)

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        child = a.copy()
        mask = self.rng.random(self.k_max) < 0.5
        child[mask] = b[mask]
        return child

    def _mutate(self, positions: np.ndarray) -> np.ndarray:
        out = positions.copy()
        for idx in range(self.k_max):
            if float(self.rng.random()) < self.mutation_rate:
                out[idx, 0] += float(self.rng.normal(0.0, self.gaussian_sigma))
                out[idx, 1] += float(self.rng.normal(0.0, self.gaussian_sigma))
                core.repair_position(
                    out,
                    idx,
                    self.placement_mask,
                    self.d_min,
                    self.rng,
                    self.repair_max_tries,
                )
        return out

    def optimize(self) -> core.SAState:
        population = self._init_population()
        if not population:
            raise RuntimeError("Failed to initialize population under max_evals budget")

        generation = 0
        best = max(population, key=lambda item: item.metrics.score)
        self.history.append(self._history_row(generation, best, best))

        while self.eval_count < self.max_evals:
            generation += 1
            population.sort(key=lambda item: item.metrics.score, reverse=True)
            next_population = population[: self.elite_size]

            while len(next_population) < self.population_size and self.eval_count < self.max_evals:
                parent_a = self._tournament_select(population)
                parent_b = self._tournament_select(population)
                child_positions = self._crossover(parent_a.positions, parent_b.positions)
                child_positions = self._mutate(child_positions)
                child_metrics = self._evaluate_positions(child_positions)
                next_population.append(core.SAState(positions=child_positions.copy(), metrics=child_metrics))

            population = next_population
            current_best = max(population, key=lambda item: item.metrics.score)
            if current_best.metrics.score > best.metrics.score:
                best = core.SAState(positions=current_best.positions.copy(), metrics=current_best.metrics)
            self.history.append(self._history_row(generation, current_best, best))

        return best

    def _history_row(self, generation: int, current_best: core.SAState, best: core.SAState) -> Dict[str, float]:
        return {
            "generation": generation,
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
            "best_penalty": best.metrics.penalty,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Base-station deployment via genetic algorithm")
    parser.add_argument("--height-map", required=True, type=str, help="Path to grayscale height map")
    parser.add_argument("--k-max", required=True, type=int, help="Number of sites in each layout")
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
    parser.add_argument("--gaussian-sigma", default=6.0, type=float, help="Std of Gaussian mutation in pixels")
    parser.add_argument("--population-size", default=24, type=int, help="Population size per generation")
    parser.add_argument("--elite-size", default=4, type=int, help="Number of elites preserved each generation")
    parser.add_argument("--tournament-size", default=3, type=int, help="Tournament size for parent selection")
    parser.add_argument("--mutation-rate", default=0.3, type=float, help="Probability of mutating each site")
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
        "best_positions_xy_rounded": np.rint(best.positions).astype(int).tolist(),
    }
    best_metrics_path.write_text(json.dumps(best_metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    predictor = core.RadioMapPredictor(args.model_path, args.network_type, args.device)
    height_map = core.load_height_map(args.height_map)
    optimizer = GeneticAlgorithmOptimizer(
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
        population_size=args.population_size,
        elite_size=args.elite_size,
        tournament_size=args.tournament_size,
        mutation_rate=args.mutation_rate,
        gaussian_sigma=args.gaussian_sigma,
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
