"""
命令示例:
python run_sa.py \
  --height-map data/b.png \
  --k-max 6 \
  --coverage-target 0.9 \
  --spectral-efficiency-target 3.5 \
  --max-evals 200 \
  --device mps \
  --model-path /Users/epiphanyer/Desktop/coding/paper_experiment/Heuristic/model/RMNet.pt \
  --network-type rmnet \
  --output-dir outputs/smoke_test_sa

算法核心思路:
模拟退火(SA)从一个初始站点集合出发，每步随机选一个站点做局部高斯扰动或重采样，
然后用代理模型重新计算 coverage、spectral_efficiency、channel_capacity_mbps 和 score。若新解更优则直接接受；
若更差，也允许以与当前温度相关的概率接受，从而跳出局部最优。随着温度下降，
搜索会从“探索”逐渐过渡到“收敛”。

参数说明:
--height-map: 输入灰度高度图，0 视为可统计的地面 RoI。
--k-max: 需要同时部署的站点数量。
--coverage-target: 目标覆盖率，用于 score 中的惩罚项。
--spectral-efficiency-target: 目标平均频谱效率，用于 score 中的惩罚项。
--max-evals: 最大代理模型评估次数。
--output-dir: 输出目录。
--initial-temp: 模拟退火初始温度，越大越容易接受差解。
--cooling-rate: 温度衰减率，越接近 1 降温越慢。
--gaussian-sigma: 高斯扰动标准差，单位像素。
--d-min: 站点之间的最小距离约束，单位像素。
--repair-max-tries: 修复非法站点位置的最大尝试次数。
--w1: 覆盖率不足惩罚权重。
--w2: 频谱效率不足惩罚权重。
--coverage-threshold-db: 最强单站接收功率覆盖阈值，单位 dBm。
--noise-coefficient-db: 接收机噪声系数，单位 dB。总噪声功率由热噪声密度、带宽和该系数在代码中计算。
--model-path: 代理模型权重路径。
--network-type: 代理模型类型。
--device: 推理设备，例如 cpu/cuda/mps。

脚本逻辑说明:
这个脚本是仓库里的公共 SA 实现和指标定义入口。每次给定一个站点布局后，会先生成每个站点的预测图；
每个像素选接收功率最大的站点作为服务站点，用它判定覆盖；其余站点的接收功率视为干扰，
再结合由热噪声密度、带宽和噪声系数推导出的总噪声功率计算 `SINR = S / (I + N)`，
并在整个 RoI 上同时得到平均频谱效率 `log2(1 + SINR)` 和平均信道容量
`CHANNEL_BANDWIDTH_HZ * log2(1 + SINR) / 1e6`，单位 Mbps。
SA 通过“随机扰动 + 温度控制接受差解”的方式在连续空间中搜索更优布局。

"""

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

import eval_radiomap_local
from test.preview_utils import (
    save_height_map_preview,
    save_pathgain_preview,
    save_site_map_preview,
)

_PREDICTOR_CACHE: Dict[Tuple[str, str, str], "RadioMapPredictor"] = {}
TX_POWER_DBM = 23.0
PATHLOSS_THRESHOLD_DB = 140.0
DEFAULT_COVERAGE_THRESHOLD_DB = TX_POWER_DBM - PATHLOSS_THRESHOLD_DB
THERMAL_NOISE_DENSITY_DBM_PER_HZ = -174.0
CHANNEL_BANDWIDTH_HZ = 20_000_000.0
BITS_PER_MEGABIT = 1_000_000.0
NOISE_COEFFICIENT_DB = 10.0
ROI_COUNT_THRESHOLDS = [1, 2]
SCORE_COVERAGE_FLOOR_OFFSET = 0.08
SCORE_COVERAGE_FLOOR_MIN = 0.55
SCORE_COVERAGE_FLOOR_MAX = 0.75
SCORE_MARGIN_SCALE_DB = 6.0
SCORE_COVERAGE_WEIGHT = 1.2
SCORE_MARGIN_WEIGHT = 0.3
SCORE_SE_TARGET_CLIP = 1.0
SCORE_COVERAGE_PENALTY_WEIGHT = 12.0
SCORE_SE_PENALTY_WEIGHT = 0.5


@dataclass
class EvalResult:
    coverage: float
    spectral_efficiency: float
    channel_capacity: float
    redundancy_rate: float
    score: float
    base_score: float
    penalty: float

    @property
    def capacity(self) -> float:
        return self.spectral_efficiency


@dataclass
class SAState:
    positions: np.ndarray
    metrics: EvalResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous-space base-station deployment via simulated annealing")
    parser.add_argument("--height-map", required=True, type=str, help="Path to grayscale height map")
    parser.add_argument("--k-max", required=True, type=int, help="Maximum number of sites")
    parser.add_argument("--coverage-target", required=True, type=float, help="Target coverage ratio")
    parser.add_argument(
        "--spectral-efficiency-target",
        required=True,
        type=float,
        help="Target average spectral efficiency",
    )
    parser.add_argument("--max-evals", required=True, type=int, help="Maximum number of radiomap evaluations")
    parser.add_argument("--output-dir", required=True, type=str, help="Output directory")

    parser.add_argument("--initial-temp", default=1.0, type=float, help="Initial SA temperature")
    parser.add_argument("--cooling-rate", default=0.995, type=float, help="Geometric cooling rate")
    parser.add_argument("--gaussian-sigma", default=6.0, type=float, help="Std of Gaussian perturbation in pixels")
    parser.add_argument("--d-min", default=12.0, type=float, help="Minimum pairwise site distance in pixels")
    parser.add_argument("--repair-max-tries", default=100, type=int, help="Maximum attempts when repairing one site")
    parser.add_argument("--w1", default=1.0, type=float, help="Penalty weight for coverage shortfall")
    parser.add_argument("--w2", default=1.0, type=float, help="Penalty weight for spectral-efficiency shortfall")
    parser.add_argument(
        "--coverage-threshold-db",
        default=DEFAULT_COVERAGE_THRESHOLD_DB,
        type=float,
        help="Coverage threshold on strongest per-pixel received power in dBm",
    )
    parser.add_argument(
        "--noise-coefficient-db",
        default=NOISE_COEFFICIENT_DB,
        type=float,
        help="Receiver noise coefficient in dB; total noise power is computed in code",
    )
    parser.add_argument("--model-path", required=True, type=str, help="Checkpoint path")
    parser.add_argument("--network-type", required=True, type=str, help="Network type")
    parser.add_argument("--device", default="mps", type=str, help="Torch device, e.g. cpu/cuda/mps")
    return parser.parse_args()


def compute_total_noise_power_dbm(noise_coefficient_db: float) -> float:
    return (
        THERMAL_NOISE_DENSITY_DBM_PER_HZ
        + 10.0 * math.log10(CHANNEL_BANDWIDTH_HZ)
        + noise_coefficient_db
    )


def compute_total_noise_power_mw(noise_coefficient_db: float) -> float:
    return float(np.power(10.0, compute_total_noise_power_dbm(noise_coefficient_db) / 10.0))


def compute_score_coverage_floor(coverage_target: float) -> float:
    return float(np.clip(
        coverage_target + SCORE_COVERAGE_FLOOR_OFFSET,
        SCORE_COVERAGE_FLOOR_MIN,
        SCORE_COVERAGE_FLOOR_MAX,
    ))


def load_height_map(path: str) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.uint8)


def save_history_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class RadioMapPredictor:
    def __init__(
        self,
        model_path: str,
        network_type: str,
        device_name: str,
    ) -> None:
        self.device = torch.device(device_name)
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        self.model_path = str(path)
        self.network_type = network_type
        self.model = eval_radiomap_local.load_model(self.model_path, self.network_type, self.device)

    def predict_site_maps(self, height_map: np.ndarray, site_positions: np.ndarray) -> np.ndarray:
        if height_map.ndim != 2:
            raise ValueError(f"height_map must be HxW grayscale, got shape {height_map.shape}")
        if site_positions.ndim != 2 or site_positions.shape[1] != 2:
            raise ValueError("site_positions must have shape [K, 2]")

        height, width = height_map.shape
        rounded = np.rint(site_positions).astype(int)
        predictions: List[np.ndarray] = []

        for x, y in rounded:
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Site out of bounds after rounding: ({x}, {y}) vs ({width}, {height})")

            site_map = np.zeros_like(height_map, dtype=np.uint8)
            site_map[y, x] = eval_radiomap_local.encode_tx_gray_value(int(height_map[y, x]))
            inputs = np.stack([height_map, site_map], axis=2)
            tensor = eval_radiomap_local._numpy_image_to_tensor(inputs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = torch.clamp(self.model(tensor), 0, 1).detach().cpu().squeeze().numpy().astype(np.float32)
            # Direct mapping: model output in [0, 1] is mapped to the dataset's
            # negative pathgain-like dB range [DB_MIN, DB_MAX].
            predictions.append(eval_radiomap_local._to_db_scale(pred).astype(np.float32))

        if not predictions:
            raise ValueError("site_positions must contain at least one site")
        return np.stack(predictions, axis=0).astype(np.float32)

    def predict(self, height_map: np.ndarray, site_positions: np.ndarray) -> np.ndarray:
        site_maps_db = self.predict_site_maps(height_map, site_positions)
        # Build the per-pixel serving-link map from the strongest site response.
        return np.max(site_maps_db, axis=0).astype(np.float32)


def get_predictor(
    model_path: str,
    network_type: str,
    device: str,
) -> RadioMapPredictor:
    key = (model_path, network_type, device)
    predictor = _PREDICTOR_CACHE.get(key)
    if predictor is None:
        predictor = RadioMapPredictor(
            model_path=model_path,
            network_type=network_type,
            device_name=device,
        )
        _PREDICTOR_CACHE[key] = predictor
    return predictor


def predict(
    height_map: np.ndarray,
    site_positions: np.ndarray,
    model_path: str,
    network_type: str,
    device: str,
) -> np.ndarray:
    predictor = get_predictor(
        model_path=model_path,
        network_type=network_type,
        device=device,
    )
    return predictor.predict(height_map=height_map, site_positions=site_positions)


def build_placement_mask(height_map: np.ndarray) -> np.ndarray:
    return np.asarray(height_map > 0, dtype=bool)


def build_roi_mask(height_map: np.ndarray) -> np.ndarray:
    mask = height_map == 0
    if not mask.any():
        mask = np.ones_like(height_map, dtype=bool)
    return mask


def random_valid_point(
    rng: np.random.Generator,
    valid_mask: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    ys, xs = np.where(valid_mask)
    idx = int(rng.integers(0, len(xs)))
    x = float(xs[idx]) + float(rng.uniform(-0.49, 0.49))
    y = float(ys[idx]) + float(rng.uniform(-0.49, 0.49))
    return np.array([np.clip(x, 0.0, width - 1.0), np.clip(y, 0.0, height - 1.0)], dtype=np.float32)


def nearest_valid_point(point: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(valid_mask)
    if len(xs) == 0:
        raise ValueError("No valid pixels available")
    dx = xs.astype(np.float32) - point[0]
    dy = ys.astype(np.float32) - point[1]
    idx = int(np.argmin(dx * dx + dy * dy))
    return np.array([float(xs[idx]), float(ys[idx])], dtype=np.float32)


def pairwise_min_distance(positions: np.ndarray, idx: int) -> float:
    if len(positions) <= 1:
        return float("inf")
    deltas = positions - positions[idx]
    dist = np.sqrt(np.sum(deltas * deltas, axis=1))
    dist[idx] = np.inf
    return float(np.min(dist))


def is_position_valid(positions: np.ndarray, idx: int, valid_mask: np.ndarray, d_min: float) -> bool:
    x, y = np.rint(positions[idx]).astype(int)
    height, width = valid_mask.shape
    if not (0 <= x < width and 0 <= y < height):
        return False
    if not bool(valid_mask[y, x]):
        return False
    return pairwise_min_distance(positions, idx) >= d_min


def repair_position(
    positions: np.ndarray,
    idx: int,
    valid_mask: np.ndarray,
    d_min: float,
    rng: np.random.Generator,
    repair_max_tries: int,
) -> None:
    height, width = valid_mask.shape
    positions[idx, 0] = float(np.clip(positions[idx, 0], 0.0, width - 1.0))
    positions[idx, 1] = float(np.clip(positions[idx, 1], 0.0, height - 1.0))

    rounded = np.rint(positions[idx]).astype(int)
    if not valid_mask[rounded[1], rounded[0]]:
        positions[idx] = nearest_valid_point(positions[idx], valid_mask)

    if is_position_valid(positions, idx, valid_mask, d_min):
        return

    anchor = positions[idx].copy()
    for _ in range(repair_max_tries):
        candidate = nearest_valid_point(anchor, valid_mask)
        candidate += rng.normal(0.0, 0.75, size=2).astype(np.float32)
        candidate[0] = float(np.clip(candidate[0], 0.0, width - 1.0))
        candidate[1] = float(np.clip(candidate[1], 0.0, height - 1.0))
        candidate_round = np.rint(candidate).astype(int)
        if not valid_mask[candidate_round[1], candidate_round[0]]:
            candidate = nearest_valid_point(candidate, valid_mask)
        positions[idx] = candidate
        if is_position_valid(positions, idx, valid_mask, d_min):
            return

    for _ in range(repair_max_tries):
        positions[idx] = random_valid_point(rng, valid_mask, width, height)
        if is_position_valid(positions, idx, valid_mask, d_min):
            return

    raise RuntimeError("Failed to repair site position under current d_min constraint")


def initialize_positions(
    k_max: int,
    valid_mask: np.ndarray,
    d_min: float,
    rng: np.random.Generator,
    repair_max_tries: int,
) -> np.ndarray:
    height, width = valid_mask.shape
    positions = np.zeros((k_max, 2), dtype=np.float32)
    for idx in range(k_max):
        positions[idx] = random_valid_point(rng, valid_mask, width, height)
        repair_position(positions, idx, valid_mask, d_min, rng, repair_max_tries)
    return positions


def evaluate(
    pathgain_db: np.ndarray,
    summed_rx_power_mw: np.ndarray,
    roi_mask: np.ndarray,
    coverage_target: float,
    spectral_efficiency_target: float,
    w1: float,
    w2: float,
    coverage_threshold_db: float,
    noise_coefficient_db: float,
    covered_site_counts: np.ndarray | None = None,
) -> EvalResult:
    if pathgain_db.shape != roi_mask.shape:
        raise ValueError(f"pathgain_db shape {pathgain_db.shape} does not match roi_mask shape {roi_mask.shape}")
    if summed_rx_power_mw.shape != roi_mask.shape:
        raise ValueError(
            f"summed_rx_power_mw shape {summed_rx_power_mw.shape} does not match roi_mask shape {roi_mask.shape}"
        )
    if covered_site_counts is not None and covered_site_counts.shape != roi_mask.shape:
        raise ValueError(
            f"covered_site_counts shape {covered_site_counts.shape} does not match roi_mask shape {roi_mask.shape}"
        )
    strongest_pathgain_db = pathgain_db[roi_mask].astype(np.float64)
    total_rx_power_mw = summed_rx_power_mw[roi_mask].astype(np.float64)
    if strongest_pathgain_db.size == 0:
        raise ValueError("roi_mask selects no pixels for evaluation")

    strongest_rx_power_dbm = TX_POWER_DBM + strongest_pathgain_db
    strongest_rx_power_mw = np.power(10.0, strongest_rx_power_dbm / 10.0)
    coverage = float(np.mean(strongest_rx_power_dbm >= coverage_threshold_db))

    noise_power_mw = compute_total_noise_power_mw(noise_coefficient_db)
    interference_power_mw = np.maximum(total_rx_power_mw - strongest_rx_power_mw, 0.0)
    sinr_linear = strongest_rx_power_mw / np.maximum(interference_power_mw + noise_power_mw, 1e-30)
    spectral_efficiency = float(np.mean(np.log2(1.0 + sinr_linear)))
    channel_capacity = float(CHANNEL_BANDWIDTH_HZ * spectral_efficiency / BITS_PER_MEGABIT)
    redundancy_rate = 0.0
    if covered_site_counts is not None:
        roi_covered_site_counts = covered_site_counts[roi_mask].astype(np.int32)
        covered_pixels = roi_covered_site_counts >= ROI_COUNT_THRESHOLDS[0]
        covered_pixel_count = int(np.count_nonzero(covered_pixels))
        if covered_pixel_count > 0:
            redundant_pixel_count = int(np.count_nonzero(roi_covered_site_counts >= ROI_COUNT_THRESHOLDS[1]))
            redundancy_rate = redundant_pixel_count / covered_pixel_count

    # score = w1 * coverage + w2 * spectral_efficiency - penalty
    coverage_floor = compute_score_coverage_floor(coverage_target)
    rss_margin = float(
        np.mean(
            np.tanh((strongest_rx_power_dbm - coverage_threshold_db) / max(SCORE_MARGIN_SCALE_DB, 1e-6))
        )
    )
    coverage_shortfall = max(0.0, coverage_floor - coverage)
    se_target = float(np.clip(float(spectral_efficiency_target), 1e-6, SCORE_SE_TARGET_CLIP))
    se_shortfall = max(0.0, se_target - spectral_efficiency)
    coverage_penalty_term = SCORE_COVERAGE_PENALTY_WEIGHT * (coverage_shortfall ** 2)
    se_penalty_term = SCORE_SE_PENALTY_WEIGHT * (se_shortfall ** 2)
    penalty = coverage_penalty_term + se_penalty_term
    base_score = (
        SCORE_COVERAGE_WEIGHT * coverage
        + SCORE_MARGIN_WEIGHT * rss_margin
    )
    score = base_score - penalty
    return EvalResult(
        coverage=coverage,
        spectral_efficiency=spectral_efficiency,
        channel_capacity=channel_capacity,
        redundancy_rate=redundancy_rate,
        score=score,
        base_score=base_score,
        penalty=penalty,
    )


class SimulatedAnnealingOptimizer:
    def __init__(
        self,
        predictor: RadioMapPredictor,
        height_map: np.ndarray,
        k_max: int,
        coverage_target: float,
        spectral_efficiency_target: float,
        max_evals: int,
        initial_temp: float,
        cooling_rate: float,
        gaussian_sigma: float,
        d_min: float,
        repair_max_tries: int,
        w1: float,
        w2: float,
        coverage_threshold_db: float,
        noise_coefficient_db: float,
    ) -> None:
        self.predictor = predictor
        self.height_map = height_map
        self.height, self.width = height_map.shape
        self.k_max = k_max
        self.coverage_target = coverage_target
        self.spectral_efficiency_target = spectral_efficiency_target
        self.max_evals = max_evals
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.gaussian_sigma = gaussian_sigma
        self.d_min = d_min
        self.repair_max_tries = repair_max_tries
        self.w1 = w1
        self.w2 = w2
        self.coverage_threshold_db = coverage_threshold_db
        self.noise_coefficient_db = noise_coefficient_db
        self.placement_mask = build_placement_mask(height_map)
        self.roi_mask = build_roi_mask(height_map)
        self.rng = np.random.default_rng()
        self.eval_count = 0
        self.history: List[Dict[str, float]] = []

    def _evaluate_positions(self, positions: np.ndarray) -> EvalResult:
        self.eval_count += 1
        site_pathgain_db = self.predictor.predict_site_maps(self.height_map, positions)
        pathgain_db = np.max(site_pathgain_db, axis=0).astype(np.float32)
        rx_power_dbm = TX_POWER_DBM + site_pathgain_db.astype(np.float64)
        summed_rx_power_mw = np.sum(np.power(10.0, rx_power_dbm / 10.0), axis=0).astype(np.float32)
        covered_site_counts = np.count_nonzero(rx_power_dbm >= self.coverage_threshold_db, axis=0).astype(np.int16)
        return evaluate(
            pathgain_db=pathgain_db,
            summed_rx_power_mw=summed_rx_power_mw,
            roi_mask=self.roi_mask,
            coverage_target=self.coverage_target,
            spectral_efficiency_target=self.spectral_efficiency_target,
            w1=self.w1,
            w2=self.w2,
            coverage_threshold_db=self.coverage_threshold_db,
            noise_coefficient_db=self.noise_coefficient_db,
            covered_site_counts=covered_site_counts,
        )

    def _propose_neighbor(self, current: np.ndarray) -> Tuple[np.ndarray, int, str]:
        proposal = current.copy()
        idx = int(self.rng.integers(0, self.k_max))

        if float(self.rng.random()) < 0.8:
            proposal[idx, 0] += float(self.rng.normal(0.0, self.gaussian_sigma))
            proposal[idx, 1] += float(self.rng.normal(0.0, self.gaussian_sigma))
            move_type = "gaussian"
        else:
            proposal[idx] = random_valid_point(self.rng, self.placement_mask, self.width, self.height)
            move_type = "resample"

        repair_position(
            proposal,
            idx,
            self.placement_mask,
            self.d_min,
            self.rng,
            self.repair_max_tries,
        )
        return proposal, idx, move_type

    def optimize(self) -> SAState:
        current_positions = initialize_positions(
            k_max=self.k_max,
            valid_mask=self.placement_mask,
            d_min=self.d_min,
            rng=self.rng,
            repair_max_tries=self.repair_max_tries,
        )
        current_metrics = self._evaluate_positions(current_positions)
        current = SAState(positions=current_positions.copy(), metrics=current_metrics)
        best = SAState(positions=current_positions.copy(), metrics=current_metrics)
        temperature = self.initial_temp

        self.history.append(
            self._history_row(
                step=0,
                temperature=temperature,
                move_index=-1,
                move_type="init",
                accepted=1,
                current=current.metrics,
                best=best.metrics,
            )
        )

        step = 0
        while self.eval_count < self.max_evals:
            step += 1
            proposal_positions, idx, move_type = self._propose_neighbor(current.positions)
            proposal_metrics = self._evaluate_positions(proposal_positions)

            delta = proposal_metrics.score - current.metrics.score
            accepted = False
            if delta > 0:
                accepted = True
            else:
                prob = math.exp(delta / max(temperature, 1e-12))
                if float(self.rng.random()) < prob:
                    accepted = True

            if accepted:
                current = SAState(positions=proposal_positions.copy(), metrics=proposal_metrics)
                if current.metrics.score > best.metrics.score:
                    best = SAState(positions=current.positions.copy(), metrics=current.metrics)

            temperature *= self.cooling_rate
            self.history.append(
                self._history_row(
                    step=step,
                    temperature=temperature,
                    move_index=idx,
                    move_type=move_type,
                    accepted=int(accepted),
                    current=current.metrics,
                    best=best.metrics,
                )
            )

        return best

    def _history_row(
        self,
        step: int,
        temperature: float,
        move_index: int,
        move_type: str,
        accepted: int,
        current: EvalResult,
        best: EvalResult,
    ) -> Dict[str, float]:
        return {
            "step": step,
            "eval_count": self.eval_count,
            "temperature": float(temperature),
            "move_index": move_index,
            "move_type": move_type,
            "accepted": accepted,
            "current_score": current.score,
            "current_coverage": current.coverage,
            "current_spectral_efficiency": current.spectral_efficiency,
            "current_channel_capacity_mbps": current.channel_capacity,
            "current_redundancy_rate": current.redundancy_rate,
            "current_penalty": current.penalty,
            "best_score": best.score,
            "best_coverage": best.coverage,
            "best_spectral_efficiency": best.spectral_efficiency,
            "best_channel_capacity_mbps": best.channel_capacity,
            "best_redundancy_rate": best.redundancy_rate,
            "best_penalty": best.penalty,
        }


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    height_map = load_height_map(args.height_map)
    predictor = RadioMapPredictor(
        model_path=args.model_path,
        network_type=args.network_type,
        device_name=args.device,
    )

    optimizer = SimulatedAnnealingOptimizer(
        predictor=predictor,
        height_map=height_map,
        k_max=args.k_max,
        coverage_target=args.coverage_target,
        spectral_efficiency_target=args.spectral_efficiency_target,
        max_evals=args.max_evals,
        initial_temp=args.initial_temp,
        cooling_rate=args.cooling_rate,
        gaussian_sigma=args.gaussian_sigma,
        d_min=args.d_min,
        repair_max_tries=args.repair_max_tries,
        w1=args.w1,
        w2=args.w2,
        coverage_threshold_db=args.coverage_threshold_db,
        noise_coefficient_db=args.noise_coefficient_db,
    )
    best = optimizer.optimize()

    best_layout_path = output_dir / "best_layout.npy"
    history_path = output_dir / "history.csv"
    best_metrics_path = output_dir / "best_metrics.json"
    best_prediction_path = output_dir / "best_prediction.png"
    height_map_preview_path = output_dir / "height_map.png"
    site_map_preview_path = output_dir / "site_map.png"

    np.save(best_layout_path, best.positions.astype(np.float32))
    save_history_csv(history_path, optimizer.history)
    best_pathloss_db = predictor.predict(height_map, best.positions)
    save_pathgain_preview(
        best_pathloss_db,
        best.positions,
        best_prediction_path,
    )
    save_height_map_preview(height_map, height_map_preview_path)
    save_site_map_preview(height_map, best.positions, site_map_preview_path)

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
        "height_map": str(Path(args.height_map).resolve()),
        "k_max": args.k_max,
        "spectral_efficiency_target": args.spectral_efficiency_target,
        "noise_coefficient_db": args.noise_coefficient_db,
        "total_noise_power_dbm": compute_total_noise_power_dbm(args.noise_coefficient_db),
        "best_positions_xy": best.positions.astype(float).tolist(),
        "best_positions_xy_rounded": np.rint(best.positions).astype(int).tolist(),
    }
    best_metrics_path.write_text(json.dumps(best_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"best_layout_npy: {best_layout_path.resolve()}")
    print(f"best_prediction_png: {best_prediction_path.resolve()}")
    print(f"height_map_png: {height_map_preview_path.resolve()}")
    print(f"site_map_png: {site_map_preview_path.resolve()}")
    print(f"history_csv: {history_path.resolve()}")
    print(f"best_metrics_json: {best_metrics_path.resolve()}")
    print(f"best_score: {best.metrics.score:.6f}")
    print(f"coverage: {best.metrics.coverage:.6f}")
    print(f"spectral_efficiency: {best.metrics.spectral_efficiency:.6f}")
    print(f"channel_capacity_mbps: {best.metrics.channel_capacity:.6f}")
    print(f"redundancy_rate: {best.metrics.redundancy_rate:.6f}")
    print(f"eval_count: {optimizer.eval_count}")
    print(f"checkpoint: {predictor.model_path}")
    print(f"network_type: {predictor.network_type}")
    print(f"best_positions_xy: {np.array2string(best.positions, precision=3)}")
    print(f"total_runtime_sec: {time.perf_counter() - start_time:.6f}")


if __name__ == "__main__":
    main()
