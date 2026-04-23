"""注释
命令:
- `python -m Autobs.train_ppo \
   --city_map_path /abs/path/to/maps \
   --dataset_limit 128 \
   -r score`

参数含义:
- `--city_map_path`: 可以是单个地图文件、包含多张地图的目录，或逗号分隔的多个路径。
- `--dataset_limit`: 限制参与训练的地图数量。
- `--dataset_offset`: 设定数据集子集起始偏移。
- `--dataset_stride`: 设定数据集抽样步长。
- `-r, --reward_type`: 可选 `coverage`、`spectral_efficiency`、`score`、`score_v2`，其中 `capacity`
  兼容映射到 `spectral_efficiency`。
- 本文件只保留 PPO 训练环境真正使用的地图读取、数据集子集选择、动作掩码、与 `Heuristic/run_sa.py` 对齐的指标计算。
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from Autobs.paths import CONFIG_PATH
from Autobs.pmnet_adapter import infer_surrogate


def _load_env_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return deepcopy(config.get("env", {}))


ENV_CONFIG = _load_env_config()
ACTION_SPACE_SIZE = int(ENV_CONFIG.get("action_space_size", 32))
MAP_SIZE = int(ENV_CONFIG.get("map_size", 256))
UPSAMPLING_FACTOR = MAP_SIZE // ACTION_SPACE_SIZE
NON_BUILDING_PIXEL = float(ENV_CONFIG.get("non_building_pixel", 1.0))
ROI_THRESHOLD = float(ENV_CONFIG.get("roi_threshold", 0.01))
ROI_IS_BLACK = bool(ENV_CONFIG.get("roi_is_black", True))
TX_POWER_DBM = float(ENV_CONFIG.get("tx_power_dbm", 23.0))
PATHLOSS_THRESHOLD_DB = float(ENV_CONFIG.get("pathloss_threshold_db", 140.0))
DB_MIN = float(ENV_CONFIG.get("pathgain_db_min", -162.0))
DB_MAX = float(ENV_CONFIG.get("pathgain_db_max", -75.0))
CHANNEL_BANDWIDTH_HZ = float(ENV_CONFIG.get("channel_bandwidth_hz", 20_000_000.0))
BITS_PER_MEGABIT = 1_000_000.0
DEFAULT_COVERAGE_THRESHOLD_DB = TX_POWER_DBM - PATHLOSS_THRESHOLD_DB
DEFAULT_COVERAGE_TARGET = float(ENV_CONFIG.get("coverage_target", 0.9))
DEFAULT_SPECTRAL_EFFICIENCY_TARGET = float(ENV_CONFIG.get("spectral_efficiency_target", 3.5))
DEFAULT_W1 = float(ENV_CONFIG.get("w1", 1.0))
DEFAULT_W2 = float(ENV_CONFIG.get("w2", 1.0))
THERMAL_NOISE_DENSITY_DBM_PER_HZ = -174.0
DEFAULT_NOISE_COEFFICIENT_DB = float(ENV_CONFIG.get("noise_coefficient_db", 10.0))
DEFAULT_SCORE_V2_COVERAGE_FLOOR_OFFSET = float(ENV_CONFIG.get("score_v2_coverage_floor_offset", 0.08))
DEFAULT_SCORE_V2_COVERAGE_FLOOR_MIN = float(ENV_CONFIG.get("score_v2_coverage_floor_min", 0.55))
DEFAULT_SCORE_V2_COVERAGE_FLOOR_MAX = float(ENV_CONFIG.get("score_v2_coverage_floor_max", 0.75))
DEFAULT_SCORE_V2_GATE_WIDTH = float(ENV_CONFIG.get("score_v2_gate_width", 0.02))
DEFAULT_SCORE_V2_SE_FLOOR = float(ENV_CONFIG.get("score_v2_se_floor", 0.35))
DEFAULT_SCORE_V2_MARGIN_SCALE_DB = float(ENV_CONFIG.get("score_v2_margin_scale_db", 6.0))
DEFAULT_SCORE_V2_COVERAGE_WEIGHT = float(ENV_CONFIG.get("score_v2_coverage_weight", 0.5))
DEFAULT_SCORE_V2_MARGIN_WEIGHT = float(ENV_CONFIG.get("score_v2_margin_weight", 1.0))
DEFAULT_SCORE_V2_SE_WEIGHT = float(ENV_CONFIG.get("score_v2_se_weight", 2.0))
DEFAULT_SCORE_V2_COVERAGE_PENALTY_WEIGHT = float(ENV_CONFIG.get("score_v2_coverage_penalty_weight", 8.0))
DEFAULT_SCORE_V2_SE_PENALTY_WEIGHT = float(ENV_CONFIG.get("score_v2_se_penalty_weight", 1.5))


def dbm_to_mw(dbm):
    return 10 ** (np.asarray(dbm, dtype=float) / 10.0)


def mw_to_dbm(mw):
    mw = np.asarray(mw, dtype=float)
    out = np.full_like(mw, -np.inf, dtype=float)
    mask = mw > 0
    out[mask] = 10.0 * np.log10(mw[mask])
    return out


def load_map_normalized(file_path: str | Path) -> np.ndarray:
    image = Image.open(file_path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def _slice_dataset_paths(
    paths: list[str],
    dataset_limit: int | None = None,
    dataset_offset: int = 0,
    dataset_stride: int = 1,
) -> list[str]:
    if dataset_stride <= 0:
        raise ValueError("dataset_stride must be a positive integer")
    if dataset_offset < 0:
        raise ValueError("dataset_offset must be a non-negative integer")
    sliced = paths[dataset_offset::dataset_stride]
    if dataset_limit is not None:
        if dataset_limit <= 0:
            raise ValueError("dataset_limit must be a positive integer when provided")
        sliced = sliced[:dataset_limit]
    if not sliced:
        raise ValueError("No dataset samples selected after applying dataset subset arguments")
    return sliced


def resolve_city_map_paths(
    city_map_path,
    default_path: str | Path,
    dataset_limit: int | None = None,
    dataset_offset: int = 0,
    dataset_stride: int = 1,
) -> list[str]:
    default_path = str(default_path)
    if not city_map_path:
        return [default_path]
    if isinstance(city_map_path, (list, tuple)):
        paths = [str(path) for path in city_map_path if path] or [default_path]
        return _slice_dataset_paths(paths, dataset_limit, dataset_offset, dataset_stride)
    if isinstance(city_map_path, str):
        candidate = Path(city_map_path)
        if candidate.is_dir():
            allowed_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            paths = sorted(
                str(path)
                for path in candidate.iterdir()
                if path.suffix.lower() in allowed_exts
            )
            if not paths:
                raise ValueError(f"No image files found in city_map_path dir: {city_map_path}")
            return _slice_dataset_paths(paths, dataset_limit, dataset_offset, dataset_stride)
        if "," in city_map_path:
            paths = [item.strip() for item in city_map_path.split(",") if item.strip()] or [default_path]
            return _slice_dataset_paths(paths, dataset_limit, dataset_offset, dataset_stride)
        return [city_map_path]
    return [default_path]


def _roi_mask(pixel_map: np.ndarray) -> np.ndarray:
    return pixel_map <= ROI_THRESHOLD if ROI_IS_BLACK else pixel_map >= ROI_THRESHOLD

# 按 block 内是否存在合法像素判定合法。
def _action_block_bounds(pixel_map: np.ndarray, row_r: int, col_r: int) -> tuple[int, int, int, int]:
    height, width = pixel_map.shape
    row_step = max(height // ACTION_SPACE_SIZE, 1)
    col_step = max(width // ACTION_SPACE_SIZE, 1)
    row0 = row_r * row_step
    row1 = min((row_r + 1) * row_step, height)
    col0 = col_r * col_step
    col1 = min((col_r + 1) * col_step, width)
    return row0, row1, col0, col1


def calc_action_mask(pixel_map: np.ndarray) -> np.ndarray:
    mask = np.zeros((ACTION_SPACE_SIZE, ACTION_SPACE_SIZE), dtype=np.float32)
    for row_r in range(ACTION_SPACE_SIZE):
        for col_r in range(ACTION_SPACE_SIZE):
            row0, row1, col0, col1 = _action_block_bounds(pixel_map, row_r, col_r)
            block = pixel_map[row0:row1, col0:col1]
            mask[row_r, col_r] = 1.0 if np.any(block > 0.0) else 0.0
    return mask.reshape(-1).astype(np.float32)

# 返回 block 内最近中心的合法像素，不再固定回 block 中心
def calc_upsampling_loc(action: int, pixel_map: np.ndarray | None = None) -> tuple[int, int]:
    row_r, col_r = divmod(action, ACTION_SPACE_SIZE)
    if pixel_map is None:
        row = row_r * UPSAMPLING_FACTOR + (UPSAMPLING_FACTOR - 1) // 2
        col = col_r * UPSAMPLING_FACTOR + (UPSAMPLING_FACTOR - 1) // 2
        return int(row), int(col)

    row0, row1, col0, col1 = _action_block_bounds(pixel_map, row_r, col_r)
    block = pixel_map[row0:row1, col0:col1] > 0.0
    ys, xs = np.where(block)
    if len(xs) == 0:
        row = row0 + max((row1 - row0 - 1) // 2, 0)
        col = col0 + max((col1 - col0 - 1) // 2, 0)
        return int(row), int(col)

    center_y = (row1 - row0 - 1) / 2.0
    center_x = (col1 - col0 - 1) / 2.0
    distances = (ys.astype(np.float32) - center_y) ** 2 + (xs.astype(np.float32) - center_x) ** 2
    best_idx = int(np.argmin(distances))
    row = row0 + int(ys[best_idx])
    col = col0 + int(xs[best_idx])
    return int(row), int(col)


def build_tx_layer(pixel_map: np.ndarray, tx_locs: list[tuple[int, int]]) -> np.ndarray:
    tx_layer = np.zeros_like(pixel_map, dtype=float)
    for row, col in tx_locs:
        if 0 <= row < pixel_map.shape[0] and 0 <= col < pixel_map.shape[1]:
            tx_layer[row, col] = float(pixel_map[row, col]) if float(pixel_map[row, col]) > 0 else 0.5
    return tx_layer


def normalized_to_pathgain_db(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    array = np.clip(array, 0.0, 1.0)
    return array * (DB_MAX - DB_MIN) + DB_MIN


def get_powermap(pixel_map: np.ndarray, tx_layer: np.ndarray, pmnet=None) -> np.ndarray:
    inputs = np.stack([pixel_map, tx_layer], axis=2)
    predictor = pmnet or infer_surrogate
    pathgain_db = normalized_to_pathgain_db(predictor(inputs))
    pathgain_db[~_roi_mask(pixel_map)] = DB_MIN
    return pathgain_db


def compute_total_noise_power_dbm(noise_coefficient_db: float) -> float:
    return THERMAL_NOISE_DENSITY_DBM_PER_HZ + 10.0 * np.log10(CHANNEL_BANDWIDTH_HZ) + noise_coefficient_db


def compute_total_noise_power_mw(noise_coefficient_db: float) -> float:
    return float(np.power(10.0, compute_total_noise_power_dbm(noise_coefficient_db) / 10.0))


def build_roi_mask(pixel_map: np.ndarray) -> np.ndarray:
    mask = _roi_mask(pixel_map)
    if not mask.any():
        mask = np.ones_like(pixel_map, dtype=bool)
    return mask


def compute_score_v2_coverage_floor(
    coverage_target: float,
    floor_offset: float = DEFAULT_SCORE_V2_COVERAGE_FLOOR_OFFSET,
    floor_min: float = DEFAULT_SCORE_V2_COVERAGE_FLOOR_MIN,
    floor_max: float = DEFAULT_SCORE_V2_COVERAGE_FLOOR_MAX,
) -> float:
    return float(np.clip(coverage_target + floor_offset, floor_min, floor_max))


def _sigmoid(value: float) -> float:
    clipped = float(np.clip(value, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-clipped)))


def load_heuristic_targets(targets_path: str | Path | None) -> dict[str, dict[str, float]]:
    if not targets_path:
        return {}
    path = Path(targets_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_targets = payload.get("targets_by_image", payload)
    if not isinstance(raw_targets, dict):
        raise ValueError(f"Invalid heuristic targets payload: {path}")
    targets: dict[str, dict[str, float]] = {}
    for key, value in raw_targets.items():
        if not isinstance(value, dict):
            continue
        if "coverage_target" not in value or "spectral_efficiency_target" not in value:
            continue
        targets[str(Path(key).expanduser().resolve())] = {
            "coverage_target": float(value["coverage_target"]),
            "spectral_efficiency_target": float(value["spectral_efficiency_target"]),
        }
    return targets


def lookup_heuristic_targets(
    targets: dict[str, dict[str, float]],
    map_path: str | Path | None,
    fallback_coverage_target: float = DEFAULT_COVERAGE_TARGET,
    fallback_spectral_efficiency_target: float = DEFAULT_SPECTRAL_EFFICIENCY_TARGET,
) -> tuple[float, float]:
    if not map_path:
        return fallback_coverage_target, fallback_spectral_efficiency_target
    resolved = str(Path(map_path).expanduser().resolve())
    target = targets.get(resolved)
    if not target:
        return fallback_coverage_target, fallback_spectral_efficiency_target
    return float(target["coverage_target"]), float(target["spectral_efficiency_target"])


def evaluate_radio_metrics(
    pathgain_db: np.ndarray,
    summed_rx_power_mw: np.ndarray,
    roi_mask: np.ndarray,
    coverage_target: float = DEFAULT_COVERAGE_TARGET,
    spectral_efficiency_target: float = DEFAULT_SPECTRAL_EFFICIENCY_TARGET,
    w1: float = DEFAULT_W1,
    w2: float = DEFAULT_W2,
    coverage_threshold_db: float = DEFAULT_COVERAGE_THRESHOLD_DB,
    noise_coefficient_db: float = DEFAULT_NOISE_COEFFICIENT_DB,
) -> dict[str, float]:
    if pathgain_db.shape != roi_mask.shape:
        raise ValueError(f"pathgain_db shape {pathgain_db.shape} does not match roi_mask shape {roi_mask.shape}")
    if summed_rx_power_mw.shape != roi_mask.shape:
        raise ValueError(
            f"summed_rx_power_mw shape {summed_rx_power_mw.shape} does not match roi_mask shape {roi_mask.shape}"
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
    base_score = coverage + spectral_efficiency
    penalty = (
        w1 * max(0.0, coverage_target - coverage)
        + w2 * max(0.0, spectral_efficiency_target - spectral_efficiency)
    )
    score = base_score - penalty
    coverage_floor = compute_score_v2_coverage_floor(coverage_target)
    rss_margin = float(
        np.mean(
            np.tanh(
                (strongest_rx_power_dbm - coverage_threshold_db) / max(DEFAULT_SCORE_V2_MARGIN_SCALE_DB, 1e-6)
            )
        )
    )
    score_v2_gate = _sigmoid((coverage - coverage_floor) / max(DEFAULT_SCORE_V2_GATE_WIDTH, 1e-6))
    coverage_shortfall = max(0.0, coverage_floor - coverage)
    se_shortfall = max(0.0, DEFAULT_SCORE_V2_SE_FLOOR - spectral_efficiency)
    score_v2_penalty = (
        DEFAULT_SCORE_V2_COVERAGE_PENALTY_WEIGHT * (coverage_shortfall ** 2)
        + DEFAULT_SCORE_V2_SE_PENALTY_WEIGHT * (se_shortfall ** 2)
    )
    score_v2 = (
        DEFAULT_SCORE_V2_COVERAGE_WEIGHT * coverage
        + DEFAULT_SCORE_V2_MARGIN_WEIGHT * rss_margin
        + DEFAULT_SCORE_V2_SE_WEIGHT * score_v2_gate * spectral_efficiency
        - score_v2_penalty
    )
    return {
        "coverage": coverage,
        "spectral_efficiency": spectral_efficiency,
        "channel_capacity": channel_capacity,
        "score": score,
        "base_score": base_score,
        "penalty": penalty,
        "coverage_floor": coverage_floor,
        "rss_margin": rss_margin,
        "score_v2_gate": score_v2_gate,
        "score_v2_penalty": score_v2_penalty,
        "score_v2": score_v2,
    }


def get_site_pathgain_maps(city_map: np.ndarray, tx_locs: list[tuple[int, int]], pmnet=None) -> np.ndarray:
    if not tx_locs:
        return np.zeros((0, *city_map.shape), dtype=np.float32)
    site_maps = []
    for tx_loc in tx_locs:
        tx_layer = build_tx_layer(city_map, [tx_loc])
        site_maps.append(get_powermap(city_map, tx_layer, pmnet=pmnet))
    return np.stack(site_maps, axis=0).astype(np.float32)


# evaluate_radio_metrics
def get_stats(
    city_map: np.ndarray,
    tx_locs: list[tuple[int, int]],
    pmnet=None,
    coverage_target: float = DEFAULT_COVERAGE_TARGET,
    spectral_efficiency_target: float = DEFAULT_SPECTRAL_EFFICIENCY_TARGET,
    w1: float = DEFAULT_W1,
    w2: float = DEFAULT_W2,
    coverage_threshold_db: float = DEFAULT_COVERAGE_THRESHOLD_DB,
    noise_coefficient_db: float = DEFAULT_NOISE_COEFFICIENT_DB,
) -> tuple[np.ndarray, dict[str, float]]:
    site_pathgain_db = get_site_pathgain_maps(city_map, tx_locs, pmnet=pmnet)
    if site_pathgain_db.size == 0:
        empty_map = np.full_like(city_map, DB_MIN, dtype=np.float32)
        metrics = evaluate_radio_metrics(
            pathgain_db=empty_map,
            summed_rx_power_mw=np.zeros_like(city_map, dtype=np.float32),
            roi_mask=build_roi_mask(city_map),
            coverage_target=coverage_target,
            spectral_efficiency_target=spectral_efficiency_target,
            w1=w1,
            w2=w2,
            coverage_threshold_db=coverage_threshold_db,
            noise_coefficient_db=noise_coefficient_db,
        )
        return empty_map, metrics

    strongest_pathgain_db = np.max(site_pathgain_db, axis=0).astype(np.float32)
    rx_power_dbm = TX_POWER_DBM + site_pathgain_db.astype(np.float64)
    summed_rx_power_mw = np.sum(np.power(10.0, rx_power_dbm / 10.0), axis=0).astype(np.float32)
    metrics = evaluate_radio_metrics(
        pathgain_db=strongest_pathgain_db,
        summed_rx_power_mw=summed_rx_power_mw,
        roi_mask=build_roi_mask(city_map),
        coverage_target=coverage_target,
        spectral_efficiency_target=spectral_efficiency_target,
        w1=w1,
        w2=w2,
        coverage_threshold_db=coverage_threshold_db,
        noise_coefficient_db=noise_coefficient_db,
    )
    return strongest_pathgain_db, metrics


def select_reward(metrics: dict[str, float], reward_type: str) -> float:
    if reward_type == "coverage":
        return float(metrics["coverage"])
    if reward_type in {"capacity", "spectral_efficiency"}:
        return float(metrics["spectral_efficiency"])
    if reward_type == "channel_capacity":
        return float(metrics["channel_capacity"])
    if reward_type == "score":
        return float(metrics["score"])
    if reward_type == "score_v2":
        return float(metrics["score_v2"])
    raise ValueError(f"Unsupported reward_type: {reward_type}")
