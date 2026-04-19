"""
用途:
  无线接入点决策环境的基础工具函数，包括地图读取、候选动作映射，以及与 Heuristic 对齐的覆盖/容量/冗余率统计。

示例命令:
  无。该文件是公共模块，供环境和主入口导入。

参数说明:
  load_map_normalized(filepath): 读取灰度地图并归一化到 [0,1]。
  resolve_city_map_paths(city_map_path, default_path): 解析单文件、目录或逗号分隔路径列表。
  calc_action_mask(pixel_map): 计算缩小动作空间中的可选位置掩码。
  calc_upsampling_loc(action): 将缩小动作空间索引映射回 256x256 地图坐标。
  get_stats(city_map, tx_locs, pmnet): 计算路径增益图、覆盖率、容量和冗余率。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import yaml
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((ROOT_DIR / "base_config.yaml").read_text(encoding="utf-8")) or {}
ENV_CFG = CONFIG.get("env", {}) if isinstance(CONFIG, dict) else {}
HEURISTIC_DB_MIN = -162.0
HEURISTIC_DB_MAX = -75.0
TX_POWER_DBM = float(ENV_CFG.get("tx_power_dbm", 23.0))
PATHLOSS_THRESHOLD_DB = float(ENV_CFG.get("pathloss_threshold_db", 140.0))
THERMAL_NOISE_DENSITY_DBM_PER_HZ = -174.0
CHANNEL_BANDWIDTH_HZ = 20_000_000.0
NOISE_COEFFICIENT_DB = float(ENV_CFG.get("noise_coefficient_db", 10.0))
GROUND_TX_NORM = 0.5

action_space_size = int(ENV_CFG.get("action_space_size", 32))
map_size = int(ENV_CFG.get("map_size", 256))
upsampling_factor = map_size // action_space_size
roi_threshold = float(ENV_CFG.get("roi_threshold", 0.1))
roi_is_black = bool(ENV_CFG.get("roi_is_black", False))
tx_threshold = float(ENV_CFG.get("tx_threshold", roi_threshold))
tx_is_black = bool(ENV_CFG.get("tx_is_black", roi_is_black))
redundancy_count_thresholds = np.array(ENV_CFG.get("redundancy_count_thresholds", [1, 2]), dtype=int)
redundancy_target_cfg = ENV_CFG.get("redundancy_target", {"ideal": 0.45, "tolerance": 0.15})
coverage_threshold_db = float(
    ENV_CFG.get("coverage_threshold_db", TX_POWER_DBM - PATHLOSS_THRESHOLD_DB)
)

# 01 -> dB
def normalized_pathgain_to_db(pathgain_map: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(pathgain_map, dtype=np.float32), 0.0, 1.0)
    return clipped * (HEURISTIC_DB_MAX - HEURISTIC_DB_MIN) + HEURISTIC_DB_MIN

# dB -> 01
def db_pathgain_to_normalized(pathgain_db: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(pathgain_db, dtype=np.float32), HEURISTIC_DB_MIN, HEURISTIC_DB_MAX)
    return (clipped - HEURISTIC_DB_MIN) / (HEURISTIC_DB_MAX - HEURISTIC_DB_MIN)


def compute_total_noise_power_dbm(noise_coefficient_db: float = NOISE_COEFFICIENT_DB) -> float:
    return (
        THERMAL_NOISE_DENSITY_DBM_PER_HZ
        + 10.0 * np.log10(CHANNEL_BANDWIDTH_HZ)
        + float(noise_coefficient_db)
    )


def compute_total_noise_power_mw(noise_coefficient_db: float = NOISE_COEFFICIENT_DB) -> float:
    return float(np.power(10.0, compute_total_noise_power_dbm(noise_coefficient_db) / 10.0))


def load_map_normalized(filepath: str) -> np.ndarray:
    image = Image.open(filepath).convert("L")
    return np.array(image, dtype=np.float32) / 255.0


def resolve_city_map_paths(city_map_path, default_path: str) -> list[str]:
    if city_map_path is None or city_map_path == "":
        return [default_path]
    if isinstance(city_map_path, (list, tuple)):
        paths = [str(p) for p in city_map_path if p]
        return paths or [default_path]
    if isinstance(city_map_path, str):
        path = Path(city_map_path)
        if path.is_dir():
            allowed_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            paths = sorted(str(p) for p in path.iterdir() if p.suffix.lower() in allowed_exts)
            if not paths:
                raise ValueError(f"No image files found in city_map_path dir: {city_map_path}")
            return paths
        if "," in city_map_path:
            paths = [p.strip() for p in city_map_path.split(",") if p.strip()]
            return paths or [default_path]
        return [city_map_path]
    return [default_path]


# 得到的是RoI区域
def _roi_mask(pixel_map: np.ndarray) -> np.ndarray:
    if roi_is_black:
        return pixel_map <= roi_threshold
    return pixel_map >= roi_threshold

# 得到的是RoI区域 传入的是buildingsWHeight地图
def _tx_mask(pixel_map: np.ndarray) -> np.ndarray:
    # config: tx_is_black = roi_is_black
    #   roi_is_black: false
    #   roi_threshold: 0.01
    if tx_is_black:
        return pixel_map <= tx_threshold
    # 一定返回地图非黑的建筑物灰度位置
    return pixel_map >= tx_threshold


def encode_tx_normalized_value(gray: float) -> float:
    gray = float(gray)
    if gray > 0.0:
        return gray
    return GROUND_TX_NORM


def default_redundancy_target() -> dict[str, float]:
    target = redundancy_target_cfg if isinstance(redundancy_target_cfg, dict) else {}
    ideal = float(target.get("ideal", 0.45))
    tolerance = float(target.get("tolerance", 0.15))
    return {"ideal": ideal, "tolerance": max(tolerance, 1e-6)}


def normalize_redundancy_target(value) -> dict[str, float]:
    if isinstance(value, dict):
        ideal = float(value.get("ideal", default_redundancy_target()["ideal"]))
        tolerance = float(value.get("tolerance", default_redundancy_target()["tolerance"]))
        return {"ideal": ideal, "tolerance": max(tolerance, 1e-6)}
    return default_redundancy_target()


# pixel_map是二维的
def calc_action_mask(pixel_map: np.ndarray) -> np.ndarray:
    # upsampling_factor = map_size // action_space_size
    # 采样upsampling_factor的中心位置作为动作点
    idx = np.arange((upsampling_factor - 1) // 2, map_size, upsampling_factor)
    # 找到当前pixel_map中tx所在RoI区域采样块内的所有中心位置行，再从选出的这些行_tx_mask(pixel_map)[idx]中选择中心位置列
    action_pixels = np.where(_tx_mask(pixel_map)[idx][:, idx], 1, 0)
    # 从原始mask里，按固定间隔采样出一张“低分辨率网格”，只保留这些中心点的True/False情况
    return action_pixels.reshape(-1).astype(np.float32)


def calc_upsampling_loc(action: int) -> tuple[int, int]:
    row_r, col_r = divmod(action, action_space_size)
    row = row_r * upsampling_factor + (upsampling_factor - 1) // 2
    col = col_r * upsampling_factor + (upsampling_factor - 1) // 2
    return row, col


def get_powermap(pixel_map: np.ndarray, tx_layer: np.ndarray, pmnet: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    inputs = np.stack([pixel_map, tx_layer], axis=2)
    power_map = pmnet(inputs)
    # 非RoI区域全是黑色
    power_map[~_roi_mask(pixel_map)] = 0
    return power_map


def calc_coverage(city_map: np.ndarray, strongest_pathgain_db: np.ndarray) -> float:
    roi_mask = _roi_mask(city_map)
    roi_pathgain_db = np.asarray(strongest_pathgain_db, dtype=np.float32)[roi_mask]
    if roi_pathgain_db.size == 0:
        return 0.0
    strongest_rx_power_dbm = TX_POWER_DBM + roi_pathgain_db.astype(np.float64)
    return float(np.mean(strongest_rx_power_dbm >= coverage_threshold_db))


def calc_capacity(city_map: np.ndarray, strongest_pathgain_db: np.ndarray, total_rx_power_mw: np.ndarray) -> float:
    """

    Parameters
    ----------
    city_map
    strongest_pathgain_db:max(pathloss,...)
    total_rx_power_mw:sum(pathloss,...)

    Returns
    -------

    """
    roi_mask = _roi_mask(city_map)
    strongest_roi_db = np.asarray(strongest_pathgain_db, dtype=np.float64)[roi_mask]
    # 某个位置接收到的“所有信号功率之和”，包含最强基站信号（有用信号）和其他基站信号（干扰）
    total_roi_mw = np.asarray(total_rx_power_mw, dtype=np.float64)[roi_mask]
    if strongest_roi_db.size == 0:
        return 0.0
    strongest_rx_power_dbm = TX_POWER_DBM + strongest_roi_db
    strongest_rx_power_mw = np.power(10.0, strongest_rx_power_dbm / 10.0)
    noise_power_mw = compute_total_noise_power_mw()
    interference_power_mw = np.maximum(total_roi_mw - strongest_rx_power_mw, 0.0)
    sinr_linear = strongest_rx_power_mw / np.maximum(interference_power_mw + noise_power_mw, 1e-30)
    return float(np.mean(np.log2(1.0 + sinr_linear)))


def calc_redundancy_rate(city_map: np.ndarray, site_pathgain_db: np.ndarray) -> float:
    if site_pathgain_db.size == 0:
        return 0.0
    roi_mask = _roi_mask(city_map)
    if not np.any(roi_mask):
        return 0.0

    rx_power_dbm = TX_POWER_DBM + np.asarray(site_pathgain_db, dtype=np.float64)
    # 沿着站点通道维度统计非零的数量，原始返回的是布尔值
    covered_counts = np.count_nonzero(rx_power_dbm >= coverage_threshold_db, axis=0)
    roi_covered_counts = covered_counts[roi_mask]
    if roi_covered_counts.size == 0:
        return 0.0

    covered_pixels = roi_covered_counts >= int(redundancy_count_thresholds[0])
    # 统计大于redundancy_count_thresholds[0]的数量 bool -> int，np.count_nonzero返回标量
    covered_pixel_count = int(np.count_nonzero(covered_pixels))
    if covered_pixel_count == 0:
        return 0.0

    redundant_pixels = roi_covered_counts >= int(redundancy_count_thresholds[1])
    redundant_pixel_count = int(np.count_nonzero(redundant_pixels))
    # 衡量的是如果1个就能有这么大的覆盖率，你要用2个还是这么多就有些冗余了
    # 比值一定小于等于1
    return float(redundant_pixel_count / covered_pixel_count)


def redundancy_balance_score(value: float, target=None) -> float:
    # default_target_cfg = {"ideal": 0.45, "tolerance": 0.15}
    target_cfg = normalize_redundancy_target(target)
    deviation = abs(float(value) - float(target_cfg["ideal"]))
    tolerance = float(target_cfg["tolerance"])
    #         1
    #         ▲
    #        / \
    #       /   \
    #      /     \
    #     0-------0
    #  ideal±tolerance
    return max(0.0, 1.0 - deviation / tolerance)


def get_stats(city_map: np.ndarray, tx_locs: list[tuple[int, int]], pmnet: Callable[[np.ndarray], np.ndarray]):
    site_pathgain_db = []

    for row, col in tx_locs:
        single_tx = np.zeros_like(city_map, dtype=float)
        if 0 <= row < city_map.shape[0] and 0 <= col < city_map.shape[1]:
            single_tx[row, col] = encode_tx_normalized_value(float(city_map[row, col]))
        single_site_norm = get_powermap(city_map, single_tx, pmnet).astype(np.float32)
        site_pathgain_db.append(normalized_pathgain_to_db(single_site_norm))

    if site_pathgain_db:
        site_pathgain_db_stack = np.stack(site_pathgain_db, axis=0).astype(np.float32)
        strongest_pathgain_db = np.max(site_pathgain_db_stack, axis=0).astype(np.float32)
        rx_power_dbm = TX_POWER_DBM + site_pathgain_db_stack.astype(np.float64)
        total_rx_power_mw = np.sum(np.power(10.0, rx_power_dbm / 10.0), axis=0).astype(np.float64)
    else:
        site_pathgain_db_stack = np.empty((0,) + city_map.shape, dtype=np.float32)
        strongest_pathgain_db = np.full_like(city_map, HEURISTIC_DB_MIN, dtype=np.float32)
        total_rx_power_mw = np.zeros_like(city_map, dtype=np.float64)

    pathgain = db_pathgain_to_normalized(strongest_pathgain_db)
    coverage_reward = calc_coverage(city_map, strongest_pathgain_db)
    capacity_reward = calc_capacity(city_map, strongest_pathgain_db, total_rx_power_mw)
    redundancy_rate = calc_redundancy_rate(city_map, site_pathgain_db_stack)
    return pathgain, coverage_reward, capacity_reward, redundancy_rate


if  __name__ == "__main__":
    city_map_path = "/Users/epiphanyer/Desktop/coding/paper_experiment/dataset/png/buildingsWHeight/0.png"
    pixel_map = load_map_normalized(city_map_path)
    candidates = calc_action_mask(pixel_map)
    print(candidates.shape)