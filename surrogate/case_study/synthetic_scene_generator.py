"""
用途:
  程序化生成一个与 RadioMap3DSeer 输入统计更接近的 256m x 256m 合成街区场景，
  导出 buildingsWHeight / antennasWHeight / mesh / metadata，作为后续 Sionna 仿真的输入。

直接运行命令:
  python -m paper_experiment.surrogate.case_study.synthetic_scene_generator
  python -m paper_experiment.surrogate.case_study.synthetic_scene_generator --scene-id synthetic_a --tx-id 0 --seed 7
  python -m paper_experiment.surrogate.case_study.synthetic_scene_generator --output-root /tmp/case_study_scene --block-rows 4 --block-cols 4

参数说明:
  --output-root: 输出目录根，内部会生成 png/buildingsWHeight、png/antennasWHeight、meshes、metadata、blender。
  --scene-id: 建筑高度图与 metadata 的 scene 标识。
  --tx-id: 发射机图与 metadata 的 tx 标识。
  --seed: 随机种子；固定后可复现实验场景。
  --map-size-px: 输出图大小，默认 256，对应 256m x 256m。
  --world-size-m: 世界边长，默认 256m。
  --layout-mode: 建筑布局模式；`global_scatter` 为全图分散式采样，`block_grid` 为旧的分块街区模式。
  --block-rows/--block-cols: 程序化街区划分数。
  --road-width-m: 纵横主路宽度。
  --road-width-jitter-m: 道路宽度随机扰动幅度。
  --height-min-m/--height-max-m: 建筑高度采样范围。
  --target-occupancy: 目标建筑占比，用于控制楼体尺寸采样强度。
  --min-buildings-per-block/--max-buildings-per-block: 每个街区内采样楼栋数范围。
  --tx-margin-m: 发射机相对屋顶边缘的最小安全边距。
  --tx-anchor: 发射机在屋顶上的落点策略；`corner` 保持当前靠角落放置，`center` 放在屋顶中心，`random_inner` 在屋顶内部随机采样。
  --mesh-tile-length-m: 导出建筑 PLY 时单个面片的最大边长；仅作为近似 IRT 10 m tiling 的实验开关。默认值足够大，因此主链路不会切分建筑面。

逻辑说明:
  脚本先按指定布局模式生成二维 footprint 与高度，再同时导出三类结果：
  1. 数据集风格灰度输入图；
  2. 由长方体建筑拼成的 PLY mesh；
  3. Blender 可导入的 JSON manifest。
  这样可以先保证仿真输入的米制几何、像素栅格和统计分布一致，再决定是否进一步进入 Blender 精修。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .case_study_paths import BLENDER_ROOT, METADATA_ROOT, MESH_ROOT, OUTPUT_ROOT, PNG_ROOT


BUILDING_HEIGHT_MIN_M = 6.6
BUILDING_HEIGHT_MAX_M = 19.8
TX_HEIGHT_MIN_M = 19.5
TX_HEIGHT_MAX_M = 22.8
TX_HEIGHT_VALUE_MIN = 192
TX_HEIGHT_VALUE_MAX = 255
BUILDING_HEIGHT_LEVELS = np.linspace(BUILDING_HEIGHT_MIN_M, BUILDING_HEIGHT_MAX_M, 255, dtype=np.float32)


@dataclass(frozen=True)
class SyntheticSceneConfig:
    scene_id: str = "synthetic_scene"
    tx_id: str = "0"
    seed: int = 7
    map_size_px: int = 256
    world_size_m: float = 256.0
    layout_mode: str = "global_scatter"
    block_rows: int = 4
    block_cols: int = 4
    road_width_m: float = 12.0
    road_width_jitter_m: float = 2.0
    height_min_m: float = BUILDING_HEIGHT_MIN_M
    height_max_m: float = BUILDING_HEIGHT_MAX_M
    target_occupancy: float = 0.24
    min_buildings_per_block: int = 2
    max_buildings_per_block: int = 4
    building_spacing_m: float = 4.5
    footprint_min_edge_m: float = 12.0
    footprint_max_edge_m: float = 32.0
    tx_margin_m: float = 2.5
    tx_height_offset_m: float = 3.0
    tx_anchor: str = "corner"
    mesh_tile_length_m: float = 1000.0
    annex_probability: float = 0.08
    annex_max_count: int = 1
    height_jitter_m: float = 2.2


@dataclass(frozen=True)
class Building:
    building_id: str
    x0_m: float
    y0_m: float
    x1_m: float
    y1_m: float
    height_m: float


@dataclass(frozen=True)
class SceneBundle:
    config: SyntheticSceneConfig
    buildings: list[Building]
    height_map_m: np.ndarray
    buildings_u8: np.ndarray
    tx_u8: np.ndarray
    tx_position_m: tuple[float, float, float]
    building_ratio: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def encode_tx_value(z_m: float) -> np.uint8:
    z_clamped = min(max(float(z_m), TX_HEIGHT_MIN_M), TX_HEIGHT_MAX_M)
    if TX_HEIGHT_MAX_M == TX_HEIGHT_MIN_M:
        return np.uint8(TX_HEIGHT_VALUE_MAX)
    scaled = (z_clamped - TX_HEIGHT_MIN_M) / (TX_HEIGHT_MAX_M - TX_HEIGHT_MIN_M)
    value = round(scaled * (TX_HEIGHT_VALUE_MAX - TX_HEIGHT_VALUE_MIN) + TX_HEIGHT_VALUE_MIN)
    return np.uint8(min(max(value, 0), 255))


def height_to_uint8(height_map_m: np.ndarray, cfg: SyntheticSceneConfig) -> np.ndarray:
    out = np.zeros_like(height_map_m, dtype=np.uint8)
    mask = height_map_m > 0.0
    if not np.any(mask):
        return out
    heights = np.clip(height_map_m[mask], cfg.height_min_m, cfg.height_max_m)
    scaled = (heights - cfg.height_min_m) / max(cfg.height_max_m - cfg.height_min_m, 1e-6)
    out[mask] = np.round(1.0 + scaled * 254.0).astype(np.uint8)
    return out


def sample_discrete_building_height(rng: np.random.Generator, cfg: SyntheticSceneConfig) -> float:
    valid = BUILDING_HEIGHT_LEVELS[
        (BUILDING_HEIGHT_LEVELS >= cfg.height_min_m - 1e-6)
        & (BUILDING_HEIGHT_LEVELS <= cfg.height_max_m + 1e-6)
    ]
    if valid.size == 0:
        valid = BUILDING_HEIGHT_LEVELS
    return float(valid[int(rng.integers(0, len(valid)))])


def snap_height_to_dataset_level(height_m: float, cfg: SyntheticSceneConfig) -> float:
    valid = BUILDING_HEIGHT_LEVELS[
        (BUILDING_HEIGHT_LEVELS >= cfg.height_min_m - 1e-6)
        & (BUILDING_HEIGHT_LEVELS <= cfg.height_max_m + 1e-6)
    ]
    if valid.size == 0:
        valid = BUILDING_HEIGHT_LEVELS
    idx = int(np.argmin(np.abs(valid - float(height_m))))
    return float(valid[idx])


def world_to_grid(x_m: float, y_m: float, cfg: SyntheticSceneConfig) -> tuple[int, int]:
    cell = cfg.world_size_m / float(cfg.map_size_px)
    col = int(np.floor(x_m / cell))
    row = int(np.floor(y_m / cell))
    row = max(0, min(cfg.map_size_px - 1, row))
    col = max(0, min(cfg.map_size_px - 1, col))
    return row, col


def choose_tx_position(buildings: list[Building], cfg: SyntheticSceneConfig) -> tuple[float, float, float]:
    min_rooftop_height_m = 16.5
    center_150_margin = max((cfg.world_size_m - 150.0) / 2.0, 0.0)
    center_230_margin = max((cfg.world_size_m - 230.0) / 2.0, 0.0)

    def _building_center(building: Building) -> tuple[float, float]:
        return ((building.x0_m + building.x1_m) * 0.5, (building.y0_m + building.y1_m) * 0.5)

    def _inside_window(building: Building, margin: float) -> bool:
        cx, cy = _building_center(building)
        return margin <= cx <= cfg.world_size_m - margin and margin <= cy <= cfg.world_size_m - margin

    rooftop_candidates = [b for b in buildings if b.height_m >= min_rooftop_height_m]
    if not rooftop_candidates:
        rooftop_candidates = list(buildings)

    preferred = [b for b in rooftop_candidates if _inside_window(b, center_150_margin)]
    if not preferred:
        preferred = [b for b in rooftop_candidates if _inside_window(b, center_230_margin)]
    if not preferred:
        preferred = rooftop_candidates

    world_center = cfg.world_size_m * 0.5
    ranked = sorted(
        preferred,
        key=lambda b: (
            -b.height_m,
            -((b.x1_m - b.x0_m) * (b.y1_m - b.y0_m)),
            abs(_building_center(b)[0] - world_center) + abs(_building_center(b)[1] - world_center),
        ),
    )
    best = ranked[0]
    width = best.x1_m - best.x0_m
    depth = best.y1_m - best.y0_m
    margin_x = min(max(cfg.tx_margin_m, 0.0), max(width * 0.25, 0.0))
    margin_y = min(max(cfg.tx_margin_m, 0.0), max(depth * 0.25, 0.0))
    center_x = (best.x0_m + best.x1_m) * 0.5
    center_y = (best.y0_m + best.y1_m) * 0.5
    if cfg.tx_anchor == "center":
        x = center_x
        y = center_y
    elif cfg.tx_anchor == "random_inner":
        rng = np.random.default_rng(cfg.seed + 10_003)
        inner_x0 = min(best.x0_m + margin_x, best.x1_m)
        inner_x1 = max(best.x1_m - margin_x, inner_x0)
        inner_y0 = min(best.y0_m + margin_y, best.y1_m)
        inner_y1 = max(best.y1_m - margin_y, inner_y0)
        x = float(rng.uniform(inner_x0, inner_x1)) if inner_x1 > inner_x0 else center_x
        y = float(rng.uniform(inner_y0, inner_y1)) if inner_y1 > inner_y0 else center_y
    else:
        x = best.x0_m + margin_x
        y = best.y0_m + margin_y
        x = min(x, best.x1_m - cfg.tx_margin_m)
        y = min(y, best.y1_m - cfg.tx_margin_m)
    z = min(best.height_m + cfg.tx_height_offset_m, TX_HEIGHT_MAX_M)
    z = max(z, TX_HEIGHT_MIN_M)
    return (x, y, z)


def _block_edges(
    total: float,
    block_count: int,
    road_width: float,
    jitter: float,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    roads = np.full(block_count + 1, float(road_width), dtype=np.float32)
    if jitter > 0.0:
        noise = rng.uniform(-jitter, jitter, size=block_count + 1)
        roads = np.maximum(roads + noise, max(road_width * 0.45, 4.0))
    usable = total - float(roads.sum())
    usable = max(usable, total * 0.35)
    weights = rng.uniform(0.75, 1.35, size=block_count)
    weights /= weights.sum()
    spans = usable * weights
    edges: list[tuple[float, float]] = []
    cursor = float(roads[0])
    for idx in range(block_count):
        start = cursor
        end = start + float(spans[idx])
        edges.append((start, end))
        cursor = end + float(roads[idx + 1])
    return edges


def _overlaps(candidate: Building, existing: list[Building], spacing: float) -> bool:
    for current in existing:
        if not (
            candidate.x1_m + spacing <= current.x0_m
            or candidate.x0_m >= current.x1_m + spacing
            or candidate.y1_m + spacing <= current.y0_m
            or candidate.y0_m >= current.y1_m + spacing
        ):
            return True
    return False


def _make_annexes(
    rng: np.random.Generator,
    cfg: SyntheticSceneConfig,
    row_idx: int,
    col_idx: int,
    start_index: int,
    main: Building,
    existing: list[Building],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> list[Building]:
    annexes: list[Building] = []
    if rng.random() > cfg.annex_probability:
        return annexes
    annex_count = int(rng.integers(1, cfg.annex_max_count + 1))
    side_order = ["west", "east", "south", "north"]
    rng.shuffle(side_order)
    for side in side_order[:annex_count]:
        base_w = main.x1_m - main.x0_m
        base_d = main.y1_m - main.y0_m
        annex_w = float(rng.uniform(base_w * 0.22, base_w * 0.48))
        annex_d = float(rng.uniform(base_d * 0.22, base_d * 0.52))
        if side == "west":
            x1 = main.x0_m + float(rng.uniform(-0.8, 0.8))
            x0 = x1 - annex_w
            y0 = float(rng.uniform(main.y0_m - annex_d * 0.2, main.y1_m - annex_d))
            y1 = y0 + annex_d
        elif side == "east":
            x0 = main.x1_m + float(rng.uniform(-0.8, 0.8))
            x1 = x0 + annex_w
            y0 = float(rng.uniform(main.y0_m - annex_d * 0.2, main.y1_m - annex_d))
            y1 = y0 + annex_d
        elif side == "south":
            y1 = main.y0_m + float(rng.uniform(-0.8, 0.8))
            y0 = y1 - annex_d
            x0 = float(rng.uniform(main.x0_m - annex_w * 0.2, main.x1_m - annex_w))
            x1 = x0 + annex_w
        else:
            y0 = main.y1_m + float(rng.uniform(-0.8, 0.8))
            y1 = y0 + annex_d
            x0 = float(rng.uniform(main.x0_m - annex_w * 0.2, main.x1_m - annex_w))
            x1 = x0 + annex_w
        x0 = max(x_bounds[0] + 0.5, x0)
        y0 = max(y_bounds[0] + 0.5, y0)
        x1 = min(x_bounds[1] - 0.5, x1)
        y1 = min(y_bounds[1] - 0.5, y1)
        if x1 - x0 < cfg.footprint_min_edge_m * 0.3 or y1 - y0 < cfg.footprint_min_edge_m * 0.3:
            continue
        annex = Building(
            building_id=f"b_{row_idx}_{col_idx}_{start_index + len(annexes)}",
            x0_m=x0,
            y0_m=y0,
            x1_m=x1,
            y1_m=y1,
            height_m=snap_height_to_dataset_level(
                np.clip(main.height_m + rng.uniform(-cfg.height_jitter_m, cfg.height_jitter_m), cfg.height_min_m, cfg.height_max_m),
                cfg,
            ),
        )
        if not _overlaps(annex, existing + annexes, cfg.building_spacing_m * 0.2):
            annexes.append(annex)
    return annexes


def _sample_buildings_in_block(
    rng: np.random.Generator,
    cfg: SyntheticSceneConfig,
    row_idx: int,
    col_idx: int,
    start_index: int,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> list[Building]:
    buildings: list[Building] = []
    count = int(rng.integers(cfg.min_buildings_per_block, cfg.max_buildings_per_block + 1))
    x0, x1 = x_bounds
    y0, y1 = y_bounds
    inner_x0 = x0 + cfg.building_spacing_m
    inner_x1 = x1 - cfg.building_spacing_m
    inner_y0 = y0 + cfg.building_spacing_m
    inner_y1 = y1 - cfg.building_spacing_m
    if inner_x1 - inner_x0 < cfg.footprint_min_edge_m or inner_y1 - inner_y0 < cfg.footprint_min_edge_m:
        return buildings

    attempts = 0
    max_attempts = count * 24
    while len(buildings) < count and attempts < max_attempts:
        attempts += 1
        width = float(rng.uniform(cfg.footprint_min_edge_m, cfg.footprint_max_edge_m))
        depth = float(rng.uniform(cfg.footprint_min_edge_m, cfg.footprint_max_edge_m))
        width = min(width, inner_x1 - inner_x0)
        depth = min(depth, inner_y1 - inner_y0)
        left = float(rng.uniform(inner_x0, inner_x1 - width))
        bottom = float(rng.uniform(inner_y0, inner_y1 - depth))
        right = left + width
        top = bottom + depth
        candidate = Building(
            building_id=f"b_{row_idx}_{col_idx}_{start_index + len(buildings)}",
            x0_m=left,
            y0_m=bottom,
            x1_m=right,
            y1_m=top,
            height_m=sample_discrete_building_height(rng, cfg),
        )
        if not _overlaps(candidate, buildings, cfg.building_spacing_m):
            buildings.append(candidate)
            annex_start = start_index + len(buildings)
            buildings.extend(
                _make_annexes(
                    rng,
                    cfg,
                    row_idx,
                    col_idx,
                    annex_start,
                    candidate,
                    buildings,
                    x_bounds,
                    y_bounds,
                )
            )
    return buildings


def _sample_buildings_global(
    rng: np.random.Generator,
    cfg: SyntheticSceneConfig,
) -> list[Building]:
    buildings: list[Building] = []
    x0 = cfg.building_spacing_m
    x1 = cfg.world_size_m - cfg.building_spacing_m
    y0 = cfg.building_spacing_m
    y1 = cfg.world_size_m - cfg.building_spacing_m
    if x1 - x0 < cfg.footprint_min_edge_m or y1 - y0 < cfg.footprint_min_edge_m:
        return buildings

    attempts = 0
    max_attempts = 6000
    while _estimate_building_occupancy(buildings, cfg) < cfg.target_occupancy and attempts < max_attempts:
        attempts += 1
        width = float(rng.uniform(cfg.footprint_min_edge_m, cfg.footprint_max_edge_m))
        depth = float(rng.uniform(cfg.footprint_min_edge_m, cfg.footprint_max_edge_m))
        width = min(width, x1 - x0)
        depth = min(depth, y1 - y0)
        left = float(rng.uniform(x0, x1 - width))
        bottom = float(rng.uniform(y0, y1 - depth))
        right = left + width
        top = bottom + depth
        candidate = Building(
            building_id=f"g_{len(buildings):04d}",
            x0_m=left,
            y0_m=bottom,
            x1_m=right,
            y1_m=top,
            height_m=sample_discrete_building_height(rng, cfg),
        )
        if _overlaps(candidate, buildings, cfg.building_spacing_m):
            continue
        buildings.append(candidate)
        annex_start = len(buildings)
        buildings.extend(
            _make_annexes(
                rng,
                cfg,
                0,
                0,
                annex_start,
                candidate,
                buildings,
                (0.0, cfg.world_size_m),
                (0.0, cfg.world_size_m),
            )
        )
    return buildings


def generate_buildings(cfg: SyntheticSceneConfig) -> list[Building]:
    rng = np.random.default_rng(cfg.seed)
    if cfg.layout_mode == "global_scatter":
        return _sample_buildings_global(rng, cfg)
    if cfg.layout_mode != "block_grid":
        raise ValueError(f"Unsupported layout_mode: {cfg.layout_mode}")
    x_edges = _block_edges(cfg.world_size_m, cfg.block_cols, cfg.road_width_m, cfg.road_width_jitter_m, rng)
    y_edges = _block_edges(cfg.world_size_m, cfg.block_rows, cfg.road_width_m, cfg.road_width_jitter_m, rng)
    buildings: list[Building] = []
    for row_idx, y_bounds in enumerate(y_edges):
        for col_idx, x_bounds in enumerate(x_edges):
            block_existing = sum(1 for building in buildings if building.building_id.startswith(f"b_{row_idx}_{col_idx}_"))
            buildings.extend(_sample_buildings_in_block(rng, cfg, row_idx, col_idx, block_existing, x_bounds, y_bounds))
    occupancy = _estimate_building_occupancy(buildings, cfg)
    attempts = 0
    while occupancy < cfg.target_occupancy and attempts < 64:
        attempts += 1
        row_idx = int(rng.integers(0, len(y_edges)))
        col_idx = int(rng.integers(0, len(x_edges)))
        block_existing = sum(1 for building in buildings if building.building_id.startswith(f"b_{row_idx}_{col_idx}_"))
        extras = _sample_buildings_in_block(rng, cfg, row_idx, col_idx, block_existing, x_edges[col_idx], y_edges[row_idx])
        if not extras:
            continue
        buildings.extend(extras[:1])
        occupancy = _estimate_building_occupancy(buildings, cfg)
    return buildings


def rasterize_height_map(buildings: list[Building], cfg: SyntheticSceneConfig) -> np.ndarray:
    height_map = np.zeros((cfg.map_size_px, cfg.map_size_px), dtype=np.float32)
    cell = cfg.world_size_m / float(cfg.map_size_px)
    for building in buildings:
        c0 = max(0, int(np.floor(building.x0_m / cell)))
        c1 = min(cfg.map_size_px, int(np.ceil(building.x1_m / cell)))
        r0 = max(0, int(np.floor(building.y0_m / cell)))
        r1 = min(cfg.map_size_px, int(np.ceil(building.y1_m / cell)))
        height_map[r0:r1, c0:c1] = np.maximum(height_map[r0:r1, c0:c1], building.height_m)
    return height_map


def _estimate_building_occupancy(buildings: list[Building], cfg: SyntheticSceneConfig) -> float:
    if not buildings:
        return 0.0
    area = sum(max(0.0, b.x1_m - b.x0_m) * max(0.0, b.y1_m - b.y0_m) for b in buildings)
    total = cfg.world_size_m * cfg.world_size_m
    return float(area / total) if total > 0.0 else 0.0


def generate_scene_bundle(cfg: SyntheticSceneConfig) -> SceneBundle:
    buildings = generate_buildings(cfg)
    if not buildings:
        raise ValueError("No buildings were generated; relax the block or footprint constraints.")
    height_map_m = rasterize_height_map(buildings, cfg)
    buildings_u8 = np.flipud(height_to_uint8(height_map_m, cfg))
    tx_position_m = choose_tx_position(buildings, cfg)
    tx_u8 = np.zeros((cfg.map_size_px, cfg.map_size_px), dtype=np.uint8)
    tx_row, tx_col = world_to_grid(tx_position_m[0], tx_position_m[1], cfg)
    tx_u8[tx_row, tx_col] = encode_tx_value(tx_position_m[2])
    tx_u8 = np.flipud(tx_u8)
    building_ratio = float((height_map_m > 0.0).mean())
    return SceneBundle(
        config=cfg,
        buildings=buildings,
        height_map_m=height_map_m,
        buildings_u8=buildings_u8,
        tx_u8=tx_u8,
        tx_position_m=tx_position_m,
        building_ratio=building_ratio,
    )


def _axis_samples(start: float, end: float, tile_length: float) -> list[float]:
    span = max(float(end) - float(start), 0.0)
    if span <= 0.0:
        return [float(start), float(end)]
    segments = max(1, int(np.ceil(span / max(tile_length, 1e-6))))
    return np.linspace(float(start), float(end), segments + 1, dtype=np.float32).tolist()


def _append_tiled_plane(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    p00: tuple[float, float, float],
    p10: tuple[float, float, float],
    p11: tuple[float, float, float],
    p01: tuple[float, float, float],
    outward_positive: bool,
) -> None:
    base = len(vertices)
    vertices.extend([p00, p10, p11, p01])
    if outward_positive:
        faces.extend([(base + 0, base + 1, base + 2), (base + 0, base + 2, base + 3)])
    else:
        faces.extend([(base + 0, base + 2, base + 1), (base + 0, base + 3, base + 2)])


def _cuboid_mesh(
    building: Building,
    tile_length_m: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    x0, y0, x1, y1, z = building.x0_m, building.y0_m, building.x1_m, building.y1_m, building.height_m
    width = x1 - x0
    depth = y1 - y0
    # Preserve the original exact cuboid triangulation for the main path when
    # no face tiling is requested. This restores the pre-tiling scene geometry.
    if tile_length_m >= max(width, depth, z):
        vertices = [
            (x0, y0, 0.0),
            (x1, y0, 0.0),
            (x1, y1, 0.0),
            (x0, y1, 0.0),
            (x0, y0, z),
            (x1, y0, z),
            (x1, y1, z),
            (x0, y1, z),
        ]
        faces = [
            (0, 1, 2), (0, 2, 3),
            (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4),
            (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6),
            (3, 0, 4), (3, 4, 7),
        ]
        return vertices, faces
    xs = _axis_samples(x0, x1, tile_length_m)
    ys = _axis_samples(y0, y1, tile_length_m)
    zs = _axis_samples(0.0, z, tile_length_m)

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    for ix in range(len(xs) - 1):
        for iy in range(len(ys) - 1):
            xa, xb = xs[ix], xs[ix + 1]
            ya, yb = ys[iy], ys[iy + 1]
            _append_tiled_plane(vertices, faces, (xa, ya, z), (xb, ya, z), (xb, yb, z), (xa, yb, z), True)
            _append_tiled_plane(vertices, faces, (xa, ya, 0.0), (xa, yb, 0.0), (xb, yb, 0.0), (xb, ya, 0.0), False)

    for ix in range(len(xs) - 1):
        for iz in range(len(zs) - 1):
            xa, xb = xs[ix], xs[ix + 1]
            za, zb = zs[iz], zs[iz + 1]
            _append_tiled_plane(vertices, faces, (xa, y0, za), (xb, y0, za), (xb, y0, zb), (xa, y0, zb), False)
            _append_tiled_plane(vertices, faces, (xa, y1, za), (xa, y1, zb), (xb, y1, zb), (xb, y1, za), True)

    for iy in range(len(ys) - 1):
        for iz in range(len(zs) - 1):
            ya, yb = ys[iy], ys[iy + 1]
            za, zb = zs[iz], zs[iz + 1]
            _append_tiled_plane(vertices, faces, (x0, ya, za), (x0, ya, zb), (x0, yb, zb), (x0, yb, za), True)
            _append_tiled_plane(vertices, faces, (x1, ya, za), (x1, yb, za), (x1, yb, zb), (x1, ya, zb), False)
    return vertices, faces


def write_ply(path: Path, vertices: list[tuple[float, float, float]], faces: list[tuple[int, ...]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for x, y, z in vertices:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        for face in faces:
            handle.write(f"{len(face)} {' '.join(str(v) for v in face)}\n")


def export_scene_meshes(mesh_root: Path, buildings: list[Building], tile_length_m: float) -> list[str]:
    ensure_dir(mesh_root)
    mesh_paths: list[str] = []
    for building in buildings:
        vertices, faces = _cuboid_mesh(building, tile_length_m=tile_length_m)
        path = mesh_root / f"{building.building_id}.ply"
        write_ply(path, vertices, faces)
        mesh_paths.append(str(path))
    return mesh_paths


def write_blender_manifest(path: Path, bundle: SceneBundle, mesh_paths: list[str]) -> None:
    ensure_dir(path.parent)
    payload = {
        "scene_id": bundle.config.scene_id,
        "tx_id": bundle.config.tx_id,
        "world_size_m": bundle.config.world_size_m,
        "map_size_px": bundle.config.map_size_px,
        "tx_position_m": bundle.tx_position_m,
        "mesh_paths": mesh_paths,
        "buildings": [asdict(building) for building in bundle.buildings],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_synthetic_scene(output_root: Path, cfg: SyntheticSceneConfig) -> SceneBundle:
    bundle = generate_scene_bundle(cfg)
    buildings_dir = output_root / "png" / "buildingsWHeight"
    tx_dir = output_root / "png" / "antennasWHeight"
    mesh_dir = output_root / "meshes"
    metadata_dir = output_root / "metadata"
    blender_dir = output_root / "blender"
    for path in (buildings_dir, tx_dir, mesh_dir, metadata_dir, blender_dir):
        ensure_dir(path)

    buildings_path = buildings_dir / f"{cfg.scene_id}.png"
    tx_path = tx_dir / f"{cfg.scene_id}_{cfg.tx_id}.png"
    metadata_path = metadata_dir / f"{cfg.scene_id}_{cfg.tx_id}.json"
    blender_manifest_path = blender_dir / f"{cfg.scene_id}_{cfg.tx_id}_manifest.json"

    Image.fromarray(bundle.buildings_u8).save(buildings_path)
    Image.fromarray(bundle.tx_u8).save(tx_path)
    np.save(metadata_dir / f"{cfg.scene_id}_height_map_m.npy", bundle.height_map_m)

    mesh_paths = export_scene_meshes(mesh_dir, bundle.buildings, tile_length_m=cfg.mesh_tile_length_m)
    write_blender_manifest(blender_manifest_path, bundle, mesh_paths)

    metadata = {
        "config": asdict(cfg),
        "building_ratio": bundle.building_ratio,
        "building_count": len(bundle.buildings),
        "tx_position_m": list(bundle.tx_position_m),
        "mesh_count": len(mesh_paths),
        "mesh_tile_length_m": cfg.mesh_tile_length_m,
        "buildings_path": str(buildings_path),
        "tx_path": str(tx_path),
        "blender_manifest_path": str(blender_manifest_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a dataset-aligned synthetic urban scene.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--scene-id", default="synthetic_scene")
    parser.add_argument("--tx-id", default="0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--map-size-px", type=int, default=256)
    parser.add_argument("--world-size-m", type=float, default=256.0)
    parser.add_argument("--layout-mode", choices=["global_scatter", "block_grid"], default="global_scatter")
    parser.add_argument("--block-rows", type=int, default=4)
    parser.add_argument("--block-cols", type=int, default=4)
    parser.add_argument("--road-width-m", type=float, default=12.0)
    parser.add_argument("--road-width-jitter-m", type=float, default=2.0)
    parser.add_argument("--height-min-m", type=float, default=BUILDING_HEIGHT_MIN_M)
    parser.add_argument("--height-max-m", type=float, default=BUILDING_HEIGHT_MAX_M)
    parser.add_argument("--target-occupancy", type=float, default=0.24)
    parser.add_argument("--min-buildings-per-block", type=int, default=2)
    parser.add_argument("--max-buildings-per-block", type=int, default=4)
    parser.add_argument("--building-spacing-m", type=float, default=4.5)
    parser.add_argument("--footprint-min-edge-m", type=float, default=12.0)
    parser.add_argument("--footprint-max-edge-m", type=float, default=32.0)
    parser.add_argument("--tx-margin-m", type=float, default=2.5)
    parser.add_argument("--tx-height-offset-m", type=float, default=3.0)
    parser.add_argument("--tx-anchor", choices=["corner", "center", "random_inner"], default="corner")
    parser.add_argument("--mesh-tile-length-m", type=float, default=1000.0)
    parser.add_argument("--annex-probability", type=float, default=0.08)
    parser.add_argument("--annex-max-count", type=int, default=1)
    parser.add_argument("--height-jitter-m", type=float, default=2.2)
    return parser


def config_from_args(args: argparse.Namespace) -> SyntheticSceneConfig:
    return SyntheticSceneConfig(
        scene_id=args.scene_id,
        tx_id=args.tx_id,
        seed=args.seed,
        map_size_px=args.map_size_px,
        world_size_m=args.world_size_m,
        layout_mode=args.layout_mode,
        block_rows=args.block_rows,
        block_cols=args.block_cols,
        road_width_m=args.road_width_m,
        road_width_jitter_m=args.road_width_jitter_m,
        height_min_m=args.height_min_m,
        height_max_m=args.height_max_m,
        target_occupancy=args.target_occupancy,
        min_buildings_per_block=args.min_buildings_per_block,
        max_buildings_per_block=args.max_buildings_per_block,
        building_spacing_m=args.building_spacing_m,
        footprint_min_edge_m=args.footprint_min_edge_m,
        footprint_max_edge_m=args.footprint_max_edge_m,
        tx_margin_m=args.tx_margin_m,
        tx_height_offset_m=args.tx_height_offset_m,
        tx_anchor=args.tx_anchor,
        mesh_tile_length_m=args.mesh_tile_length_m,
        annex_probability=args.annex_probability,
        annex_max_count=args.annex_max_count,
        height_jitter_m=args.height_jitter_m,
    )


def main() -> None:
    args = build_parser().parse_args()
    cfg = config_from_args(args)
    bundle = export_synthetic_scene(args.output_root, cfg)
    print(f"output_root={args.output_root}")
    print(f"scene_id={cfg.scene_id}")
    print(f"tx_id={cfg.tx_id}")
    print(f"building_count={len(bundle.buildings)}")
    print(f"building_ratio={bundle.building_ratio:.4f}")
    print(f"tx_position_m={bundle.tx_position_m}")
    print(f"buildings_png={args.output_root / 'png' / 'buildingsWHeight' / f'{cfg.scene_id}.png'}")
    print(f"tx_png={args.output_root / 'png' / 'antennasWHeight' / f'{cfg.scene_id}_{cfg.tx_id}.png'}")
    print(f"metadata_json={args.output_root / 'metadata' / f'{cfg.scene_id}_{cfg.tx_id}.json'}")


if __name__ == "__main__":
    main()
