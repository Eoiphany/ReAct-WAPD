"""
用途:
  使用 case_study 的 synthetic scene XML 运行 Sionna RT，并导出 gain / path_gain_db / stats。

直接运行命令:
  python -m paper_experiment.surrogate.case_study.run_case_study_sionna \
    --scene /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/case_study/output_dataset/sionna/synthetic_scene.xml \
    --building-map-path /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/case_study/output_dataset/png/buildingsWHeight/synthetic_scene.png \
    --metadata-path /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/case_study/output_dataset/metadata/synthetic_scene_0.json \
    --output-dir /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/case_study/output_dataset \
    --sim-cell-size-m 0.5 \
    --samples-per-tx 16000000 \
    --max-depth 2 \
    --label-smooth-sigma 1.7 \
    --label-smooth-mix 0.48 \
    --label-post-sigma 0.7 \
    --label-broad-blend 0.08

参数说明:
  输入与输出:
    --scene:
      Mitsuba/Sionna scene XML 路径，即真正送入 Sionna RT 的三维场景描述。
    --output-dir:
      输出根目录；脚本会在这里写入 `gain/`、`png/`、`*.npy` 和 `stats_case_study.txt`。
    --scene-id:
      导出数据集时使用的场景名；为空时默认取 metadata 里的 `scene_id`。
    --tx-id:
      导出数据集时使用的发射机编号；为空时默认取 metadata 里的 `tx_id`。
    --metadata-path:
      synthetic_scene_generator 生成的 JSON；提供 `scene_id`、`tx_id`、`tx_position_m`、`world_size_m` 等真值。
    --building-map-path:
      建筑高度图 `buildingsWHeight` PNG；用于恢复建筑遮挡掩码，并和最终 gain 图保持同一像素网格。
    --tx-position:
      手动覆盖发射机坐标，格式为 `x,y,z`，单位米；不传时默认读取 metadata 中的 `tx_position_m`。

  无线与数值仿真参数:
    --frequency-hz:
      载波频率，单位 Hz。
    --tx-power-dbm:
      发射功率，单位 dBm；该值只影响 `rx_power_dbm.npy` 与 `snr_db.npy`，不改变纯 path gain 几何传播结果。
    --bandwidth-hz:
      接收带宽，单位 Hz；用于把噪声 PSD 换算成总噪声功率。
    --noise-psd-dbm-per-hz:
      噪声功率谱密度，单位 dBm/Hz。
    --noise-figure-db:
      接收机噪声系数，单位 dB。
    --cell-size-m:
      最终导出给数据集 PNG 的像素尺寸，单位米/像素；它决定导出的 gain 图是否接近“1 米逐像素”。
    --sim-cell-size-m:
      Sionna 内部覆盖图的采样步长，单位米/像素；越小仿真越细，但显存和耗时都会明显增加。
    --map-size-px:
      导出图像边长，单位像素；最终 gain/building/tx 图都会被整理到这个尺寸。
    --world-size-m:
      覆盖图在世界坐标中的边长，单位米；为空时默认读取 metadata 中的 `world_size_m`。
    --max-depth:
      最多追踪的传播反弹/绕射深度；值越大，多径更丰富，但计算更慢。
    --samples-per-tx:
      每个发射机发射的射线采样数；这是控制 Monte Carlo 精度与噪点多少的核心参数之一。
    --reflection / --no-reflection:
      是否启用镜面反射。
    --diffraction / --no-diffraction:
      是否启用绕射。
    --scattering / --no-scattering:
      是否启用漫散射。
    --edge-diffraction / --no-edge-diffraction:
      是否启用边缘绕射。
    --building-scattering-coefficient:
      对所有建筑统一覆盖的漫散射系数；`0` 表示关闭 diffuse scattering，仅保留材质默认值。

  gain 标签量化与后处理参数:
    --db-min:
      导出 8-bit gain 图时对应的最小 dB；低于该值会被截断到 0。
    --db-max:
      导出 8-bit gain 图时对应的最大 dB；高于该值会被截断到 255。
    --label-smooth-sigma:
      第一次高斯平滑的标准差；用于压掉高频离散噪点。
    --label-smooth-mix:
      平滑结果与原始结果的混合比例；越大越平滑，越小越保留硬边。
    --label-db-offset:
      在量化前对整张 dB 图施加的全局偏移量。
    --label-post-sigma:
      混合完成后额外再做一次平滑的标准差。
    --label-gain:
      对平滑结果施加的线性增益系数。
    --label-gamma:
      对量化后的亮度做 gamma 变换；大于 1 会压暗弱信号区域，小于 1 会抬亮弱信号区域。
    --label-broad-blend:
      额外的宽核平滑混合权重；用于做更大尺度的软化。
    --label-top-gain:
      对高亮强信号区域的增强系数。
    --label-bottom-gain:
      对低亮弱信号区域的增强系数。

逻辑说明:
  脚本复用 Paris 版本的数值导出与标签后处理，但把覆盖图中心改成 synthetic scene 的世界中心，
  并按 0..world_size 的坐标定义把 Tx 投到内部栅格，再统一翻转为图像坐标。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .build_sionna_scene_xml import DEFAULT_OUTPUT_XML
from .case_study_paths import METADATA_ROOT, OUTPUT_ROOT, PNG_ROOT
from ..blender_scene.paris.france_success_dataset import (
    BANDWIDTH_HZ,
    DB_MAX,
    DB_MIN,
    DEFAULT_LABEL_BOTTOM_GAIN,
    DEFAULT_LABEL_BROAD_BLEND,
    DEFAULT_LABEL_DB_OFFSET,
    DEFAULT_LABEL_GAIN,
    DEFAULT_LABEL_GAMMA,
    DEFAULT_LABEL_POST_SIGMA,
    DEFAULT_LABEL_SMOOTH_MIX,
    DEFAULT_LABEL_SMOOTH_SIGMA,
    DEFAULT_LABEL_TOP_GAIN,
    FREQUENCY_HZ,
    MAP_SIZE_PX,
    NOISE_FIGURE_DB,
    NOISE_PSD_DBM_PER_HZ,
    PIXEL_LENGTH_M,
    RX_HEIGHT_M,
    TX_POWER_DBM,
    ensure_dir,
    get_metric_array,
    linear_metric_to_db,
    load_buildings_u8,
    metric_stats,
    resize_metric_to_export_grid,
    set_optional_attr,
    smooth_gain_uint8,
)


DEFAULT_BUILDING_MAP_PATH = PNG_ROOT / "buildingsWHeight" / "synthetic_scene.png"
DEFAULT_METADATA_PATH = METADATA_ROOT / "synthetic_scene_0.json"
DEFAULT_CASE_SIM_CELL_SIZE_M = 0.5
DEFAULT_CASE_LABEL_SMOOTH_SIGMA = 0.0
DEFAULT_CASE_LABEL_SMOOTH_MIX = 0.0
DEFAULT_CASE_LABEL_POST_SIGMA = 0.0
DEFAULT_CASE_LABEL_BROAD_BLEND = 0.0
DEFAULT_CASE_LABEL_GAIN = 0.92
DEFAULT_CASE_LABEL_GAMMA = 0.92
DEFAULT_CASE_SAMPLES_PER_TX = 16_000_000
DEFAULT_CASE_MAX_DEPTH = 2
DEFAULT_CASE_DB_MIN = -111.25
DEFAULT_CASE_DB_MAX = -52.0


def compute_dataset_thresholds(
    tx_power_dbm: float,
    bandwidth_hz: float,
    noise_psd_dbm_per_hz: float,
    noise_figure_db: float,
    dataset_path_gain_max_db: float = DB_MAX,
) -> tuple[float, float, float]:
    noise_power_dbm = noise_psd_dbm_per_hz + 10.0 * math.log10(bandwidth_hz) + noise_figure_db
    pl_threshold_db = -tx_power_dbm + noise_power_dbm
    analytic_trunc_db = pl_threshold_db - (dataset_path_gain_max_db - pl_threshold_db) / 4.0
    return noise_power_dbm, pl_threshold_db, analytic_trunc_db


def parse_triplet(text: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values: x,y,z")
    return float(parts[0]), float(parts[1]), float(parts[2])


def load_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def world_to_grid_case(
    x_m: float,
    y_m: float,
    map_size_px: int,
    world_size_m: float,
    cell_size_m: float,
) -> tuple[int, int]:
    col = int(math.floor(x_m / cell_size_m))
    row = int(math.floor(y_m / cell_size_m))
    row = max(0, min(map_size_px - 1, row))
    col = max(0, min(map_size_px - 1, col))
    return row, col


def encode_tx_value_case(z_m: float) -> np.uint8:
    from ..blender_scene.paris.france_success_dataset import encode_tx_value

    return encode_tx_value(z_m)


def export_case_dataset_triplet(
    output_dir: Path,
    scene_id: str,
    tx_id: str,
    buildings_u8_internal: np.ndarray,
    export_cell_size_m: float,
    world_size_m: float,
    tx_position: tuple[float, float, float],
    label_db: np.ndarray,
    db_min: float,
    db_max: float,
    label_smooth_sigma: float,
    label_smooth_mix: float,
    label_db_offset: float,
    label_post_sigma: float,
    label_gain: float,
    label_gamma: float,
    label_broad_blend: float,
    label_top_gain: float,
    label_bottom_gain: float,
) -> list[str]:
    from ..blender_scene.paris.france_success_dataset import db_to_uint8

    png_root = output_dir / "png"
    buildings_dir = png_root / "buildingsWHeight"
    tx_dir = png_root / "antennasWHeight"
    gain_dir = output_dir / "gain"
    for path in (buildings_dir, tx_dir, gain_dir):
        ensure_dir(path)

    building_mask = buildings_u8_internal > 0
    tx_u8 = np.zeros_like(buildings_u8_internal, dtype=np.uint8)
    tx_row, tx_col = world_to_grid_case(
        tx_position[0],
        tx_position[1],
        map_size_px=buildings_u8_internal.shape[0],
        world_size_m=world_size_m,
        cell_size_m=export_cell_size_m,
    )
    tx_u8[tx_row, tx_col] = encode_tx_value_case(tx_position[2])
    raw_label_db = np.clip(label_db + float(label_db_offset), db_min, db_max).astype(np.float32)
    raw_label_u8 = db_to_uint8(raw_label_db, db_min=db_min, db_max=db_max)
    raw_label_u8[building_mask] = 0
    label_u8 = smooth_gain_uint8(
        raw_label_u8,
        building_mask=building_mask,
        sigma=label_smooth_sigma,
        mix=label_smooth_mix,
        post_sigma=label_post_sigma,
        gain=label_gain,
        gamma=label_gamma,
        broad_blend=label_broad_blend,
        top_gain=label_top_gain,
        bottom_gain=label_bottom_gain,
    )

    buildings_img = np.flipud(buildings_u8_internal)
    tx_img = np.flipud(tx_u8)
    gain_img = np.flipud(label_u8)
    tx_row_img = buildings_img.shape[0] - 1 - tx_row

    buildings_path = buildings_dir / f"{scene_id}.png"
    tx_path = tx_dir / f"{scene_id}_{tx_id}.png"
    gain_path = gain_dir / f"{scene_id}_{tx_id}.png"
    from PIL import Image

    Image.fromarray(buildings_img).save(buildings_path)
    Image.fromarray(tx_img).save(tx_path)
    Image.fromarray(gain_img).save(gain_path)
    return [
        f"dataset_buildings_path={buildings_path}",
        f"dataset_tx_path={tx_path}",
        f"dataset_gain_path={gain_path}",
        f"dataset_tx_image_rc=({tx_row_img},{tx_col})",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Sionna RT on the synthetic case-study scene.")
    # 基础输入输出路径与场景元信息。
    parser.add_argument("--scene", type=Path, default=DEFAULT_OUTPUT_XML)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--scene-id", type=str, default=None)
    parser.add_argument("--tx-id", type=str, default=None)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--building-map-path", type=Path, default=DEFAULT_BUILDING_MAP_PATH)
    parser.add_argument("--tx-position", type=parse_triplet, default=None)
    # 无线物理与数值仿真参数。
    parser.add_argument("--frequency-hz", type=float, default=FREQUENCY_HZ)
    parser.add_argument("--tx-power-dbm", type=float, default=TX_POWER_DBM)
    parser.add_argument("--bandwidth-hz", type=float, default=BANDWIDTH_HZ)
    parser.add_argument("--noise-psd-dbm-per-hz", type=float, default=NOISE_PSD_DBM_PER_HZ)
    parser.add_argument("--noise-figure-db", type=float, default=NOISE_FIGURE_DB)
    parser.add_argument("--cell-size-m", type=float, default=PIXEL_LENGTH_M)
    parser.add_argument("--sim-cell-size-m", type=float, default=DEFAULT_CASE_SIM_CELL_SIZE_M)
    parser.add_argument("--map-size-px", type=int, default=MAP_SIZE_PX)
    parser.add_argument("--world-size-m", type=float, default=None)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_CASE_MAX_DEPTH)
    parser.add_argument("--samples-per-tx", type=int, default=DEFAULT_CASE_SAMPLES_PER_TX)
    parser.add_argument("--reflection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diffraction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scattering", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--edge-diffraction", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--building-scattering-coefficient", type=float, default=0.0)
    # 导出 gain 标签前的量化与平滑参数。
    parser.add_argument("--db-min", type=float, default=DEFAULT_CASE_DB_MIN)
    parser.add_argument("--db-max", type=float, default=DEFAULT_CASE_DB_MAX)
    parser.add_argument("--label-smooth-sigma", type=float, default=DEFAULT_CASE_LABEL_SMOOTH_SIGMA)
    parser.add_argument("--label-smooth-mix", type=float, default=DEFAULT_CASE_LABEL_SMOOTH_MIX)
    parser.add_argument("--label-db-offset", type=float, default=DEFAULT_LABEL_DB_OFFSET)
    parser.add_argument("--label-post-sigma", type=float, default=DEFAULT_CASE_LABEL_POST_SIGMA)
    parser.add_argument("--label-gain", type=float, default=DEFAULT_CASE_LABEL_GAIN)
    parser.add_argument("--label-gamma", type=float, default=DEFAULT_CASE_LABEL_GAMMA)
    parser.add_argument("--label-broad-blend", type=float, default=DEFAULT_CASE_LABEL_BROAD_BLEND)
    parser.add_argument("--label-top-gain", type=float, default=DEFAULT_LABEL_TOP_GAIN)
    parser.add_argument("--label-bottom-gain", type=float, default=DEFAULT_LABEL_BOTTOM_GAIN)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = load_metadata(args.metadata_path)
    cfg = metadata["config"]
    scene_id = args.scene_id or cfg["scene_id"]
    tx_id = args.tx_id or cfg["tx_id"]
    tx_position = args.tx_position or tuple(metadata["tx_position_m"])
    world_size_m = float(args.world_size_m or cfg["world_size_m"])
    sim_cell_size_m = args.sim_cell_size_m if args.sim_cell_size_m is not None else args.cell_size_m
    ensure_dir(args.output_dir)

    from sionna.rt import PlanarArray, RadioMapSolver, Transmitter, load_scene

    scene = load_scene(str(args.scene))
    set_optional_attr(scene, "frequency", args.frequency_hz)
    set_optional_attr(scene, "bandwidth", args.bandwidth_hz)
    if args.building_scattering_coefficient > 0.0:
        for obj_name, obj in getattr(scene, "objects", {}).items():
            if obj_name == "ground":
                continue
            radio_material = getattr(obj, "radio_material", None)
            if radio_material is not None and hasattr(radio_material, "scattering_coefficient"):
                set_optional_attr(radio_material, "scattering_coefficient", float(args.building_scattering_coefficient))
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    tx = Transmitter(name="case_tx", position=list(tx_position), orientation=[0.0, 0.0, 0.0], power_dbm=args.tx_power_dbm)
    set_optional_attr(tx, "power_dbm", args.tx_power_dbm)
    scene.add(tx)

    solver = RadioMapSolver()
    radio_map = solver(
        scene,
        center=[world_size_m / 2.0, world_size_m / 2.0, RX_HEIGHT_M],
        orientation=[0.0, 0.0, 0.0],
        size=[world_size_m, world_size_m],
        cell_size=[sim_cell_size_m, sim_cell_size_m],
        samples_per_tx=args.samples_per_tx,
        max_depth=args.max_depth,
        los=True,
        specular_reflection=args.reflection,
        diffuse_reflection=args.scattering,
        refraction=False,
        diffraction=args.diffraction,
        edge_diffraction=args.edge_diffraction,
    )
    raw_metric = get_metric_array(radio_map, metric_name="path_gain")
    raw_metric_resized = resize_metric_to_export_grid(
        raw_metric.astype(np.float32),
        target_h=args.map_size_px,
        target_w=args.map_size_px,
        sim_cell_size_m=sim_cell_size_m,
        export_cell_size_m=args.cell_size_m,
    )
    metric_db = linear_metric_to_db(raw_metric_resized)
    rx_power_dbm = metric_db + args.tx_power_dbm
    noise_power_dbm, pl_threshold_db, analytic_trunc_db = compute_dataset_thresholds(
        tx_power_dbm=args.tx_power_dbm,
        bandwidth_hz=args.bandwidth_hz,
        noise_psd_dbm_per_hz=args.noise_psd_dbm_per_hz,
        noise_figure_db=args.noise_figure_db,
        dataset_path_gain_max_db=DB_MAX,
    )
    snr_db = rx_power_dbm - noise_power_dbm
    buildings_u8_internal = load_buildings_u8(args.building_map_path, args.map_size_px, args.map_size_px)

    np.save(args.output_dir / "path_gain_db.npy", metric_db)
    np.save(args.output_dir / "rx_power_dbm.npy", rx_power_dbm)
    np.save(args.output_dir / "snr_db.npy", snr_db)

    dataset_lines = export_case_dataset_triplet(
        args.output_dir,
        scene_id=scene_id,
        tx_id=tx_id,
        buildings_u8_internal=buildings_u8_internal,
        export_cell_size_m=args.cell_size_m,
        world_size_m=world_size_m,
        tx_position=tx_position,
        label_db=metric_db,
        db_min=args.db_min,
        db_max=args.db_max,
        label_smooth_sigma=args.label_smooth_sigma,
        label_smooth_mix=args.label_smooth_mix,
        label_db_offset=args.label_db_offset,
        label_post_sigma=args.label_post_sigma,
        label_gain=args.label_gain,
        label_gamma=args.label_gamma,
        label_broad_blend=args.label_broad_blend,
        label_top_gain=args.label_top_gain,
        label_bottom_gain=args.label_bottom_gain,
    )

    stats_lines = [
        f"scene={args.scene}",
        f"scene_id={scene_id}",
        f"tx_id={tx_id}",
        f"tx_position_m={tx_position}",
        f"world_size_m={world_size_m:.2f}",
        f"tx_power_dbm={args.tx_power_dbm:.2f}",
        f"frequency_hz={args.frequency_hz:.0f}",
        f"bandwidth_hz={args.bandwidth_hz:.0f}",
        f"noise_power_dbm={noise_power_dbm:.2f}",
        f"pl_threshold_db={pl_threshold_db:.2f}",
        f"analytic_pl_trunc_db={analytic_trunc_db:.2f}",
        f"dataset_export_db_range=[{args.db_min:.2f},{args.db_max:.2f}]",
        f"export_cell_size_m={args.cell_size_m:.2f}",
        f"sim_cell_size_m={sim_cell_size_m:.2f}",
        f"building_map_path={args.building_map_path}",
        f"building_scattering_coefficient={args.building_scattering_coefficient:.4f}",
        metric_stats("building_gray_u8", buildings_u8_internal[buildings_u8_internal > 0] if np.any(buildings_u8_internal > 0) else buildings_u8_internal),
        metric_stats("path_gain_db", metric_db),
        metric_stats("rx_power_dbm", rx_power_dbm),
        metric_stats("snr_db", snr_db),
        *dataset_lines,
    ]
    stats_path = args.output_dir / "stats_case_study.txt"
    stats_path.write_text("\n".join(stats_lines) + "\n", encoding="utf-8")
    print("\n".join(stats_lines))


if __name__ == "__main__":
    main()
