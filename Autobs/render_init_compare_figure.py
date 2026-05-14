"""注释
命令示例:
python -m Autobs.render_init_compare_figure \
  --image /Users/epiphanyer/Desktop/coding/paper_experiment/dataset/png/buildingsWHeight/584.png \
  --output /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/outputs/thesis_figures/init_compare_584.svg

参数含义:
- --image: 单张待可视化的建筑高度图样本路径。
- --output: 导出的论文图路径，默认输出为 svg。
- --model-path: 代理传播模型权重路径。
- --network-type: 代理传播模型类型，默认使用 rmnet。
- --pretrain-module-state: 第一阶段策略网络参数路径。
- --rerank-module-state: 第二阶段策略网络参数路径。
- --k-max: 初始化部署的站点数。
- --device: 推理设备，支持 auto/cpu/cuda/mps。
- --coverage-target: 评价时使用的覆盖率目标，仅影响评分一致性，不影响二值覆盖图阈值。
- --spectral-efficiency-target: 评价时使用的平均频谱效率目标，仅影响评分一致性。
- --w1、--w2: 评分项中的覆盖率与频谱效率惩罚权重。
- --coverage-threshold-db: 覆盖判定门限，单位 dBm。
- --noise-coefficient-db: 噪声系数，单位 dB。

逻辑说明:
- 该脚本复用现有初始化策略与代理评估代码，在同一张地图上分别生成 Pretrain 与 Rerank 的初始化布局。
- 随后基于逐站点路径增益图计算RMNet预测传播结果、最强接收功率对应的覆盖二值图以及逐像素容量热图。
- 最终输出两行四列对比图，并在各列右侧分别放置对应的色条或图例。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_CACHE_ROOT = Path("/private/tmp/matplotlib_cache_autobs")
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

from Autobs.compare_initialization_methods import (
    LocalSurrogatePredictor,
    build_parser as build_compare_parser,
    load_map_normalized,
    run_module_state_init,
)
from Autobs.paths import DEFAULT_RMNET_WEIGHTS, PACKAGE_ROOT
from Autobs.utils import (
    BITS_PER_MEGABIT,
    CHANNEL_BANDWIDTH_HZ,
    DEFAULT_COVERAGE_TARGET,
    DEFAULT_COVERAGE_THRESHOLD_DB,
    DEFAULT_NOISE_COEFFICIENT_DB,
    DEFAULT_SPECTRAL_EFFICIENCY_TARGET,
    DEFAULT_W1,
    DEFAULT_W2,
    TX_POWER_DBM,
    build_roi_mask,
    compute_total_noise_power_mw,
    get_site_pathgain_maps,
)


DEFAULT_IMAGE = PACKAGE_ROOT.parent / "dataset" / "png" / "buildingsWHeight" / "584.png"
DEFAULT_OUTPUT = PACKAGE_ROOT / "outputs" / "thesis_figures" / "init_compare_584.svg"
DEFAULT_PRETRAIN = PACKAGE_ROOT / "outputs" / "pretrain" / "best_module_state.pt"
DEFAULT_RERANK = PACKAGE_ROOT / "outputs" / "rerank" / "best_module_state.pt"
DEFAULT_METHOD_LABELS = ("Pretrain", "Rerank")
DEFAULT_BUILDING_MASK_DIR = PACKAGE_ROOT.parent / "dataset" / "png" / "buildings_complete"


def build_parser() -> argparse.ArgumentParser:
    compare_defaults = build_compare_parser().parse_args([])
    parser = argparse.ArgumentParser(description="Render an initialization comparison figure for one map.")
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_RMNET_WEIGHTS))
    parser.add_argument("--network-type", type=str, default="rmnet")
    parser.add_argument("--pretrain-module-state", type=str, default=str(DEFAULT_PRETRAIN))
    parser.add_argument("--rerank-module-state", type=str, default=str(DEFAULT_RERANK))
    parser.add_argument("--k-max", type=int, default=1)
    parser.add_argument("--policy-version", type=str, default=compare_defaults.policy_version)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--coverage-target", type=float, default=DEFAULT_COVERAGE_TARGET)
    parser.add_argument("--spectral-efficiency-target", type=float, default=DEFAULT_SPECTRAL_EFFICIENCY_TARGET)
    parser.add_argument("--w1", type=float, default=DEFAULT_W1)
    parser.add_argument("--w2", type=float, default=DEFAULT_W2)
    parser.add_argument("--coverage-threshold-db", type=float, default=DEFAULT_COVERAGE_THRESHOLD_DB)
    parser.add_argument("--noise-coefficient-db", type=float, default=DEFAULT_NOISE_COEFFICIENT_DB)
    return parser


def build_eval_namespace(args: argparse.Namespace) -> argparse.Namespace:
    namespace = argparse.Namespace()
    namespace.k_max = int(args.k_max)
    namespace.policy_version = str(args.policy_version)
    namespace.device = str(args.device)
    namespace.coverage_target = float(args.coverage_target)
    namespace.spectral_efficiency_target = float(args.spectral_efficiency_target)
    namespace.w1 = float(args.w1)
    namespace.w2 = float(args.w2)
    namespace.coverage_threshold_db = float(args.coverage_threshold_db)
    namespace.noise_coefficient_db = float(args.noise_coefficient_db)
    return namespace


def load_building_mask(image_path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    mask_path = DEFAULT_BUILDING_MASK_DIR / f"{image_path.stem}.png"
    if not mask_path.exists():
        return np.zeros(image_shape, dtype=bool)
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    if mask.shape != image_shape:
        raise ValueError(f"Building mask shape {mask.shape} does not match image shape {image_shape}")
    return mask > 0


def compute_pixelwise_fields(
    pixel_map: np.ndarray,
    building_mask: np.ndarray,
    tx_locs: list[tuple[int, int]],
    predictor: LocalSurrogatePredictor,
    coverage_threshold_db: float,
    noise_coefficient_db: float,
) -> dict[str, np.ndarray | float]:
    roi_mask = build_roi_mask(pixel_map)
    eval_mask = np.logical_and(roi_mask, ~building_mask)
    site_pathgain_db = get_site_pathgain_maps(pixel_map, tx_locs, pmnet=predictor)
    if site_pathgain_db.size == 0:
        strongest_pathgain_db = np.full_like(pixel_map, -162.0, dtype=np.float32)
        strongest_rx_power_dbm = TX_POWER_DBM + strongest_pathgain_db
        capacity_map = np.zeros_like(pixel_map, dtype=np.float32)
    else:
        strongest_pathgain_db = np.max(site_pathgain_db, axis=0).astype(np.float32)
        strongest_rx_power_dbm = (TX_POWER_DBM + strongest_pathgain_db).astype(np.float32)
        rx_power_dbm = TX_POWER_DBM + site_pathgain_db.astype(np.float64)
        total_rx_power_mw = np.sum(np.power(10.0, rx_power_dbm / 10.0), axis=0).astype(np.float64)
        strongest_rx_power_mw = np.power(10.0, strongest_rx_power_dbm.astype(np.float64) / 10.0)
        noise_power_mw = compute_total_noise_power_mw(noise_coefficient_db)
        interference_power_mw = np.maximum(total_rx_power_mw - strongest_rx_power_mw, 0.0)
        sinr_linear = strongest_rx_power_mw / np.maximum(interference_power_mw + noise_power_mw, 1e-30)
        capacity_map = (CHANNEL_BANDWIDTH_HZ * np.log2(1.0 + sinr_linear) / BITS_PER_MEGABIT).astype(np.float32)

    coverage_binary = np.logical_and(eval_mask, strongest_rx_power_dbm >= coverage_threshold_db)
    coverage_display = np.full(pixel_map.shape, 2, dtype=np.int16)
    coverage_display[eval_mask] = 0
    coverage_display[coverage_binary] = 1

    rmnet_display = strongest_pathgain_db.astype(np.float32)

    capacity_display = np.full(pixel_map.shape, np.nan, dtype=np.float32)
    capacity_display[eval_mask] = capacity_map[eval_mask]

    return {
        "roi_mask": roi_mask,
        "building_mask": building_mask,
        "eval_mask": eval_mask,
        "coverage_binary": coverage_binary.astype(np.uint8),
        "coverage_display": coverage_display,
        "rmnet_display": rmnet_display,
        "capacity_display": capacity_display,
        "capacity_mean_mbps": float(np.nanmean(capacity_display)),
    }


def render_figure(
    *,
    pixel_map: np.ndarray,
    building_mask: np.ndarray,
    pretrain_payload: dict[str, object],
    rerank_payload: dict[str, object],
    pretrain_fields: dict[str, np.ndarray | float],
    rerank_fields: dict[str, np.ndarray | float],
    output_path: Path,
) -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"

    coverage_cmap = ListedColormap(["#203864", "#70AD47", "#D9D9D9"])
    capacity_cmap = plt.get_cmap("magma").copy()
    capacity_cmap.set_bad("#D9D9D9")
    map_cmap = plt.get_cmap("gray")
    rmnet_cmap = plt.get_cmap("viridis").copy()
    rmnet_cmap.set_bad("#D9D9D9")

    pre_capacity = np.asarray(pretrain_fields["capacity_display"], dtype=np.float32)
    rerank_capacity = np.asarray(rerank_fields["capacity_display"], dtype=np.float32)
    vmax = float(np.nanmax(np.stack([np.nan_to_num(pre_capacity, nan=0.0), np.nan_to_num(rerank_capacity, nan=0.0)], axis=0)))
    vmax = max(vmax, 1e-6)

    pre_rmnet = np.asarray(pretrain_fields["rmnet_display"], dtype=np.float32)
    rerank_rmnet = np.asarray(rerank_fields["rmnet_display"], dtype=np.float32)
    rmnet_vmin = float(np.nanmin(np.stack([np.nan_to_num(pre_rmnet, nan=0.0), np.nan_to_num(rerank_rmnet, nan=0.0)], axis=0)))
    rmnet_vmax = float(np.nanmax(np.stack([np.nan_to_num(pre_rmnet, nan=0.0), np.nan_to_num(rerank_rmnet, nan=0.0)], axis=0)))

    fig = plt.figure(figsize=(18.4, 9.8))
    outer = GridSpec(
        4,
        4,
        figure=fig,
        height_ratios=[0.12, 1.0, 1.0, 0.20],
        width_ratios=[1.0, 1.0, 1.0, 1.0],
        hspace=0.16,
        wspace=0.018,
    )

    title_axes = [fig.add_subplot(outer[0, i]) for i in range(4)]
    row_label_axes = [fig.add_subplot(outer[1, 0]), fig.add_subplot(outer[2, 0])]
    axes = np.empty((2, 4), dtype=object)

    # 第1列只有主图
    axes[0, 0] = fig.add_subplot(outer[1, 0])
    axes[1, 0] = fig.add_subplot(outer[2, 0])

    # 第2/3/4列改成“上两行主图 + 第四行辅助元素”
    axes[0, 1] = fig.add_subplot(outer[1, 1])
    axes[1, 1] = fig.add_subplot(outer[2, 1])
    rmnet_bottom_ax = fig.add_subplot(outer[3, 1])

    axes[0, 2] = fig.add_subplot(outer[1, 2])
    axes[1, 2] = fig.add_subplot(outer[2, 2])
    cover_bottom_ax = fig.add_subplot(outer[3, 2])
    cover_bottom_ax.axis("off")

    axes[0, 3] = fig.add_subplot(outer[1, 3])
    axes[1, 3] = fig.add_subplot(outer[2, 3])
    cap_bottom_ax = fig.add_subplot(outer[3, 3])

    # 先固定整体子图布局，再基于最终坐标放置底部色条/图例，避免后续坐标漂移
    fig.subplots_adjust(left=0.035, right=0.995, top=0.972, bottom=0.055)

    payloads = (pretrain_payload, rerank_payload)
    fields = (pretrain_fields, rerank_fields)
    for row, (label, payload, field) in enumerate(zip(DEFAULT_METHOD_LABELS, payloads, fields, strict=True)):
        tx_points = np.asarray(payload["positions_xy"], dtype=np.float32)
        map_ax = axes[row, 0]
        rmnet_ax = axes[row, 1]
        coverage_ax = axes[row, 2]
        capacity_ax = axes[row, 3]

        map_ax.imshow(pixel_map, cmap=map_cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
        rmnet_im = rmnet_ax.imshow(
            pre_rmnet if row == 0 else rerank_rmnet,
            cmap=rmnet_cmap,
            vmin=rmnet_vmin,
            vmax=rmnet_vmax,
            interpolation="nearest",
        )
        coverage_ax.imshow(np.asarray(field["coverage_display"]), cmap=coverage_cmap, vmin=0, vmax=2, interpolation="nearest")
        capacity_im = capacity_ax.imshow(
            pre_capacity if row == 0 else rerank_capacity,
            cmap=capacity_cmap,
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )

        for ax in (map_ax, rmnet_ax, coverage_ax, capacity_ax):
            ax.scatter(
                tx_points[:, 0],
                tx_points[:, 1],
                marker="^",
                s=54,
                c="#C00000",
                edgecolors="white",
                linewidths=0.8,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("#808080")

        first_col = int(round(float(tx_points[0, 0])))
        first_row = int(round(float(tx_points[0, 1])))
        metric_text = (
            f"S1({first_row},{first_col})\n"
            f"Coverage={float(payload['coverage']):.4f}\n"
            f"Capacity={float(payload['channel_capacity_mbps']):.4f}Mbps"
        )
        map_ax.text(
            0.02,
            0.03,
            metric_text,
            transform=map_ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.0,
            color="#1F1F1F",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.9, "linewidth": 0.35},
        )

        for ax in (map_ax, rmnet_ax, coverage_ax, capacity_ax):
            for point_idx, (x, y) in enumerate(tx_points, start=1):
                col_int = int(round(float(x)))
                row_int = int(round(float(y)))
                ax.annotate(
                    f"S{point_idx}({row_int},{col_int})",
                    xy=(x, y),
                    xytext=(6, -8),
                    textcoords="offset points",
                    fontsize=8.5,
                    color="#1F1F1F",
                    bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "alpha": 0.88, "linewidth": 0.35},
                )

    title_texts = ["Original map", "RMNet prediction", "Coverage map", "Capacity heatmap"]
    for ax, text in zip(title_axes, title_texts, strict=True):
        ax.axis("off")
        ax.text(0.5, 0.35, text, ha="center", va="center", fontsize=15, fontfamily="Times New Roman")

    for ax, text in zip(row_label_axes, ["Pretrain", "Rerank"], strict=True):
        bbox = ax.get_position()
        ax.remove()
        label_x = max(0.022, bbox.x0 - 0.042)
        fig.text(
            label_x,
            0.5 * (bbox.y0 + bbox.y1),
            text,
            ha="center",
            va="center",
            rotation=90,
            fontsize=15,
            fontfamily="Times New Roman",
        )

    # 第2列和第4列色条、以及第3列图例统一放在对应列的下面
    rmnet_bottom_ax.set_axis_off()
    rmnet_ref_bbox = axes[1, 1].get_position()
    rmnet_bottom_bbox = rmnet_bottom_ax.get_position()
    rmnet_hcax = fig.add_axes(
        [
            rmnet_ref_bbox.x0,
            rmnet_bottom_bbox.y0 + rmnet_bottom_bbox.height * 0.32,
            rmnet_ref_bbox.width,
            rmnet_bottom_bbox.height * 0.26,
        ]
    )
    rmnet_colorbar = fig.colorbar(rmnet_im, cax=rmnet_hcax, orientation="horizontal")
    rmnet_colorbar.set_label("Predicted path gain(dB)", fontsize=10, labelpad=4)
    rmnet_colorbar.ax.xaxis.label.set_family("Times New Roman")
    rmnet_colorbar.ax.tick_params(labelsize=8, pad=1)

    cap_bottom_ax.set_axis_off()
    cap_ref_bbox = axes[1, 3].get_position()
    cap_bottom_bbox = cap_bottom_ax.get_position()
    cap_hcax = fig.add_axes(
        [
            cap_ref_bbox.x0,
            cap_bottom_bbox.y0 + cap_bottom_bbox.height * 0.32,
            cap_ref_bbox.width,
            cap_bottom_bbox.height * 0.26,
        ]
    )
    capacity_colorbar = fig.colorbar(capacity_im, cax=cap_hcax, orientation="horizontal")
    capacity_colorbar.set_label("Capacity(Mbps)", fontsize=10, labelpad=4)
    capacity_colorbar.ax.xaxis.label.set_family("Times New Roman")
    capacity_colorbar.ax.tick_params(labelsize=8, pad=1)

    legend_handles = [
        Patch(facecolor="#70AD47", edgecolor="none", label="Covered"),
        Patch(facecolor="#203864", edgecolor="none", label="Uncovered"),
        Patch(facecolor="#D9D9D9", edgecolor="none", label="Masked area"),
    ]
    cover_bottom_ax.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=3,
        frameon=False,
        fontsize=11,
        prop={"family": "Times New Roman", "size": 8},
        labelspacing=0.8,
        handlelength=1.4,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    legend = cover_bottom_ax.get_legend()
    if legend is not None:
        legend.set_clip_on(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    image_path = Path(args.image).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    predictor = LocalSurrogatePredictor(args.model_path, args.network_type, args.device)
    eval_args = build_eval_namespace(args)
    pixel_map = load_map_normalized(image_path)
    building_mask = load_building_mask(image_path, pixel_map.shape)

    pretrain_payload = run_module_state_init(
        method_name="pretrain_init",
        map_path=image_path,
        predictor=predictor,
        args=eval_args,
        module_state_path=args.pretrain_module_state,
    )
    rerank_payload = run_module_state_init(
        method_name="rerank_init",
        map_path=image_path,
        predictor=predictor,
        args=eval_args,
        module_state_path=args.rerank_module_state,
    )

    pretrain_tx_locs = [(int(y), int(x)) for x, y in pretrain_payload["positions_xy"]]
    rerank_tx_locs = [(int(y), int(x)) for x, y in rerank_payload["positions_xy"]]

    pretrain_fields = compute_pixelwise_fields(
        pixel_map,
        building_mask,
        pretrain_tx_locs,
        predictor,
        coverage_threshold_db=args.coverage_threshold_db,
        noise_coefficient_db=args.noise_coefficient_db,
    )
    rerank_fields = compute_pixelwise_fields(
        pixel_map,
        building_mask,
        rerank_tx_locs,
        predictor,
        coverage_threshold_db=args.coverage_threshold_db,
        noise_coefficient_db=args.noise_coefficient_db,
    )

    render_figure(
        pixel_map=pixel_map,
        building_mask=building_mask,
        pretrain_payload=pretrain_payload,
        rerank_payload=rerank_payload,
        pretrain_fields=pretrain_fields,
        rerank_fields=rerank_fields,
        output_path=output_path,
    )

    print(f"saved_svg={output_path}")
    print(f"pretrain_positions_xy={pretrain_payload['positions_xy']}")
    print(f"rerank_positions_xy={rerank_payload['positions_xy']}")
    print(f"pretrain_coverage={float(pretrain_payload['coverage']):.6f}")
    print(f"rerank_coverage={float(rerank_payload['coverage']):.6f}")
    print(f"pretrain_capacity_mbps={float(pretrain_payload['channel_capacity_mbps']):.6f}")
    print(f"rerank_capacity_mbps={float(rerank_payload['channel_capacity_mbps']):.6f}")


if __name__ == "__main__":
    main()
