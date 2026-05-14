"""注释
命令示例:
python -m py_compile visualization/live_step_renderer.py

参数含义:
- `render_roi_coverage(...)`: 根据当前地图和站点集合生成 RoI 覆盖率二值图。
- `render_deployment_prediction(...)`: 根据当前地图和站点集合生成当前部署效果预测图。
- `_get_predictor(...)`: 复用已加载的代理模型，避免每个 step 重复加载权重。
- `preload_predictor(...)`: 提前加载指定设备上的代理模型，减少首帧延迟。

逻辑说明:
本文件负责把实时 step 状态渲染成两张图。为保证 Web 版逐步刷新，它在服务进程内缓存代理模型，后续 step 直接复用，
避免每次图片刷新都重新起解释器并重新加载模型。
"""

from __future__ import annotations

import os
from pathlib import Path
import yaml

_CACHE_ROOT = Path("/private/tmp/matplotlib_cache_visualization")
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from Autobs.compare_initialization_methods import LocalSurrogatePredictor
from Autobs.render_init_compare_figure import compute_pixelwise_fields, load_building_mask
from Autobs.utils import load_map_normalized
from visualization.runtime_config import resolve_local_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REACT_CONFIG = yaml.safe_load((PROJECT_ROOT / "ReAct" / "base_config.yaml").read_text(encoding="utf-8")) or {}
REACT_MODEL_CFG = REACT_CONFIG.get("surrogate_models", {}) if isinstance(REACT_CONFIG, dict) else {}
_PREDICTOR_CACHE: dict[tuple[str, str], LocalSurrogatePredictor] = {}
_AVAILABLE_FONT_NAMES = {item.name for item in font_manager.fontManager.ttflist}


def _resolve_model_path(eval_model: str) -> Path:
    config = REACT_MODEL_CFG.get(eval_model) if isinstance(REACT_MODEL_CFG, dict) else None
    if not isinstance(config, dict):
        raise KeyError(f"Unknown eval_model: {eval_model}")
    weights_rel = config.get("weights_path")
    if not weights_rel:
        raise KeyError(f"Missing weights_path for eval_model={eval_model}")
    return (PROJECT_ROOT / "ReAct" / weights_rel).resolve()


def _configure_matplotlib() -> None:
    serif_candidates = ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]
    available_serif = [name for name in serif_candidates if name in _AVAILABLE_FONT_NAMES] or ["DejaVu Serif"]
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = available_serif
    plt.rcParams["axes.unicode_minus"] = False


def _normalize_render_device(render_device: str | None) -> str:
    if render_device and render_device.strip():
        return render_device.strip()
    return resolve_local_device()


def _get_predictor(eval_model: str, render_device: str | None = None) -> LocalSurrogatePredictor:
    device_name = _normalize_render_device(render_device)
    cache_key = (eval_model, device_name)
    predictor = _PREDICTOR_CACHE.get(cache_key)
    if predictor is None:
        predictor = LocalSurrogatePredictor(str(_resolve_model_path(eval_model)), eval_model, device_name)
        _PREDICTOR_CACHE[cache_key] = predictor
    return predictor


def preload_predictor(eval_model: str, render_device: str | None = None) -> None:
    _get_predictor(eval_model, render_device)


def _tx_points_from_sites(sites: list[dict]) -> np.ndarray:
    if not sites:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([[float(site["col"]), float(site["row"])] for site in sites], dtype=np.float32)


def _tx_locs_from_sites(sites: list[dict]) -> list[tuple[int, int]]:
    return [(int(site["row"]), int(site["col"])) for site in sites]


def _metrics_text(sites: list[dict], fields: dict[str, object]) -> str:
    coverage_binary = np.asarray(fields["coverage_binary"], dtype=np.uint8)
    roi_mask = np.asarray(fields["eval_mask"], dtype=bool)
    coverage = float(coverage_binary[roi_mask].mean()) if roi_mask.any() else 0.0
    capacity_display = np.asarray(fields["capacity_display"], dtype=np.float32)
    se_value = float(np.nanmean(capacity_display[roi_mask])) if roi_mask.any() else 0.0
    if np.isnan(se_value):
        se_value = 0.0
    lines = [
        f"Sites={len(sites)}",
        f"Coverage={coverage * 100:.2f}%",
        f"SE={se_value:.2f} bps/Hz",
    ]
    return "\n".join(lines)


def _annotate_sites(ax, tx_points: np.ndarray, sites: list[dict]) -> None:
    if tx_points.size == 0:
        return
    ax.scatter(
        tx_points[:, 0],
        tx_points[:, 1],
        marker="^",
        s=92,
        c="#C00000",
        edgecolors="#1F1F1F",
        linewidths=0.9,
        zorder=5,
        clip_on=False,
    )
    for point_idx, ((x, y), site) in enumerate(zip(tx_points, sites, strict=True), start=1):
        row_int = int(site["row"])
        col_int = int(site["col"])
        ax.annotate(
            f"S{point_idx}({row_int},{col_int})",
            xy=(x, y),
            xytext=(6, -8),
            textcoords="offset points",
            fontsize=9.2,
            color="#1F1F1F",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "alpha": 0.92, "linewidth": 0.45},
            zorder=6,
            annotation_clip=False,
        )


def _style_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#808080")


def _compute_fields(
    map_path: Path,
    sites: list[dict],
    eval_model: str,
    render_device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    pixel_map = load_map_normalized(map_path)
    building_mask = load_building_mask(map_path, pixel_map.shape)
    predictor = _get_predictor(eval_model, render_device)
    fields = compute_pixelwise_fields(
        pixel_map,
        building_mask,
        _tx_locs_from_sites(sites),
        predictor,
        coverage_threshold_db=-117.0,
        noise_coefficient_db=10.0,
    )
    return pixel_map, building_mask, fields


def render_roi_coverage(
    map_path: Path,
    sites: list[dict],
    output_path: Path,
    eval_model: str,
    render_device: str | None = None,
) -> None:
    _configure_matplotlib()
    pixel_map, _building_mask, fields = _compute_fields(map_path, sites, eval_model, render_device)
    tx_points = _tx_points_from_sites(sites)
    coverage_cmap = ListedColormap(["#203864", "#70AD47", "#D9D9D9"])

    fig = plt.figure(figsize=(6.2, 6.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.12], hspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[1, 0])

    ax.imshow(np.asarray(fields["coverage_display"]), cmap=coverage_cmap, vmin=0, vmax=2, interpolation="nearest")
    _annotate_sites(ax, tx_points, sites)
    _style_axis(ax)
    ax.set_title("RoI Coverage Binary Map", fontsize=15, pad=10)

    legend_ax.axis("off")
    legend_handles = [
        Patch(facecolor="#70AD47", edgecolor="none", label="Covered"),
        Patch(facecolor="#203864", edgecolor="none", label="Uncovered"),
        Patch(facecolor="#D9D9D9", edgecolor="none", label="Masked"),
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=10,
        prop={"family": plt.rcParams["font.serif"][0], "size": 9},
        handlelength=1.4,
        columnspacing=1.0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_deployment_prediction(
    map_path: Path,
    sites: list[dict],
    output_path: Path,
    eval_model: str,
    render_device: str | None = None,
) -> None:
    _configure_matplotlib()
    pixel_map, _building_mask, fields = _compute_fields(map_path, sites, eval_model, render_device)
    tx_points = _tx_points_from_sites(sites)
    eval_mask = np.asarray(fields["eval_mask"], dtype=bool)
    raw_rmnet_display = np.asarray(fields["rmnet_display"], dtype=np.float32)
    valid_values = raw_rmnet_display[eval_mask]
    if valid_values.size == 0:
        vmin = -162.0
        vmax = -161.0
    else:
        vmin = float(np.min(valid_values))
        vmax = float(np.max(valid_values))
    rmnet_display = np.full(pixel_map.shape, vmin, dtype=np.float32)
    rmnet_display[eval_mask] = raw_rmnet_display[eval_mask]
    if vmax <= vmin:
        vmax = vmin + 1.0

    fig = plt.figure(figsize=(6.2, 6.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.12], hspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    cbar_ax = fig.add_subplot(gs[1, 0])

    im = ax.imshow(rmnet_display, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    _annotate_sites(ax, tx_points, sites)
    _style_axis(ax)
    ax.set_title(f"{eval_model.upper()} Prediction", fontsize=15, pad=10)

    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Predicted path gain(dB)", fontsize=10, labelpad=4)
    cbar.ax.xaxis.label.set_family(plt.rcParams["font.serif"][0])
    cbar.ax.tick_params(labelsize=8, pad=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_capacity_heatmap(
    map_path: Path,
    sites: list[dict],
    output_path: Path,
    eval_model: str,
    render_device: str | None = None,
) -> None:
    _configure_matplotlib()
    pixel_map, _building_mask, fields = _compute_fields(map_path, sites, eval_model, render_device)
    tx_points = _tx_points_from_sites(sites)
    eval_mask = np.asarray(fields["eval_mask"], dtype=bool)
    capacity_display = np.asarray(fields["capacity_display"], dtype=np.float32)
    valid_values = capacity_display[eval_mask]
    if valid_values.size == 0:
        vmin = 0.0
        vmax = 1.0
    else:
        vmin = float(np.nanmin(valid_values))
        vmax = float(np.nanmax(valid_values))
    heatmap = np.full(pixel_map.shape, np.nan, dtype=np.float32)
    heatmap[eval_mask] = capacity_display[eval_mask]
    if vmax <= vmin:
        vmax = vmin + 1.0
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#D9D9D9")

    fig = plt.figure(figsize=(6.2, 6.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.12], hspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    cbar_ax = fig.add_subplot(gs[1, 0])

    im = ax.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    _annotate_sites(ax, tx_points, sites)
    _style_axis(ax)
    ax.set_title("Channel Capacity Heatmap", fontsize=15, pad=10)

    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Capacity (Mbps)", fontsize=10, labelpad=4)
    cbar.ax.xaxis.label.set_family(plt.rcParams["font.serif"][0])
    cbar.ax.tick_params(labelsize=8, pad=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_metric_history_trend(
    metric_history: list[dict],
    output_path: Path,
) -> None:
    _configure_matplotlib()
    labels = [str(item.get("step", "")) for item in metric_history]
    xs = np.arange(len(metric_history), dtype=np.float32)
    coverage = np.asarray([float(item.get("coverage", 0.0) or 0.0) * 100.0 for item in metric_history], dtype=np.float32)
    se_values = np.asarray([float(item.get("spectral_efficiency", 0.0) or 0.0) for item in metric_history], dtype=np.float32)

    fig, ax1 = plt.subplots(figsize=(6.4, 4.6))
    ax2 = ax1.twinx()
    ax1.plot(xs, coverage, color="#2F75B5", marker="o", linewidth=2.2, label="Coverage")
    ax2.plot(xs, se_values, color="#C55A11", marker="s", linewidth=2.2, label="SE")
    ax1.fill_between(xs, coverage, color="#2F75B5", alpha=0.12)
    ax2.fill_between(xs, se_values, color="#C55A11", alpha=0.10)
    ax1.set_ylabel("Coverage (%)", color="#2F75B5", fontsize=10)
    ax2.set_ylabel("SE (bps/Hz)", color="#C55A11", fontsize=10)
    ax1.set_xlabel("Deployment step", fontsize=10)
    ax1.set_title("Coverage and SE Trend", fontsize=14, pad=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.28)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels if labels else ["init"], fontsize=9)
    ax1.tick_params(axis="y", labelcolor="#2F75B5", labelsize=9)
    ax2.tick_params(axis="y", labelcolor="#C55A11", labelsize=9)
    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
