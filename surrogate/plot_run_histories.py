"""注释
命令示例:
cd /Users/epiphanyer/Desktop/coding && python -m paper_experiment.surrogate.plot_run_histories \
  --runs-root /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/runs \
  --output-root /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/plots

若当前目录已经是 `/Users/epiphanyer/Desktop/coding/paper_experiment`，则使用:
python -m surrogate.plot_run_histories \
  --runs-root /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/runs \
  --output-root /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/plots

参数说明:
- --runs-root: 训练结果根目录，内部应包含 `pmnet_usc/16_...` 这类实验目录。
- --output-root: 聚合曲线图输出目录，脚本会写入 8 张 PNG 图。

脚本逻辑:
- 扫描 `surrogate/runs` 下 5 个模型在 USC 与 RadioMap3DSeer 上的 `history.json`。
- 按数据集分别生成 8 张总览图：`USC` 与 `RadioMap3DSeer` 各输出 Train Loss、Best Validation RMSE、Validation MAE、Validation R²。
- 保持学术风格绘图，并遵守“中文宋体、西文 Times New Roman”的既有字体约束。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "surrogate" / "runs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "surrogate" / "test" / "runs" / "training_curve_overview"
MODEL_ORDER = ("pmnet", "rmnet", "unet", "transunet", "radiounet")
DATASET_ORDER = ("usc", "radiomap3dseer")
MODEL_DISPLAY_NAMES = {
    "pmnet": "PMNet",
    "rmnet": "RMNet",
    "unet": "U-Net",
    "transunet": "TransUNet",
    "radiounet": "RadioUNet",
}
DATASET_DISPLAY_NAMES = {
    "usc": "USC",
    "radiomap3dseer": "RadioMap3DSeer",
}
PLOT_SPECS = (
    ("train_loss", "Train Loss", "Training Loss", "train_loss"),
    ("best_val_rmse", "Best Validation RMSE", "Best Validation RMSE", "best_val_rmse"),
    ("val_mae", "Validation MAE", "Validation MAE", "val_mae"),
    ("val_r2", "Validation R²", "Validation R²", "val_r2"),
)
HIGH_CONTRAST_COLORS = {
    "pmnet": "#4C78A8",
    "rmnet": "#F58518",
    "unet": "#54A24B",
    "transunet": "#B279A2",
    "radiounet": "#E45756",
}
BEST_MARKER_METRICS = {"best_val_rmse", "val_mae", "val_r2"}

try:
    from .utils import ACADEMIC_COLOR_CYCLE, configure_plot_style
except ImportError:
    from paper_experiment.surrogate.utils import ACADEMIC_COLOR_CYCLE, configure_plot_style


@dataclass(frozen=True)
class RunHistory:
    model_type: str
    dataset: str
    run_dir: Path
    history: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    """
    功能: 解析命令行参数并返回绘图配置。
    输入: 无显式函数参数，读取命令行中的 `--runs-root` 与 `--output-root`。
    输出: `argparse.Namespace`，包含结果目录与输出目录。
    示例: `cd /Users/epiphanyer/Desktop/coding && python -m paper_experiment.surrogate.plot_run_histories --runs-root ./paper_experiment/surrogate/runs`。
    时间: 2026-04-26。
    """
    parser = argparse.ArgumentParser(description="Plot aggregated surrogate training curves for all models.")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT), type=str)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), type=str)
    return parser.parse_args()


def discover_run_histories(runs_root: Path) -> list[RunHistory]:
    """
    功能: 扫描 runs 目录并收集每个模型-数据集组合的 `history.json`。
    输入: `runs_root` 为训练结果根目录路径。
    输出: `list[RunHistory]`，每项包含模型名、数据集名、实验目录与逐 epoch 历史。
    示例: `discover_run_histories(Path('paper_experiment/surrogate/runs'))`。
    时间: 2026-04-26。
    """
    run_histories: list[RunHistory] = []
    for model_type in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            parent_dir = runs_root / f"{model_type}_{dataset}"
            if not parent_dir.exists():
                continue
            candidate_dirs = sorted((path for path in parent_dir.iterdir() if path.is_dir()), key=lambda item: item.name)
            if not candidate_dirs:
                continue
            run_dir = candidate_dirs[-1]
            history_path = run_dir / "history.json"
            if not history_path.exists():
                raise FileNotFoundError(f"Missing history.json: {history_path}")
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                raise ValueError(f"history.json must be a list: {history_path}")
            run_histories.append(
                RunHistory(
                    model_type=model_type,
                    dataset=dataset,
                    run_dir=run_dir,
                    history=history,
                )
            )
    if not run_histories:
        raise FileNotFoundError(f"No run histories found under {runs_root}")
    return run_histories


def build_mpl_cache(output_root: Path) -> None:
    """
    功能: 为 matplotlib 配置本地缓存目录，避免运行时写入到不可控位置。
    输入: `output_root` 为图像输出根目录路径。
    输出: 无，副作用是设置 `MPLCONFIGDIR` 与 `XDG_CACHE_HOME` 环境变量。
    示例: `build_mpl_cache(Path('paper_experiment/surrogate/test/runs/training_curve_overview'))`。
    时间: 2026-04-26。
    """
    mpl_cache_dir = output_root / ".mpl-cache"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache_dir))


def get_color_map() -> dict[str, str]:
    """
    功能: 为 5 个模型生成稳定颜色映射，保证跨图一致性。
    输入: 无。
    输出: `dict[str, str]`，键为模型名，值为十六进制颜色。
    示例: `get_color_map()['pmnet']` 返回 PMNet 对应颜色。
    时间: 2026-04-26。
    """
    return {
        model_type: HIGH_CONTRAST_COLORS.get(model_type, ACADEMIC_COLOR_CYCLE[index % len(ACADEMIC_COLOR_CYCLE)])
        for index, model_type in enumerate(MODEL_ORDER)
    }


def compute_smooth_curve(values: list[float], window: int = 5) -> np.ndarray:
    """
    功能: 对单次训练历史做简单滑动平均平滑，提升科研绘图可读性。
    输入: `values` 为原始标量序列，`window` 为滑动窗口长度。
    输出: `np.ndarray`，长度与输入一致的平滑序列。
    示例: `compute_smooth_curve([0.5, 0.4, 0.3, 0.2], window=3)`。
    时间: 2026-04-26。
    """
    if not values:
        return np.asarray([], dtype=float)
    if window <= 1 or len(values) < 3:
        return np.asarray(values, dtype=float)
    effective_window = max(3, min(window, len(values) if len(values) % 2 == 1 else len(values) - 1))
    if effective_window <= 1:
        return np.asarray(values, dtype=float)
    radius = effective_window // 2
    padded = np.pad(np.asarray(values, dtype=float), (radius, radius), mode="edge")
    kernel = np.ones(effective_window, dtype=float) / float(effective_window)
    return np.convolve(padded, kernel, mode="valid")


def get_best_epoch_index(metric_key: str, values: list[float]) -> int:
    """
    功能: 根据指标方向返回最佳 epoch 在序列中的位置索引。
    输入: `metric_key` 为指标名，`values` 为该指标逐 epoch 数值序列。
    输出: `int`，最佳点对应的 0-based 索引。
    示例: `get_best_epoch_index('best_val_rmse', [0.2, 0.1, 0.12])`。
    时间: 2026-04-26。
    """
    if metric_key == "val_r2":
        return int(np.argmax(np.asarray(values, dtype=float)))
    return int(np.argmin(np.asarray(values, dtype=float)))


def annotate_rmnet_best(ax: Any, xs: list[int], ys: list[float], metric_key: str, color: str) -> None:
    """
    功能: 在验证指标图中标注 RMNet 的最佳性能点，强调其最优结果。
    输入: `ax` 为 matplotlib 轴对象，`xs`/`ys` 为曲线坐标，`metric_key` 为指标名，`color` 为标注颜色。
    输出: 无，副作用是在坐标轴上添加最佳点圆标记与文字注释。
    示例: `annotate_rmnet_best(ax, [1,2,3], [0.05,0.04,0.03], 'best_val_rmse', '#F58518')`。
    时间: 2026-04-26。
    """
    if metric_key not in BEST_MARKER_METRICS or not xs:
        return
    best_index = get_best_epoch_index(metric_key, ys)
    best_x = xs[best_index]
    best_y = ys[best_index]
    metric_label = "Best RMNet"
    value_text = f"{metric_label}: {best_y:.4f}" if metric_key != "val_r2" else f"{metric_label}: {best_y:.4f}"
    ax.scatter([best_x], [best_y], s=120, color=color, edgecolors="black", linewidths=0.9, zorder=6)
    ax.annotate(
        value_text,
        xy=(best_x, best_y),
        xytext=(14, 10 if metric_key == "val_r2" else -16),
        textcoords="offset points",
        fontsize=11,
        color=color,
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2, "shrinkA": 2, "shrinkB": 4},
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": color, "alpha": 0.92},
    )


def filter_run_histories_by_dataset(run_histories: list[RunHistory], dataset: str) -> list[RunHistory]:
    """
    功能: 从全部实验历史中筛出指定数据集对应的曲线集合。
    输入: `run_histories` 为全部实验历史，`dataset` 为目标数据集键名，例如 `usc`。
    输出: `list[RunHistory]`，仅包含指定数据集的实验历史。
    示例: `filter_run_histories_by_dataset(histories, 'usc')`。
    时间: 2026-04-26。
    """
    return [run_history for run_history in run_histories if run_history.dataset == dataset]


def plot_metric_figure(run_histories: list[RunHistory], dataset: str, metric_key: str, title: str, ylabel: str, output_path: Path) -> None:
    """
    功能: 将单个数据集上所有模型的同类指标画到一张学术风格曲线图中。
    输入: `run_histories` 为所有实验历史，`dataset` 为目标数据集键名，`metric_key` 为指标键名，`title`/`ylabel` 为图标题与纵轴标题，`output_path` 为 PNG 路径。
    输出: 无，副作用是在 `output_path` 写出图像文件。
    示例: `plot_metric_figure(histories, 'usc', 'train_loss', 'USC Train Loss', 'Training Loss', Path('usc_train_loss.png'))`。
    时间: 2026-04-26。
    """
    import matplotlib

    matplotlib.use("Agg")
    configure_plot_style()
    import matplotlib.pyplot as plt

    color_map = get_color_map()
    dataset_histories = filter_run_histories_by_dataset(run_histories, dataset)
    fig, ax = plt.subplots(figsize=(9.4, 5.8))

    ax.set_facecolor("#FFFFFF")
    ax.minorticks_on()
    ax.grid(True, which="major", color="#C9D1D9", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.grid(True, which="minor", color="#E5E7EB", linestyle="-", linewidth=0.35, alpha=0.55)

    for run_history in dataset_histories:
        xs = [int(row["epoch"]) for row in run_history.history if row.get("epoch") is not None and row.get(metric_key) is not None]
        ys = [float(row[metric_key]) for row in run_history.history if row.get("epoch") is not None and row.get(metric_key) is not None]
        if not xs:
            continue

        label = MODEL_DISPLAY_NAMES[run_history.model_type]
        color = color_map[run_history.model_type]
        smooth_ys = compute_smooth_curve(ys, window=5 if metric_key == "train_loss" else 3)

        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.1,
            alpha=0.20,
            linestyle="-",
            zorder=1,
        )
        ax.plot(
            xs,
            smooth_ys,
            label=label,
            color=color,
            linestyle="-",
            marker="s" if run_history.model_type == "rmnet" else "o",
            markevery=max(1, len(xs) // 8),
            linewidth=3.0 if run_history.model_type == "rmnet" and metric_key in BEST_MARKER_METRICS else 2.4,
            markersize=6.2 if run_history.model_type == "rmnet" else 5.0,
            alpha=0.98,
            zorder=4 if run_history.model_type == "rmnet" else 3,
        )

        if run_history.model_type == "rmnet":
            annotate_rmnet_best(ax, xs, ys, metric_key, color)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    legend_kwargs = {
        "ncol": 1,
        "title": "Architecture",
        "frameon": True,
        "fancybox": True,
        "framealpha": 0.95,
        "borderpad": 0.5,
        "handlelength": 2.2,
    }
    if metric_key == "val_r2":
        legend = ax.legend(loc="lower right", bbox_to_anchor=(0.985, 0.14), **legend_kwargs)
    else:
        legend = ax.legend(loc="upper right", **legend_kwargs)
    legend.get_frame().set_edgecolor("#B8B8B8")
    legend.get_frame().set_linewidth(1.0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_all_figures(run_histories: list[RunHistory], output_root: Path) -> list[Path]:
    """
    功能: 按预设的 4 个指标为两个数据集批量导出总览图。
    输入: `run_histories` 为所有实验历史，`output_root` 为图像输出目录。
    输出: `list[Path]`，包含已写出的 8 张图路径。
    示例: `export_all_figures(histories, Path('paper_experiment/surrogate/plots'))`。
    时间: 2026-04-26。
    """
    exported_paths: list[Path] = []
    for dataset in DATASET_ORDER:
        dataset_display_name = DATASET_DISPLAY_NAMES[dataset]
        for metric_key, metric_title, ylabel, file_stub in PLOT_SPECS:
            file_name = f"{dataset}_{file_stub}_all_models.png"
            title = f"{dataset_display_name} {metric_title}"
            output_path = output_root / file_name
            plot_metric_figure(
                run_histories,
                dataset=dataset,
                metric_key=metric_key,
                title=title,
                ylabel=ylabel,
                output_path=output_path,
            )
            exported_paths.append(output_path)
    return exported_paths


def main() -> None:
    """
    功能: 组织聚合训练曲线绘图全流程，并打印输出路径。
    输入: 无显式函数参数，依赖命令行参数。
    输出: 无，副作用是生成图像并向标准输出打印结果。
    示例: `cd /Users/epiphanyer/Desktop/coding && python -m paper_experiment.surrogate.plot_run_histories --output-root ./paper_experiment/surrogate/plots`。
    时间: 2026-04-26。
    """
    args = parse_args()
    runs_root = Path(args.runs_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    build_mpl_cache(output_root)
    run_histories = discover_run_histories(runs_root)
    exported_paths = export_all_figures(run_histories, output_root)
    print("Exported figures:")
    for figure_path in exported_paths:
        print(f"- {figure_path}")


if __name__ == "__main__":
    main()
