"""注释
命令示例:
python -m surrogate.summarize_surrogate_runs \
  --runs-root /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/runs \
  --output-root /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/all_model_summary

若当前目录在 `/Users/epiphanyer/Desktop/coding`，则使用:
python -m paper_experiment.surrogate.summarize_surrogate_runs \
  --runs-root /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/runs \
  --output-root /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/all_model_summary

参数说明:
- --runs-root: 训练结果根目录，内部应包含 `pmnet_usc/16_...` 这类实验目录。
- --output-root: 汇总产物输出目录；脚本会写入 `csv/md/json/png`。
- --batch-size: 评估时 DataLoader 的 batch size。
- --num-workers: 评估时 DataLoader worker 数量。
- --device: 推理设备，`auto/cpu/cuda/mps`。

脚本逻辑:
- 扫描 `surrogate/runs` 中的五类模型在两个数据集上的 best checkpoint。
- 依据训练脚本原始口径重新评估性能指标：
  `USC` 复现验证集划分，`RadioMap3DSeer` 使用保存的 `test_split.csv`。
- 输出一个汇总表，并在两个数据集上各选一个固定样本生成 3×N 对比图：
  第一行为 ground truth，第二行为模型预测，第三行为 absolute error。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .data_surrogate import (
        RadioMap3DSeerDataset,
        USCDataset,
        numeric_sort_key,
        pair_sort_key,
        read_radiomap_sample_pairs,
        resolve_usc_sample_ids,
    )
    from .model_registry import build_model, select_prediction
    from .train_usc_surrogate import split_sample_ids_deterministically
    from .utils import configure_plot_style, get_device, load_checkpoint
except ImportError:
    from paper_experiment.surrogate.data_surrogate import (
        RadioMap3DSeerDataset,
        USCDataset,
        numeric_sort_key,
        pair_sort_key,
        read_radiomap_sample_pairs,
        resolve_usc_sample_ids,
    )
    from paper_experiment.surrogate.model_registry import build_model, select_prediction
    from paper_experiment.surrogate.train_usc_surrogate import split_sample_ids_deterministically
    from paper_experiment.surrogate.utils import configure_plot_style, get_device, load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "surrogate" / "runs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "surrogate" / "test" / "runs" / "all_model_summary"
DEFAULT_DATA_ROOTS = {
    "usc": PROJECT_ROOT / "usc-data",
    "radiomap3dseer": PROJECT_ROOT / "dataset",
}
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
FIGURE_TITLES = {
    "usc": "Cross-Model Qualitative Comparison of Ground Truth, Predictions, and Absolute Errors on the USC Dataset",
    "radiomap3dseer": "Cross-Model Qualitative Comparison of Ground Truth, Predictions, and Absolute Errors on the RadioMap3DSeer Dataset",
}
DB_MIN = -162.0
DB_MAX = -75.0


@dataclass(frozen=True)
class ExperimentRun:
    dataset: str
    model_type: str
    run_dir: Path
    checkpoint_path: Path
    config: dict[str, Any]
    metrics_summary: dict[str, Any]


@dataclass(frozen=True)
class MetricRow:
    dataset: str
    model_type: str
    split_name: str
    sample_count: int
    rmse: float
    mae: float
    r2: float
    checkpoint_path: str
    run_dir: str


@dataclass(frozen=True)
class FigureSampleMetrics:
    rmse: float
    mae: float
    r2: float


@dataclass(frozen=True)
class FigureColumn:
    model_type: str
    prediction: np.ndarray
    label: np.ndarray
    metrics: FigureSampleMetrics


class USCNamedDataset(Dataset):
    def __init__(self, data_root: Path, sample_ids: list[str]):
        self.sample_ids = list(sample_ids)
        self.dataset = USCDataset(str(data_root), self.sample_ids)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        inputs, target = self.dataset[idx]
        return inputs, target, self.sample_ids[idx]


class RadioMapNamedDataset(Dataset):
    def __init__(self, data_root: Path, sample_pairs: list[tuple[str, str]], use_height: bool):
        self.sample_pairs = list(sample_pairs)
        self.dataset = RadioMap3DSeerDataset(str(data_root), self.sample_pairs, use_height=use_height)

    def __len__(self) -> int:
        return len(self.sample_pairs)

    def __getitem__(self, idx: int):
        inputs, target, scene_id, tx_id = self.dataset[idx]
        return inputs, target, f"{scene_id}_{tx_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize surrogate run metrics and export cross-model figures.")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT), type=str)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), type=str)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], type=str)
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return get_device()
    return torch.device(device_name)


def discover_experiments(runs_root: Path) -> dict[tuple[str, str], ExperimentRun]:
    experiments: dict[tuple[str, str], ExperimentRun] = {}
    for model_type in MODEL_ORDER:
        for dataset in ("usc", "radiomap3dseer"):
            parent_dir = runs_root / f"{model_type}_{dataset}"
            if not parent_dir.exists():
                continue
            candidate_dirs = sorted((path for path in parent_dir.iterdir() if path.is_dir()), key=lambda item: item.name)
            if not candidate_dirs:
                continue
            run_dir = candidate_dirs[-1]
            checkpoint_paths = sorted(run_dir.glob("*_best.pt"))
            if not checkpoint_paths:
                raise FileNotFoundError(f"Missing *_best.pt under {run_dir}")
            config_path = run_dir / "config.json"
            metrics_path = run_dir / "metrics_summary.json"
            config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
            metrics_summary = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
            experiments[(dataset, model_type)] = ExperimentRun(
                dataset=dataset,
                model_type=model_type,
                run_dir=run_dir,
                checkpoint_path=checkpoint_paths[0],
                config=config,
                metrics_summary=metrics_summary,
            )
    return experiments


def resolve_local_data_root(dataset: str) -> Path:
    data_root = DEFAULT_DATA_ROOTS[dataset]
    if not data_root.exists():
        raise FileNotFoundError(f"Missing local data root for {dataset}: {data_root}")
    return data_root


def resolve_usc_val_sample_ids(experiment: ExperimentRun, data_root: Path) -> list[str]:
    val_split_path = experiment.run_dir / "val_split.csv"
    if val_split_path.exists():
        return _read_single_column_csv(val_split_path)

    sample_ids = resolve_usc_sample_ids(str(data_root), None)
    train_ratio = float(experiment.config.get("train_ratio", 0.9))
    seed = int(experiment.config.get("seed", 42))
    _, val_ids = split_sample_ids_deterministically(sample_ids, train_ratio=train_ratio, seed=seed)
    return val_ids


def resolve_radiomap_split_pairs(experiment: ExperimentRun, split_name: str) -> list[tuple[str, str]]:
    split_path = experiment.run_dir / f"{split_name}_split.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_name} split for {experiment.model_type}: {split_path}")
    return read_radiomap_sample_pairs(split_path)


def _read_single_column_csv(csv_path: Path) -> list[str]:
    values: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if row and row[0].strip():
                values.append(row[0].strip())
    return values


def build_named_loader(
    *,
    experiment: ExperimentRun,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, list[str]]:
    dataset = experiment.dataset
    data_root = resolve_local_data_root(dataset)
    if dataset == "usc":
        sample_ids = resolve_usc_val_sample_ids(experiment, data_root)
        named_dataset = USCNamedDataset(data_root, sample_ids)
        ordered_names = list(sample_ids)
    else:
        sample_pairs = resolve_radiomap_split_pairs(experiment, "test")
        use_height = bool(experiment.config.get("use_height", True))
        named_dataset = RadioMapNamedDataset(data_root, sample_pairs, use_height=use_height)
        ordered_names = [f"{scene_id}_{tx_id}" for scene_id, tx_id in sample_pairs]

    loader = DataLoader(
        named_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    return loader, ordered_names


def evaluate_experiment(
    *,
    experiment: ExperimentRun,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[MetricRow, list[str]]:
    loader, ordered_names = build_named_loader(
        experiment=experiment,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    input_channels = 2
    model = build_model(
        model_type=experiment.model_type,
        output_stride=int(experiment.config.get("output_stride", 16)),
        in_channels=input_channels,
    )
    load_checkpoint(model, str(experiment.checkpoint_path), strict=True)
    model = model.to(device)
    model.eval()

    total_rmse = 0.0
    total_mae = 0.0
    total_r2 = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets, _names in loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            preds = torch.clamp(select_prediction(model(inputs)), 0.0, 1.0)

            diff = preds - targets
            batch_rmse = float(torch.sqrt(torch.mean(diff.square())).item())
            batch_mae = float(torch.mean(torch.abs(diff)).item())
            target_mean = torch.mean(targets)
            ss_res = torch.sum(diff.square())
            ss_tot = torch.sum((targets - target_mean).square())
            batch_r2 = 0.0 if float(ss_tot.item()) == 0.0 else float((1.0 - ss_res / ss_tot).item())

            batch_size_current = inputs.size(0)
            total_rmse += batch_rmse * batch_size_current
            total_mae += batch_mae * batch_size_current
            total_r2 += batch_r2 * batch_size_current
            total_samples += batch_size_current

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    split_name = "val" if experiment.dataset == "usc" else "test"
    denom = max(total_samples, 1)
    metric_row = MetricRow(
        dataset=experiment.dataset,
        model_type=experiment.model_type,
        split_name=split_name,
        sample_count=total_samples,
        rmse=total_rmse / denom,
        mae=total_mae / denom,
        r2=total_r2 / denom,
        checkpoint_path=str(experiment.checkpoint_path),
        run_dir=str(experiment.run_dir),
    )
    return metric_row, ordered_names


def choose_figure_sample(dataset: str, ordered_names: list[str]) -> str:
    if not ordered_names:
        raise ValueError(f"No samples available for dataset: {dataset}")
    if dataset == "usc":
        ranked_names = sorted(ordered_names, key=numeric_sort_key)
    else:
        pair_names = [tuple(name.split("_", 1)) for name in ordered_names]
        ranked_names = [f"{scene_id}_{tx_id}" for scene_id, tx_id in sorted(pair_names, key=pair_sort_key)]
    return ranked_names[len(ranked_names) // 2]


def resolve_best_val_sample_names(experiment: ExperimentRun) -> list[str]:
    if experiment.dataset == "usc":
        return resolve_usc_val_sample_ids(experiment, resolve_local_data_root("usc"))
    sample_pairs = resolve_radiomap_split_pairs(experiment, "val")
    return [f"{scene_id}_{tx_id}" for scene_id, tx_id in sample_pairs]


def parse_best_val_history_metrics(experiment: ExperimentRun) -> dict[str, float | None]:
    history_path = experiment.run_dir / "history.json"
    if not history_path.exists():
        return {"best_val_rmse": None, "best_val_mae": None, "best_val_r2": None}

    history = json.loads(history_path.read_text(encoding="utf-8"))
    rows = [row for row in history if row.get("val_rmse") is not None]
    if not rows:
        return {"best_val_rmse": None, "best_val_mae": None, "best_val_r2": None}

    best_row = min(rows, key=lambda row: row["val_rmse"])
    return {
        "best_val_rmse": best_row.get("val_rmse"),
        "best_val_mae": best_row.get("val_mae"),
        "best_val_r2": best_row.get("val_r2"),
    }


def parse_eval_metrics(experiment: ExperimentRun) -> dict[str, float | None]:
    eval_metrics_path = experiment.run_dir / "eval" / "metrics_summary.json"
    if not eval_metrics_path.exists():
        return {"best_val_rmse": None, "best_val_mae": None, "best_val_r2": None}

    payload = json.loads(eval_metrics_path.read_text(encoding="utf-8"))
    if payload.get("eval_split") != "val":
        return {"best_val_rmse": None, "best_val_mae": None, "best_val_r2": None}

    return {
        "best_val_rmse": payload.get("eval_rmse"),
        "best_val_mae": payload.get("eval_mae"),
        "best_val_r2": payload.get("eval_r2"),
    }


def build_best_val_metric_row(
    *,
    experiment: ExperimentRun,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> MetricRow:
    summary_metrics = {
        "best_val_rmse": experiment.metrics_summary.get("best_val_rmse"),
        "best_val_mae": experiment.metrics_summary.get("best_val_mae"),
        "best_val_r2": experiment.metrics_summary.get("best_val_r2"),
    }
    eval_metrics = parse_eval_metrics(experiment)
    history_metrics = parse_best_val_history_metrics(experiment)
    merged_metrics = {
        metric_name: summary_metrics.get(metric_name)
        if summary_metrics.get(metric_name) is not None
        else eval_metrics.get(metric_name)
        if eval_metrics.get(metric_name) is not None
        else history_metrics.get(metric_name)
        for metric_name in ("best_val_rmse", "best_val_mae", "best_val_r2")
    }

    if any(merged_metrics[metric_name] is None for metric_name in ("best_val_rmse", "best_val_mae", "best_val_r2")):
        fallback_metrics, _ = evaluate_experiment(
            experiment=experiment,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        merged_metrics = {
            "best_val_rmse": fallback_metrics.rmse,
            "best_val_mae": fallback_metrics.mae,
            "best_val_r2": fallback_metrics.r2,
        }

    sample_names = resolve_best_val_sample_names(experiment)
    return MetricRow(
        dataset=experiment.dataset,
        model_type=experiment.model_type,
        split_name="best_val",
        sample_count=len(sample_names),
        rmse=float(merged_metrics["best_val_rmse"]),
        mae=float(merged_metrics["best_val_mae"]),
        r2=float(merged_metrics["best_val_r2"]),
        checkpoint_path=str(experiment.checkpoint_path),
        run_dir=str(experiment.run_dir),
    )


def load_single_sample(
    *,
    dataset: str,
    sample_name: str,
    use_height: bool,
) -> tuple[np.ndarray, np.ndarray]:
    data_root = resolve_local_data_root(dataset)
    if dataset == "usc":
        dataset_obj = USCDataset(str(data_root), [sample_name])
        inputs, target = dataset_obj[0]
        return inputs.numpy(), target.squeeze(0).numpy()

    scene_id, tx_id = sample_name.split("_", 1)
    dataset_obj = RadioMap3DSeerDataset(str(data_root), [(scene_id, tx_id)], use_height=use_height)
    inputs, target, _, _ = dataset_obj[0]
    return inputs.numpy(), target.squeeze(0).numpy()


def predict_single_sample(
    *,
    experiment: ExperimentRun,
    sample_name: str,
    device: torch.device,
) -> FigureColumn:
    use_height = bool(experiment.config.get("use_height", True))
    inputs_np, label_np = load_single_sample(
        dataset=experiment.dataset,
        sample_name=sample_name,
        use_height=use_height,
    )
    model = build_model(
        model_type=experiment.model_type,
        output_stride=int(experiment.config.get("output_stride", 16)),
        in_channels=int(inputs_np.shape[0]),
    )
    load_checkpoint(model, str(experiment.checkpoint_path), strict=True)
    model = model.to(device)
    model.eval()

    inputs = torch.from_numpy(inputs_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        prediction = torch.clamp(select_prediction(model(inputs)), 0.0, 1.0)

    prediction_np = prediction.detach().cpu().squeeze().numpy().astype(np.float32)
    diff = prediction_np - label_np
    metrics = FigureSampleMetrics(
        rmse=float(np.sqrt(np.mean(np.square(diff)))),
        mae=float(np.mean(np.abs(diff))),
        r2=compute_r2_numpy(prediction_np, label_np),
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return FigureColumn(
        model_type=experiment.model_type,
        prediction=prediction_np,
        label=label_np.astype(np.float32),
        metrics=metrics,
    )


def compute_r2_numpy(prediction: np.ndarray, label: np.ndarray) -> float:
    diff = label - prediction
    ss_res = float(np.sum(np.square(diff)))
    centered = label - float(np.mean(label))
    ss_tot = float(np.sum(np.square(centered)))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def to_db_scale(array: np.ndarray) -> np.ndarray:
    array = np.clip(array, 0.0, 1.0)
    return array * (DB_MAX - DB_MIN) + DB_MIN


def build_figure_title(dataset: str) -> str:
    return FIGURE_TITLES[dataset]


def save_cross_model_figure(
    *,
    dataset: str,
    sample_name: str,
    columns: list[FigureColumn],
    output_path: Path,
) -> None:
    mpl_cache_dir = output_path.parent / ".mpl-cache"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    configure_plot_style()
    import matplotlib.pyplot as plt

    ncols = len(columns)
    fig, axes = plt.subplots(3, ncols, figsize=(3.4 * ncols, 8.8), constrained_layout=True)
    if ncols == 1:
        axes = np.asarray(axes).reshape(3, 1)

    error_vmax = max(float(np.max(np.abs(column.prediction - column.label))) for column in columns)
    error_vmax = max(error_vmax, 1e-6)
    gt_im = None
    err_im = None

    row_labels = ("Ground Truth", "Prediction", "Absolute Error")
    for row_idx, row_label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(row_label, fontsize=12, labelpad=16)

    for col_idx, column in enumerate(columns):
        label_db = to_db_scale(column.label)
        pred_db = to_db_scale(column.prediction)
        error = np.abs(column.prediction - column.label)

        gt_im = axes[0, col_idx].imshow(label_db, cmap="viridis", vmin=DB_MIN, vmax=DB_MAX)
        axes[0, col_idx].set_title(
            f"{MODEL_DISPLAY_NAMES[column.model_type]}\n"
            f"RMSE={column.metrics.rmse:.4f}, MAE={column.metrics.mae:.4f}, R$^2$={column.metrics.r2:.4f}",
            fontsize=11,
        )
        axes[1, col_idx].imshow(pred_db, cmap="viridis", vmin=DB_MIN, vmax=DB_MAX)
        err_im = axes[2, col_idx].imshow(error, cmap="magma", vmin=0.0, vmax=error_vmax)

        for row_idx in range(3):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            axes[row_idx, col_idx].set_frame_on(False)

    fig.colorbar(
        gt_im,
        ax=axes[:2, :].ravel().tolist(),
        fraction=0.018,
        pad=0.01,
        label="Path Gain (dB)",
    )
    fig.colorbar(
        err_im,
        ax=axes[2, :].ravel().tolist(),
        fraction=0.026,
        pad=0.01,
        label="Absolute Error",
    )
    fig.suptitle(
        build_figure_title(dataset) + f"\nRepresentative Sample: {sample_name}",
        fontsize=14,
        y=1.05,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=500, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def write_metric_outputs(
    *,
    metric_rows: list[MetricRow],
    selected_samples: dict[str, str],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "surrogate_run_metrics_summary.csv"
    md_path = output_root / "surrogate_run_metrics_summary.md"
    json_path = output_root / "surrogate_run_metrics_summary.json"

    fieldnames = [
        "dataset",
        "model_type",
        "split_name",
        "sample_count",
        "rmse",
        "mae",
        "r2",
        "checkpoint_path",
        "run_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow(
                {
                    "dataset": row.dataset,
                    "model_type": row.model_type,
                    "split_name": row.split_name,
                    "sample_count": row.sample_count,
                    "rmse": f"{row.rmse:.6f}",
                    "mae": f"{row.mae:.6f}",
                    "r2": f"{row.r2:.6f}",
                    "checkpoint_path": row.checkpoint_path,
                    "run_dir": row.run_dir,
                }
            )

    markdown_lines = [
        "| Dataset | Model | Split | Samples | RMSE | MAE | R^2 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in metric_rows:
        markdown_lines.append(
            "| "
            + " | ".join(
                [
                    DATASET_DISPLAY_NAMES[row.dataset],
                    MODEL_DISPLAY_NAMES[row.model_type],
                    row.split_name,
                    str(row.sample_count),
                    f"{row.rmse:.6f}",
                    f"{row.mae:.6f}",
                    f"{row.r2:.6f}",
                ]
            )
            + " |"
        )
    markdown_lines.extend(
        [
            "",
            "Representative samples used for the qualitative figures:",
            "",
            f"- USC: `{selected_samples['usc']}`",
            f"- RadioMap3DSeer: `{selected_samples['radiomap3dseer']}`",
        ]
    )
    md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    json_payload = {
        "metrics": [
            {
                "dataset": row.dataset,
                "model_type": row.model_type,
                "split_name": row.split_name,
                "sample_count": row.sample_count,
                "rmse": row.rmse,
                "mae": row.mae,
                "r2": row.r2,
                "checkpoint_path": row.checkpoint_path,
                "run_dir": row.run_dir,
            }
            for row in metric_rows
        ],
        "selected_samples": selected_samples,
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    device = resolve_device(args.device)

    experiments = discover_experiments(runs_root)
    missing_keys = [(dataset, model_type) for dataset in DATASET_ORDER for model_type in MODEL_ORDER if (dataset, model_type) not in experiments]
    if missing_keys:
        raise FileNotFoundError(f"Missing experiments: {missing_keys}")

    metric_rows: list[MetricRow] = []
    selected_samples: dict[str, str] = {}
    for dataset in DATASET_ORDER:
        reference_names = resolve_best_val_sample_names(experiments[(dataset, MODEL_ORDER[0])])
        selected_samples[dataset] = choose_figure_sample(dataset, reference_names)
        for model_type in MODEL_ORDER:
            metric_rows.append(
                build_best_val_metric_row(
                    experiment=experiments[(dataset, model_type)],
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
            )

    metric_rows.sort(key=lambda row: (DATASET_ORDER.index(row.dataset), MODEL_ORDER.index(row.model_type)))
    write_metric_outputs(metric_rows=metric_rows, selected_samples=selected_samples, output_root=output_root)

    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_manifest: dict[str, Any] = {}
    for dataset in DATASET_ORDER:
        sample_name = selected_samples[dataset]
        columns = [
            predict_single_sample(
                experiment=experiments[(dataset, model_type)],
                sample_name=sample_name,
                device=device,
            )
            for model_type in MODEL_ORDER
        ]
        figure_path = figures_dir / f"{dataset}_cross_model_prediction_comparison.png"
        save_cross_model_figure(
            dataset=dataset,
            sample_name=sample_name,
            columns=columns,
            output_path=figure_path,
        )
        figure_manifest[dataset] = {
            "sample_name": sample_name,
            "figure_path": str(figure_path),
            "title": build_figure_title(dataset),
        }

    (output_root / "figure_manifest.json").write_text(json.dumps(figure_manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "device": str(device), "selected_samples": selected_samples}, indent=2))


if __name__ == "__main__":
    main()
