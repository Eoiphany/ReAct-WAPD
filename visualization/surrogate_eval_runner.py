from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from surrogate.data_surrogate import (
    RadioMap3DSeerDataset,
    USCDataset,
    resolve_radiomap_sample_pairs,
    resolve_usc_sample_ids,
)
from surrogate.model_registry import build_model, select_prediction
from surrogate.utils import configure_plot_style, get_device, load_checkpoint


MODEL_LABELS = {
    "pmnet": "PMNet",
    "rmnet": "RMNet",
    "radiounet": "RadioUNet",
    "transunet": "TransUNet",
    "unet": "UNet",
}
DATASET_LABELS = {"radiomap3dseer": "RadioMap3DSeer", "usc": "USC"}
DB_MIN = -162.0
DB_MAX = -75.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run surrogate multi-checkpoint evaluation for the visualization UI.")
    parser.add_argument("--payload-json", required=True, type=str)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_eval_csv(payload: dict[str, Any]) -> str | None:
    explicit = (payload.get("csvFile") or "").strip()
    if explicit:
        return explicit
    checkpoint_run_dir = (payload.get("checkpointRunDir") or "").strip()
    if checkpoint_run_dir:
        resolved = Path(checkpoint_run_dir).resolve()
        candidate = resolved / "test_split.csv"
        if candidate.exists():
            return str(candidate)
        child_hits = sorted({path.parent.resolve() for path in resolved.glob("*/test_split.csv")}, key=lambda item: str(item))
        if len(child_hits) == 1:
            return str(child_hits[0] / "test_split.csv")
    return None


def resolve_sample_names(dataset_key: str, data_root: str, csv_file: str | None) -> list[str]:
    if dataset_key == "usc":
        return resolve_usc_sample_ids(data_root, csv_file)
    pairs = resolve_radiomap_sample_pairs(data_root, csv_file)
    return [f"{scene_id}_{tx_id}" for scene_id, tx_id in pairs]


def format_sample_items(dataset_key: str, sample_names: list[str], limit: int = 12) -> list[str]:
    items = sample_names[:limit]
    if dataset_key == "usc":
        return [f"sample_id={item}" for item in items]
    formatted = []
    for item in items:
        scene_id, tx_id = item.split("_", 1)
        formatted.append(f"scene={scene_id}, tx={tx_id}")
    return formatted


def load_single_sample(dataset_key: str, data_root: str, sample_name: str, use_height: bool) -> tuple[np.ndarray, np.ndarray]:
    if dataset_key == "usc":
        dataset = USCDataset(data_root, [sample_name])
        inputs, target = dataset[0]
        return inputs.numpy(), target.squeeze(0).numpy()
    scene_id, tx_id = sample_name.split("_", 1)
    dataset = RadioMap3DSeerDataset(data_root, [(scene_id, tx_id)], use_height=use_height)
    inputs, target, _, _ = dataset[0]
    return inputs.numpy(), target.squeeze(0).numpy()


def to_db_scale(array: np.ndarray) -> np.ndarray:
    return np.clip(array, 0.0, 1.0) * (DB_MAX - DB_MIN) + DB_MIN


def compute_r2_numpy(prediction: np.ndarray, label: np.ndarray) -> float:
    diff = label - prediction
    ss_res = float(np.sum(np.square(diff)))
    centered = label - float(np.mean(label))
    ss_tot = float(np.sum(np.square(centered)))
    return 0.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot


def predict_single_sample(
    *,
    dataset_key: str,
    checkpoint: dict[str, Any],
    payload: dict[str, Any],
    sample_name: str,
    device: torch.device,
) -> dict[str, Any]:
    use_height = str(payload.get("useHeight", "true")).lower() == "true"
    inputs_np, label_np = load_single_sample(dataset_key, str(payload["dataRoot"]), sample_name, use_height)
    model = build_model(
        model_type=str(checkpoint["modelType"]),
        output_stride=int(payload.get("outputStride", 16)),
        in_channels=int(inputs_np.shape[0]),
    )
    load_checkpoint(model, str(checkpoint["path"]), strict=True)
    model = model.to(device)
    model.eval()
    inputs = torch.from_numpy(inputs_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        prediction = torch.clamp(select_prediction(model(inputs)), 0.0, 1.0)
    prediction_np = prediction.detach().cpu().squeeze().numpy().astype(np.float32)
    diff = prediction_np - label_np
    result = {
        "modelType": checkpoint["modelType"],
        "label": label_np.astype(np.float32),
        "prediction": prediction_np,
        "metrics": {
            "rmse": float(np.sqrt(np.mean(np.square(diff)))),
            "mae": float(np.mean(np.abs(diff))),
            "r2": compute_r2_numpy(prediction_np, label_np),
        },
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def save_compare_figure(
    *,
    dataset_key: str,
    sample_name: str,
    predictions: list[dict[str, Any]],
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

    ncols = len(predictions)
    fig, axes = plt.subplots(3, ncols, figsize=(3.4 * ncols, 8.6), constrained_layout=True)
    if ncols == 1:
        axes = np.asarray(axes).reshape(3, 1)

    error_vmax = max(float(np.max(np.abs(item["prediction"] - item["label"]))) for item in predictions)
    error_vmax = max(error_vmax, 1e-6)
    gt_im = None
    err_im = None
    for row_idx, row_label in enumerate(("Ground Truth", "Prediction", "Absolute Error")):
        axes[row_idx, 0].set_ylabel(row_label, fontsize=12, labelpad=16)

    for col_idx, item in enumerate(predictions):
        label_db = to_db_scale(item["label"])
        pred_db = to_db_scale(item["prediction"])
        error = np.abs(item["prediction"] - item["label"])
        gt_im = axes[0, col_idx].imshow(label_db, cmap="viridis", vmin=DB_MIN, vmax=DB_MAX)
        axes[1, col_idx].imshow(pred_db, cmap="viridis", vmin=DB_MIN, vmax=DB_MAX)
        err_im = axes[2, col_idx].imshow(error, cmap="magma", vmin=0.0, vmax=error_vmax)
        metrics = item["metrics"]
        axes[0, col_idx].set_title(
            f"{MODEL_LABELS.get(item['modelType'], item['modelType'])}\nRMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}",
            fontsize=11,
        )
        for row_idx in range(3):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    fig.suptitle(f"{DATASET_LABELS[dataset_key]} | sample={sample_name}", fontsize=13)
    cbar1 = fig.colorbar(gt_im, ax=axes[:2, :], fraction=0.028, pad=0.02)
    cbar1.set_label("Path gain (dB)")
    cbar2 = fig.colorbar(err_im, ax=axes[2, :], fraction=0.028, pad=0.02)
    cbar2.set_label("Absolute error")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    payload = read_json(Path(args.payload_json))
    runtime_dir = Path(payload["runtimeDir"]).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    dataset_key = str(payload["datasetKey"])
    checkpoints = list(payload.get("checkpoints") or [])
    if not checkpoints:
        raise ValueError("No checkpoints provided for evaluation.")

    csv_file = resolve_eval_csv(payload)
    resolved_sample_names = resolve_sample_names(dataset_key, str(payload["dataRoot"]), csv_file)
    provided_sample_names = [str(item).strip() for item in (payload.get("sampleNames") or []) if str(item).strip()]
    if provided_sample_names:
        available = set(resolved_sample_names)
        sample_names = [name for name in provided_sample_names if name in available]
    else:
        sample_names = resolved_sample_names
    if not sample_names:
        raise ValueError("No evaluation samples resolved.")
    sample_name = str(payload.get("selectedSample") or "").strip() or sample_names[0]
    if sample_name not in sample_names:
        if sample_name in resolved_sample_names:
            sample_names = [*sample_names, sample_name]
        else:
            raise ValueError(f"Selected sample not found: {sample_name}")
    sample_items = format_sample_items(dataset_key, sample_names, limit=max(len(sample_names), 12))
    checkpoint_run_dir = (payload.get("checkpointRunDir") or "").strip()
    if csv_file and checkpoint_run_dir and Path(csv_file).parent == Path(checkpoint_run_dir).resolve():
        sample_source = "来自当前输出目录的 test_split.csv"
    elif csv_file:
        sample_source = "来自显式指定的样本文件"
    else:
        sample_source = "来自当前数据集默认测试样本集合。"
    write_json(
        runtime_dir / "manifest.json",
        {
            "status": "preparing",
            "datasetKey": dataset_key,
            "checkpoints": checkpoints,
            "sampleNames": sample_names,
            "sampleItems": sample_items,
            "sampleSource": sample_source,
            "sampleName": sample_name,
        },
    )

    device = get_device()
    metric_rows = []
    predictions = []
    for checkpoint in checkpoints:
        print(
            f"PREDICT: dataset={dataset_key} sample={sample_name} model={checkpoint['modelType']} checkpoint={checkpoint['path']}",
            flush=True,
        )
        prediction = predict_single_sample(
            dataset_key=dataset_key,
            checkpoint=checkpoint,
            payload=payload,
            sample_name=sample_name,
            device=device,
        )
        predictions.append(prediction)
        metric_rows.append(
            {
                "modelType": checkpoint["modelType"],
                "checkpointPath": checkpoint["path"],
                "sampleName": sample_name,
                "rmse": prediction["metrics"]["rmse"],
                "mae": prediction["metrics"]["mae"],
                "r2": prediction["metrics"]["r2"],
            }
        )
        write_json(
            runtime_dir / "manifest.json",
            {
                "status": "running",
                "datasetKey": dataset_key,
                "checkpoints": checkpoints,
                "sampleNames": sample_names,
                "sampleItems": sample_items,
                "sampleSource": sample_source,
                "sampleName": sample_name,
                "metrics": metric_rows,
            },
        )

    compare_figure_path = runtime_dir / "compare_figure.png"
    save_compare_figure(
        dataset_key=dataset_key,
        sample_name=sample_name,
        predictions=predictions,
        output_path=compare_figure_path,
    )
    write_json(
        runtime_dir / "manifest.json",
        {
            "status": "done",
            "datasetKey": dataset_key,
            "checkpoints": checkpoints,
            "sampleNames": sample_names,
            "sampleItems": sample_items,
            "sampleSource": sample_source,
            "sampleName": sample_name,
            "compareFigurePath": str(compare_figure_path),
            "metrics": metric_rows,
        },
    )


if __name__ == "__main__":
    main()
