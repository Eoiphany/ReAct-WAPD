"""
用途:
  汇总 PMNet 与 RMNet 的训练历史和指标，生成对比图表与 compare_summary.json。

示例命令:
  python surrogate/vis/model_summary.py \
    --pmnet-ipynb /path/to/pmnet.ipynb \
    --rmnet-metrics /path/to/rmnet_metrics_summary.json \
    --rmnet-history /path/to/rmnet_history.json \
    --output-root surrogate/vis/outputs/pmnet_vs_rmnet

参数说明:
  --pmnet-ipynb: PMNet notebook 路径，用于提取历史日志。
  --rmnet-metrics: RMNet 的 metrics_summary.json。
  --rmnet-history: RMNet 的 history.json。
  --output-root: 输出目录。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "vis" / "outputs" / "pmnet_vs_rmnet"
EPOCH_RMSE_RE = re.compile(r"Epoch\s+(\d+)\s*/\s*\d+\s*\|\s*RMSE:\s*([0-9.]+)", re.IGNORECASE)
BEST_RMSE_RE = re.compile(r"Best\s+RMSE:\s*([0-9.]+)", re.IGNORECASE)
TEST_RMSE_RE = re.compile(r"Test\s+RMSE:\s*([0-9.]+)", re.IGNORECASE)
SAVE_PATH_RE = re.compile(r"Saved\s+best\s+model\s+to\s+(.+)", re.IGNORECASE)


def _as_text(obj):
    if obj is None:
        return ""
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return "".join(_as_text(item) for item in obj)
    return str(obj)


def extract_notebook_metrics(ipynb_path: Path) -> tuple[list[dict], dict]:
    notebook = json.loads(ipynb_path.read_text(encoding="utf-8"))
    text_chunks: list[str] = []
    for cell in notebook.get("cells", []):
        source = _as_text(cell.get("source", ""))
        if source:
            text_chunks.append(source)
        for output in cell.get("outputs", []):
            if "text" in output:
                text_chunks.append(_as_text(output.get("text")))
            elif "data" in output and "text/plain" in output["data"]:
                text_chunks.append(_as_text(output["data"]["text/plain"]))

    flat_text = "\n".join(text_chunks)
    epoch_to_rmse: dict[int, float] = {}
    for match in EPOCH_RMSE_RE.finditer(flat_text):
        epoch_to_rmse[int(match.group(1))] = float(match.group(2))

    history: list[dict] = []
    best_so_far = None
    for epoch in sorted(epoch_to_rmse):
        val_rmse = epoch_to_rmse[epoch]
        best_so_far = val_rmse if best_so_far is None else min(best_so_far, val_rmse)
        history.append(
            {
                "epoch": epoch,
                "train_loss": None,
                "val_rmse": val_rmse,
                "best_val_rmse": best_so_far,
                "lr": None,
            }
        )

    best_rmse_matches = BEST_RMSE_RE.findall(flat_text)
    test_rmse_matches = TEST_RMSE_RE.findall(flat_text)
    save_path_matches = SAVE_PATH_RE.findall(flat_text)
    metrics = {
        "best_val_rmse": float(best_rmse_matches[-1]) if best_rmse_matches else (history[-1]["best_val_rmse"] if history else None),
        "best_checkpoint": save_path_matches[-1].strip() if save_path_matches else None,
        "history_file": None,
        "curve_file": None,
        "test_rmse": float(test_rmse_matches[-1]) if test_rmse_matches else None,
        "source_notebook": str(ipynb_path),
    }
    return history, metrics


def load_history_json(json_path: Path) -> list[dict]:
    return json.loads(json_path.read_text(encoding="utf-8"))


def save_history_csv(history: list[dict], output_path: Path) -> None:
    fieldnames = ["epoch", "train_loss", "val_rmse", "best_val_rmse", "lr"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row.get("epoch"),
                    "train_loss": "" if row.get("train_loss") is None else row.get("train_loss"),
                    "val_rmse": "" if row.get("val_rmse") is None else row.get("val_rmse"),
                    "best_val_rmse": "" if row.get("best_val_rmse") is None else row.get("best_val_rmse"),
                    "lr": "" if row.get("lr") is None else row.get("lr"),
                }
            )


def save_json(data, output_path: Path) -> None:
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def plot_single_history(history: list[dict], output_path: Path, title: str) -> None:
    epochs = [row["epoch"] for row in history]
    val_rmses = [row["val_rmse"] for row in history if row.get("val_rmse") is not None]
    val_epochs = [row["epoch"] for row in history if row.get("val_rmse") is not None]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(val_epochs, val_rmses, color="#d62728", marker="s", label="Val RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_compare_val_rmse(model_a_history: list[dict], model_b_history: list[dict], output_path: Path) -> None:
    model_a_epochs = [row["epoch"] for row in model_a_history]
    model_a_vals = [row["val_rmse"] for row in model_a_history]
    model_b_epochs = [row["epoch"] for row in model_b_history]
    model_b_vals = [row["val_rmse"] for row in model_b_history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(model_a_epochs, model_a_vals, color="#4c78a8", marker="o", linewidth=2.2, markersize=5.5, label="PMNet")
    ax.plot(model_b_epochs, model_b_vals, color="#f58518", marker="s", linewidth=2.2, markersize=5.5, label="RMNet")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("PMNet vs RMNet Validation RMSE")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_compare_metrics(model_a_metrics: dict, model_b_metrics: dict, output_path: Path) -> None:
    labels = ["Best Val RMSE", "Test RMSE"]
    model_a_values = [model_a_metrics.get("best_val_rmse"), model_a_metrics.get("test_rmse")]
    model_b_values = [model_b_metrics.get("best_val_rmse"), model_b_metrics.get("test_rmse")]
    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([idx - width / 2 for idx in x], model_a_values, width=width, label="PMNet", color="#4c78a8")
    ax.bar([idx + width / 2 for idx in x], model_b_values, width=width, label="RMNet", color="#f58518")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE")
    ax.set_title("PMNet vs RMNet Final Metrics")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmnet-ipynb", required=True)
    parser.add_argument("--rmnet-metrics", required=True)
    parser.add_argument("--rmnet-history", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    root = Path(args.output_root)
    model_a_dir = root / "pmnet"
    model_b_dir = root / "rmnet"
    model_a_dir.mkdir(parents=True, exist_ok=True)
    model_b_dir.mkdir(parents=True, exist_ok=True)

    model_a_history, model_a_metrics = extract_notebook_metrics(Path(args.pmnet_ipynb))
    model_a_metrics["history_file"] = str(model_a_dir / "history.csv")
    model_a_metrics["curve_file"] = str(model_a_dir / "rmse_curve.png")
    save_history_csv(model_a_history, model_a_dir / "history.csv")
    save_json(model_a_history, model_a_dir / "history.json")
    save_json(model_a_metrics, model_a_dir / "metrics_summary.json")
    plot_single_history(model_a_history, model_a_dir / "rmse_curve.png", "PMNet Fine-Tuning History")

    model_b_metrics = json.loads(Path(args.rmnet_metrics).read_text(encoding="utf-8"))
    model_b_history = load_history_json(Path(args.rmnet_history))
    model_b_metrics["history_file"] = str(model_b_dir / "history.csv")
    model_b_metrics["curve_file"] = str(model_b_dir / "rmse_curve.png")
    save_history_csv(model_b_history, model_b_dir / "history.csv")
    save_json(model_b_history, model_b_dir / "history.json")
    save_json(model_b_metrics, model_b_dir / "metrics_summary.json")
    plot_single_history(model_b_history, model_b_dir / "rmse_curve.png", "RMNet Fine-Tuning History")

    plot_compare_val_rmse(model_a_history, model_b_history, root / "compare_val_rmse.png")
    plot_compare_metrics(model_a_metrics, model_b_metrics, root / "compare_final_metrics.png")

    compare_summary = {
        "model_a": {
            "name": "PMNet",
            "history_file": str(model_a_dir / "history.csv"),
            "metrics_file": str(model_a_dir / "metrics_summary.json"),
            "best_val_rmse": model_a_metrics.get("best_val_rmse"),
            "test_rmse": model_a_metrics.get("test_rmse"),
        },
        "model_b": {
            "name": "RMNet",
            "history_file": str(model_b_dir / "history.csv"),
            "metrics_file": str(model_b_dir / "metrics_summary.json"),
            "best_val_rmse": model_b_metrics.get("best_val_rmse"),
            "test_rmse": model_b_metrics.get("test_rmse"),
        },
        "plots": {
            "val_rmse_curve": str(root / "compare_val_rmse.png"),
            "final_metrics_bar": str(root / "compare_final_metrics.png"),
        },
        "notes": [
            "PMNet metrics were reconstructed from notebook outputs.",
            "If notebook outputs do not include train_loss, that field stays empty.",
        ],
    }
    save_json(compare_summary, root / "compare_summary.json")
    print(f"Saved outputs to: {root}")


if __name__ == "__main__":
    main()
