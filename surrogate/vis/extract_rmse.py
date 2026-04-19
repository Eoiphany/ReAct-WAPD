"""
用途:
  从 ipynb 输出日志中提取 RMSE 曲线并保存成图片。

示例命令:
  python surrogate/vis/extract_rmse.py \
    --ipynb surrogate/vis/RMSE_data.ipynb \
    --out surrogate/vis/outputs/rmse_curve.png

参数说明:
  --ipynb: 输入 notebook 路径。
  --out: 输出曲线图片路径。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


DEFAULT_OUT = Path(__file__).resolve().parents[1] / "vis" / "outputs" / "rmse_curve.png"
RMSE_RE = re.compile(r"Epoch\s+(\d+)\s*/\s*\d+\s*\|\s*RMSE:\s*([0-9.]+)", re.IGNORECASE)


def _as_text(obj):
    if obj is None:
        return ""
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return "".join(_as_text(x) for x in obj)
    return str(obj)


def _extract_from_text(text, epochs, rmses):
    for match in RMSE_RE.finditer(text):
        epochs.append(int(match.group(1)))
        rmses.append(float(match.group(2)))


def extract_rmse(ipynb_path: Path):
    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
    epochs = []
    rmses = []
    for cell in nb.get("cells", []):
        source = _as_text(cell.get("source", ""))
        if source:
            _extract_from_text(source, epochs, rmses)
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            text = ""
            if "text" in out:
                text = _as_text(out.get("text"))
            elif "data" in out and "text/plain" in out["data"]:
                text = _as_text(out["data"]["text/plain"])
            _extract_from_text(_as_text(text), epochs, rmses)
    if not epochs:
        flat = json.dumps(nb, ensure_ascii=False)
        _extract_from_text(flat, epochs, rmses)
    return epochs, rmses


def plot_rmse(epochs, rmses, out_path: Path):
    if not epochs:
        raise ValueError("No RMSE entries found in notebook outputs.")
    pairs = {}
    for epoch, rmse in zip(epochs, rmses):
        pairs[int(epoch)] = float(rmse)
    epochs = sorted(pairs.keys())
    rmses = [pairs[e] for e in epochs]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, rmses, color="#2a5c8a", linewidth=2.0, marker="o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("Training RMSE")
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipynb", required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    epochs, rmses = extract_rmse(Path(args.ipynb))
    plot_rmse(epochs, rmses, Path(args.out))


if __name__ == "__main__":
    main()
