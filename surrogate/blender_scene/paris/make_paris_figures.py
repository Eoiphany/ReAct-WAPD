#!/usr/bin/env python3

# MPLCONFIGDIR=/tmp/matplotlib \
# python /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/blender_scene/paris/make_paris_figures.py

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from paris_paths import OUTPUT_DATASET_DIR, SOURCE_DATASET_ROOT, SOURCE_TEST_RESULT_ROOT


ROOT = OUTPUT_DATASET_DIR
DATASET_ROOT = SOURCE_DATASET_ROOT
TEST_RESULT_ROOT = SOURCE_TEST_RESULT_ROOT


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def load_compare_label() -> np.ndarray:
    rendered_label = TEST_RESULT_ROOT / "label" / "00000_paris_0_label.png"
    if rendered_label.exists():
        return load_gray(rendered_label)
    return load_gray(ROOT / "gain" / "paris_0.png")


def tx_pixel(tx_img: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.where(tx_img > 0)
    if len(xs) == 0:
        return None
    return int(ys[0]), int(xs[0])


def add_tx_marker(ax, tx_rc: tuple[int, int] | None) -> None:
    if tx_rc is None:
        return
    row, col = tx_rc
    ax.scatter([col], [row], c="red", marker="x", s=36, linewidths=1.3)


def save_paris_vs_dataset0() -> None:
    generated = {
        "Urban Height Map": load_gray(ROOT / "png" / "buildingsWHeight" / "paris_oda_height.png"),
        "Access Point Height Encoding": load_gray(ROOT / "png" / "antennasWHeight" / "paris_0.png"),
        "Simulated Path-Loss Label": load_gray(ROOT / "gain" / "paris_0.png"),
    }
    reference = {
        "Urban Height Map": load_gray(DATASET_ROOT / "png" / "buildingsWHeight" / "0.png"),
        "Access Point Height Encoding": load_gray(DATASET_ROOT / "png" / "antennasWHeight" / "0_0.png"),
        "Reference Gain Label": load_gray(DATASET_ROOT / "gain" / "0_0.png"),
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    row_titles = ["Generated Paris Scene", "Reference Dataset Sample"]
    row_data = [generated, reference]
    row_tx = [tx_pixel(generated["Access Point Height Encoding"]), tx_pixel(reference["Access Point Height Encoding"])]

    for r, (title, data, tx_rc) in enumerate(zip(row_titles, row_data, row_tx)):
        for c, (panel_title, arr) in enumerate(data.items()):
            ax = axes[r, c]
            ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
            add_tx_marker(ax, tx_rc)
            ax.set_title(panel_title)
            ax.axis("off")
        axes[r, 0].set_ylabel(title)

    fig.suptitle("Comparative Visualization of Simulated Paris and Reference Dataset Samples")
    fig.savefig(ROOT / "paris_vs_dataset0_compare.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def save_label_pred_compare() -> None:
    panels = [
        ("Simulated Path-Loss Label", load_compare_label()),
        ("Model Prediction", load_gray(TEST_RESULT_ROOT / "pred" / "00000_paris_0_pred.png")),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    for ax, (title, arr) in zip(axes, panels):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle("Comparison Between the Simulated Label and the Learned Prediction")
    fig.savefig(ROOT / "paris_label_pred_compare.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    set_plot_style()
    save_paris_vs_dataset0()
    save_label_pred_compare()


if __name__ == "__main__":
    main()
