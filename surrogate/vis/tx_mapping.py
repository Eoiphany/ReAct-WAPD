"""
用途:
  可视化不同发射机坐标解释方式如何映射到图像像素，用于排查 row/col 与 x/y 的方向问题。

示例命令:
  python surrogate/vis/tx_mapping.py \
    --row 2 \
    --col 5 \
    --height 8 \
    --width 8

参数说明:
  --row/--col: 站点行列坐标。
  --height/--width: 可视化网格尺寸。
  --map-path: 可选灰度底图。
  --tx-path: 可选真实 tx 图像，用于和推导坐标对比。
  --site-json-path: 可选 JSON 文件，读取站点坐标。
  --site-index: 从 JSON 中取第几个站点。
  --out-path: 输出图片路径。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_OUT_PATH = ROOT_DIR / "vis" / "outputs" / "coord_debug" / "tx_coordinate_mapping_debug.png"
TX_HEIGHT_MIN_M = 19.5
TX_HEIGHT_MAX_M = 22.7484
TX_HEIGHT_VALUE_MIN = 192
TX_HEIGHT_VALUE_MAX = 255


def build_tx_current_eval(height: int, width: int, row: int, col: int, pixel_value: int = 255) -> Tuple[np.ndarray, Tuple[int, int]]:
    tx = np.zeros((height, width), dtype=np.uint8)
    x = row
    y = height - 1 - col
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"current-eval mapping out of bounds: row={row}, col={col}, x={x}, y={y}, hw={(height, width)}")
    tx[y, x] = pixel_value
    return tx, (x, y)


def build_tx_project_standard(height: int, width: int, row: int, col: int, pixel_value: int = 255) -> Tuple[np.ndarray, Tuple[int, int]]:
    tx = np.zeros((height, width), dtype=np.uint8)
    y = row
    x = col
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"project-standard mapping out of bounds: row={row}, col={col}, x={x}, y={y}, hw={(height, width)}")
    tx[y, x] = pixel_value
    return tx, (x, y)


def build_tx_cartesian_rc(height: int, width: int, row: int, col: int, pixel_value: int = 255) -> Tuple[np.ndarray, Tuple[int, int]]:
    tx = np.zeros((height, width), dtype=np.uint8)
    x = col
    y = height - 1 - row
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"cartesian-rc mapping out of bounds: row={row}, col={col}, x={x}, y={y}, hw={(height, width)}")
    tx[y, x] = pixel_value
    return tx, (x, y)


def load_base_map(map_path: str | None, height: int, width: int) -> np.ndarray:
    if map_path:
        arr = np.asarray(Image.open(map_path).convert("L"))
        if arr.shape != (height, width):
            raise ValueError(f"Map shape {arr.shape} does not match requested hw {(height, width)}")
        return arr
    yy, xx = np.mgrid[0:height, 0:width]
    base = ((xx + yy) % 2) * 22 + 28
    base = base.astype(np.uint8)
    base[::2, :] = np.clip(base[::2, :] + 18, 0, 255)
    base[:, ::2] = np.clip(base[:, ::2] + 10, 0, 255)
    return base


def load_actual_tx(tx_path: Optional[str], height: int, width: int) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
    if not tx_path:
        return None, None
    arr = np.asarray(Image.open(tx_path).convert("L"))
    if arr.shape != (height, width):
        raise ValueError(f"TX shape {arr.shape} does not match requested hw {(height, width)}")
    ys, xs = np.where(arr > 0)
    point = None
    if len(xs) == 1:
        point = (int(xs[0]), int(ys[0]))
    return arr, point


def encode_tx_value(z_m: Optional[float]) -> int:
    if z_m is None:
        return 255
    z_clamped = min(max(float(z_m), TX_HEIGHT_MIN_M), TX_HEIGHT_MAX_M)
    if TX_HEIGHT_MAX_M == TX_HEIGHT_MIN_M:
        return TX_HEIGHT_VALUE_MAX
    scaled = (z_clamped - TX_HEIGHT_MIN_M) / (TX_HEIGHT_MAX_M - TX_HEIGHT_MIN_M)
    value = round(scaled * (TX_HEIGHT_VALUE_MAX - TX_HEIGHT_VALUE_MIN) + TX_HEIGHT_VALUE_MIN)
    return int(min(max(value, 0), 255))


def load_site_from_json(site_json_path: Optional[str], site_index: int) -> Optional[Tuple[int, int, Optional[float]]]:
    if not site_json_path:
        return None
    with open(site_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    site = data[site_index]
    z_m = None if len(site) < 3 else float(site[2])
    return int(site[0]), int(site[1]), z_m


def draw_grid(ax, width: int, height: int) -> None:
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)


def overlay_tx(ax, base_map: np.ndarray, tx: np.ndarray, point_xy: Tuple[int, int], title: str, note: str) -> None:
    ax.imshow(base_map, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    masked = np.ma.masked_where(tx == 0, tx)
    ax.imshow(masked, cmap="autumn", vmin=0, vmax=255, interpolation="nearest", alpha=0.95)
    ax.scatter([point_xy[0]], [point_xy[1]], c="cyan", marker="x", s=120, linewidths=2.2)
    ax.set_title(f"{title}\n{note}", fontsize=10)
    ax.set_xlim(-0.5, base_map.shape[1] - 0.5)
    ax.set_ylim(base_map.shape[0] - 0.5, -0.5)
    draw_grid(ax, base_map.shape[1], base_map.shape[0])


def overlay_actual_tx(ax, base_map: np.ndarray, tx_actual: np.ndarray, actual_point: Optional[Tuple[int, int]]) -> None:
    ax.imshow(base_map, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    masked = np.ma.masked_where(tx_actual == 0, tx_actual)
    ax.imshow(masked, cmap="winter", vmin=0, vmax=255, interpolation="nearest", alpha=0.95)
    if actual_point is not None:
        ax.scatter([actual_point[0]], [actual_point[1]], c="yellow", marker="x", s=120, linewidths=2.2)
    ax.set_title("Actual TX Image", fontsize=10)
    ax.set_xlim(-0.5, base_map.shape[1] - 0.5)
    ax.set_ylim(base_map.shape[0] - 0.5, -0.5)
    draw_grid(ax, base_map.shape[1], base_map.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize how tx coordinates are mapped into image pixels.")
    parser.add_argument("--row", type=int, default=None)
    parser.add_argument("--col", type=int, default=None)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--map-path", type=str, default=None)
    parser.add_argument("--tx-path", type=str, default=None)
    parser.add_argument("--site-json-path", type=str, default=None)
    parser.add_argument("--site-index", type=int, default=0)
    parser.add_argument("--out-path", type=str, default=str(DEFAULT_OUT_PATH))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    site_from_json = load_site_from_json(args.site_json_path, args.site_index)
    z_m: Optional[float] = None
    if site_from_json is not None:
        row, col, z_m = site_from_json
    else:
        row = 2 if args.row is None else int(args.row)
        col = 5 if args.col is None else int(args.col)

    pixel_value = encode_tx_value(z_m)
    base_map = load_base_map(args.map_path, args.height, args.width)
    tx_current, point_current = build_tx_current_eval(args.height, args.width, row, col, pixel_value)
    tx_standard, point_standard = build_tx_project_standard(args.height, args.width, row, col, pixel_value)
    tx_cartesian, point_cartesian = build_tx_cartesian_rc(args.height, args.width, row, col, pixel_value)
    tx_actual, point_actual = load_actual_tx(args.tx_path, args.height, args.width)

    ncols = 4 if tx_actual is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), constrained_layout=True)
    overlay_tx(axes[0], base_map, tx_current, point_current, "Current Eval Mapping", f"row={row}, col={col}")
    overlay_tx(axes[1], base_map, tx_standard, point_standard, "Project Standard Mapping", f"row={row}, col={col}")
    overlay_tx(axes[2], base_map, tx_cartesian, point_cartesian, "Cartesian RC Mapping", f"row={row}, col={col}")
    if tx_actual is not None:
        overlay_actual_tx(axes[3], base_map, tx_actual, point_actual)

    fig.savefig(args.out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved output to: {args.out_path}")


if __name__ == "__main__":
    main()
