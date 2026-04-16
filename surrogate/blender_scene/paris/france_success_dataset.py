#!/usr/bin/env python3
"""Dataset-aligned Sionna RT script for the Paris Blender scene.

This script is adapted from ``france_success.ipynb`` and changes the runtime
configuration to match the 3D base-station dataset assumptions:

- 256 x 256 map size with 1 m pixel spacing
- carrier frequency = 3.5 GHz
- Tx power = 23 dBm
- isotropic Tx/Rx antennas
- Rx height = 1.5 m
- Tx height = rooftop + 3 m (must be reflected in the provided Tx z)
- max IRT interactions = 2
- fixed dataset export range = [-162, -75] dB

Note:
- The scene mesh resolution is already baked into ``paris.xml``. The dataset's
  "tile length = 10 m" is a geometry-generation setting, not a runtime flag in
  this script.
- Sionna is imported inside ``main()`` so this file can still be syntax-checked
  on machines where Sionna is not installed.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import struct
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


MAP_SIZE_PX = 256
PIXEL_LENGTH_M = 1.0
RX_HEIGHT_M = 1.5
TX_POWER_DBM = 23.0
NOISE_PSD_DBM_PER_HZ = -174.0
NOISE_FIGURE_DB = 20.0
BANDWIDTH_HZ = 20e6
FREQUENCY_HZ = 3.5e9
DB_MIN = -162.0
DB_MAX = -75.0
BUILDING_HEIGHT_MIN_M = 6.6
BUILDING_HEIGHT_MAX_M = 19.8
TX_HEIGHT_MIN_M = 19.5
TX_HEIGHT_MAX_M = 22.7484
TX_HEIGHT_VALUE_MIN = 192
TX_HEIGHT_VALUE_MAX = 255
# Selected from `element_211`, which stays within the dataset's building-height
# range: roof_z ~= 19.50 m, then move ~1 m inward from the roof edge and add 3 m.
DEFAULT_TX_POS = (-8.36, 24.80, 22.50)
DEFAULT_RX_POS = (-26.0, -37.0, RX_HEIGHT_M)
DEFAULT_ORIENTATION = (0.0, 0.0, 0.0)
DEFAULT_CAMERA_POS = (0.0, -180.0, 120.0)
DEFAULT_CAMERA_LOOK_AT = (0.0, 0.0, 15.0)
DEFAULT_LABEL_SMOOTH_SIGMA = 2.6
DEFAULT_LABEL_SMOOTH_MIX = 0.85
DEFAULT_LABEL_DB_OFFSET = 0.0
DEFAULT_LABEL_POST_SIGMA = 3.2
DEFAULT_LABEL_GAIN = 1.0
DEFAULT_LABEL_GAMMA = 1.0
DEFAULT_LABEL_BROAD_BLEND = 0.35
DEFAULT_LABEL_TOP_GAIN = 1.0
DEFAULT_LABEL_BOTTOM_GAIN = 1.0

_TYPE_MAP = {
    "char": ("b", np.int8),
    "int8": ("b", np.int8),
    "uchar": ("B", np.uint8),
    "uint8": ("B", np.uint8),
    "short": ("h", np.int16),
    "int16": ("h", np.int16),
    "ushort": ("H", np.uint16),
    "uint16": ("H", np.uint16),
    "int": ("i", np.int32),
    "int32": ("i", np.int32),
    "uint": ("I", np.uint32),
    "uint32": ("I", np.uint32),
    "float": ("f", np.float32),
    "float32": ("f", np.float32),
    "double": ("d", np.float64),
    "float64": ("d", np.float64),
}


def parse_triplet(text: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values: x,y,z")
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid numeric triplet: {text}") from exc


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def squeeze_to_2d(array: Any) -> np.ndarray:
    arr = np.asarray(array)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] <= 4:
            return arr[0]
        if arr.shape[-1] <= 4:
            return arr[..., 0]
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Could not reduce array to 2D, got shape {arr.shape}")
    return arr


def center_crop_or_pad(array: np.ndarray, target_h: int = MAP_SIZE_PX, target_w: int = MAP_SIZE_PX) -> np.ndarray:
    src_h, src_w = array.shape
    dst = np.zeros((target_h, target_w), dtype=array.dtype)

    src_y0 = max((src_h - target_h) // 2, 0)
    src_x0 = max((src_w - target_w) // 2, 0)
    src_y1 = min(src_y0 + target_h, src_h)
    src_x1 = min(src_x0 + target_w, src_w)

    crop = array[src_y0:src_y1, src_x0:src_x1]

    dst_y0 = max((target_h - crop.shape[0]) // 2, 0)
    dst_x0 = max((target_w - crop.shape[1]) // 2, 0)
    dst[dst_y0:dst_y0 + crop.shape[0], dst_x0:dst_x0 + crop.shape[1]] = crop
    return dst


def resize_metric_to_export_grid(
    array: np.ndarray,
    target_h: int,
    target_w: int,
    sim_cell_size_m: float,
    export_cell_size_m: float,
) -> np.ndarray:
    if sim_cell_size_m <= 0.0 or export_cell_size_m <= 0.0:
        raise ValueError("Cell sizes must be positive")

    if np.isclose(sim_cell_size_m, export_cell_size_m):
        return center_crop_or_pad(array, target_h, target_w).astype(np.float32)

    scale = export_cell_size_m / sim_cell_size_m

    if scale > 1.0:
        factor = int(round(scale))
        if not np.isclose(scale, factor):
            raise ValueError(
                f"Unsupported resize ratio: export_cell_size_m/sim_cell_size_m={scale:.4f}. "
                "Use an integer ratio such as 2.0 for 0.5m -> 1.0m."
            )
        cropped = center_crop_or_pad(array, target_h * factor, target_w * factor)
        resized = cropped.reshape(target_h, factor, target_w, factor).mean(axis=(1, 3))
        return resized.astype(np.float32)

    factor = int(round(1.0 / scale))
    if not np.isclose(1.0 / scale, factor):
        raise ValueError(
            f"Unsupported resize ratio: sim_cell_size_m/export_cell_size_m={(1.0/scale):.4f}. "
            "Use an integer ratio for upsampling."
        )
    cropped = center_crop_or_pad(array, max(1, target_h // factor), max(1, target_w // factor))
    resized = np.repeat(np.repeat(cropped, factor, axis=0), factor, axis=1)
    return center_crop_or_pad(resized, target_h, target_w).astype(np.float32)


def linear_metric_to_db(metric: np.ndarray) -> np.ndarray:
    finite = metric[np.isfinite(metric)]
    if finite.size == 0:
        return np.full_like(metric, DB_MIN, dtype=np.float32)

    # If values already look like dB, keep them as-is. Otherwise convert
    # linear path gain to dB.
    q01 = float(np.quantile(finite, 0.01))
    q99 = float(np.quantile(finite, 0.99))
    if q99 <= 5.0 and q01 < -20.0:
        return metric.astype(np.float32)

    clipped = np.maximum(metric.astype(np.float64), 1e-30)
    return (10.0 * np.log10(clipped)).astype(np.float32)


def db_to_uint8(db_map: np.ndarray, db_min: float = DB_MIN, db_max: float = DB_MAX) -> np.ndarray:
    norm = (db_map - db_min) / (db_max - db_min)
    norm = np.clip(norm, 0.0, 1.0)
    return np.round(norm * 255.0).astype(np.uint8)


def metric_stats(name: str, values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return f"{name}: no finite values"
    q = np.quantile(finite, [0.0, 0.1, 0.5, 0.9, 1.0])
    return (
        f"{name}: min={q[0]:.2f}, q10={q[1]:.2f}, median={q[2]:.2f}, "
        f"q90={q[3]:.2f}, max={q[4]:.2f}"
    )


def get_metric_array(radio_map: Any, metric_name: str = "path_gain") -> np.ndarray:
    if not hasattr(radio_map, metric_name):
        raise AttributeError(f"Radio map object does not expose `{metric_name}`")

    metric = getattr(radio_map, metric_name)
    if hasattr(metric, "numpy"):
        metric = metric.numpy()
    return squeeze_to_2d(metric)


def set_optional_attr(obj: Any, name: str, value: Any) -> None:
    if hasattr(obj, name):
        try:
            setattr(obj, name, value)
        except (AttributeError, TypeError):
            pass


def height_to_uint8(height_map_m: np.ndarray) -> np.ndarray:
    out = np.zeros_like(height_map_m, dtype=np.uint8)
    mask = height_map_m > 0.0
    if not np.any(mask):
        return out
    heights = np.clip(height_map_m[mask], BUILDING_HEIGHT_MIN_M, BUILDING_HEIGHT_MAX_M)
    scaled = (heights - BUILDING_HEIGHT_MIN_M) / (BUILDING_HEIGHT_MAX_M - BUILDING_HEIGHT_MIN_M)
    out[mask] = np.round(1.0 + scaled * 254.0).astype(np.uint8)
    return out


def encode_tx_value(z_m: float) -> np.uint8:
    z_clamped = min(max(float(z_m), TX_HEIGHT_MIN_M), TX_HEIGHT_MAX_M)
    if TX_HEIGHT_MAX_M == TX_HEIGHT_MIN_M:
        return np.uint8(TX_HEIGHT_VALUE_MAX)
    scaled = (z_clamped - TX_HEIGHT_MIN_M) / (TX_HEIGHT_MAX_M - TX_HEIGHT_MIN_M)
    value = round(scaled * (TX_HEIGHT_VALUE_MAX - TX_HEIGHT_VALUE_MIN) + TX_HEIGHT_VALUE_MIN)
    return np.uint8(min(max(value, 0), 255))


def smooth_gain_db(
    label_db: np.ndarray,
    building_mask: np.ndarray,
    db_min: float,
    db_max: float,
    sigma: float,
    mix: float,
    db_offset: float,
) -> np.ndarray:
    if sigma <= 0.0 or mix <= 0.0:
        return np.clip(label_db, db_min, db_max).astype(np.float32)

    mix = min(max(float(mix), 0.0), 1.0)
    free_mask = (~building_mask).astype(np.float32)

    linear = np.zeros_like(label_db, dtype=np.float32)
    free = free_mask > 0.0
    linear[free] = np.power(10.0, label_db[free] / 10.0).astype(np.float32)

    smooth_num = gaussian_filter(linear * free_mask, sigma=sigma, mode="nearest")
    smooth_den = gaussian_filter(free_mask, sigma=sigma, mode="nearest")
    smooth_linear = np.divide(smooth_num, np.maximum(smooth_den, 1e-6), dtype=np.float32)

    mixed_linear = (1.0 - mix) * linear + mix * smooth_linear
    mixed_linear[~free] = 0.0

    # Preserve the average non-building energy so smoothing does not make
    # the entire map systematically brighter than the raw label.
    raw_mean = float(linear[free].mean()) if np.any(free) else 0.0
    mixed_mean = float(mixed_linear[free].mean()) if np.any(free) else 0.0
    if raw_mean > 0.0 and mixed_mean > 0.0:
        mixed_linear *= raw_mean / mixed_mean
        mixed_linear[~free] = 0.0

    out_db = np.full_like(label_db, db_min, dtype=np.float32)
    valid = mixed_linear > 1e-30
    out_db[valid] = 10.0 * np.log10(mixed_linear[valid])
    out_db[free] += float(db_offset)
    return np.clip(out_db, db_min, db_max).astype(np.float32)


def smooth_gain_uint8(
    label_u8: np.ndarray,
    building_mask: np.ndarray,
    sigma: float,
    mix: float,
    post_sigma: float,
    gain: float,
    gamma: float,
    broad_blend: float,
    top_gain: float,
    bottom_gain: float,
) -> np.ndarray:
    free_mask = (~building_mask).astype(np.float32)
    base = label_u8.astype(np.float32)

    def masked_gaussian(arr: np.ndarray, sigma_value: float) -> np.ndarray:
        if sigma_value <= 0.0:
            return arr.copy()
        num = gaussian_filter(arr * free_mask, sigma=sigma_value, mode="nearest")
        den = gaussian_filter(free_mask, sigma=sigma_value, mode="nearest")
        out = np.divide(num, np.maximum(den, 1e-6), dtype=np.float32)
        out[building_mask] = 0.0
        return out

    mix = min(max(float(mix), 0.0), 1.0)
    smoothed = masked_gaussian(base, sigma)
    out = (1.0 - mix) * base + mix * smoothed
    out[building_mask] = 0.0

    out = np.clip(out * float(gain) / 255.0, 0.0, 1.0)
    if gamma > 0.0 and gamma != 1.0:
        out = np.power(out, float(gamma))
    out *= 255.0
    out[building_mask] = 0.0

    out = masked_gaussian(out, post_sigma)
    out[building_mask] = 0.0

    broad_sigma = max(post_sigma * 2.5, post_sigma + 2.0)
    broad = masked_gaussian(out, broad_sigma)
    broad_blend = min(max(float(broad_blend), 0.0), 1.0)
    out = (1.0 - broad_blend) * out + broad_blend * broad
    out[building_mask] = 0.0

    h = out.shape[0]
    ramp = np.linspace(float(top_gain), float(bottom_gain), h, dtype=np.float32).reshape(h, 1)
    out *= ramp
    out[building_mask] = 0.0

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def parse_ply_header(f) -> tuple[str, list[dict[str, Any]]]:
    fmt = None
    elements = []
    curr = None
    while True:
        line = f.readline()
        if not line:
            raise ValueError("PLY header missing end_header")
        text = line.decode("ascii", errors="ignore").strip()
        if text.startswith("format "):
            fmt = text.split()[1]
        elif text.startswith("element "):
            parts = text.split()
            curr = {"name": parts[1], "count": int(parts[2]), "props": []}
            elements.append(curr)
        elif text.startswith("property ") and curr is not None:
            parts = text.split()
            if parts[1] == "list":
                curr["props"].append(
                    {
                        "name": parts[4],
                        "is_list": True,
                        "count_type": parts[2],
                        "item_type": parts[3],
                    }
                )
            else:
                curr["props"].append({"name": parts[2], "is_list": False, "type": parts[1]})
        elif text == "end_header":
            break
    if fmt is None:
        raise ValueError("PLY missing format")
    return fmt, elements


def read_vertices_faces(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    with path.open("rb") as f:
        fmt, elements = parse_ply_header(f)
        endian = "<" if fmt == "binary_little_endian" else ">"
        if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
            raise ValueError(f"Unsupported PLY format: {fmt}")

        vertices = None
        faces: list[list[int]] = []
        for elem in elements:
            name = elem["name"]
            count = elem["count"]
            props = elem["props"]
            if name == "vertex":
                if fmt == "ascii":
                    prop_names = [p["name"] for p in props]
                    ix = prop_names.index("x")
                    iy = prop_names.index("y")
                    iz = prop_names.index("z")
                    verts = []
                    for _ in range(count):
                        parts = f.readline().decode("ascii", errors="ignore").strip().split()
                        if len(parts) <= max(ix, iy, iz):
                            continue
                        verts.append([float(parts[ix]), float(parts[iy]), float(parts[iz])])
                    vertices = np.asarray(verts, dtype=np.float32)
                else:
                    dtype_fields = []
                    for p in props:
                        if p["is_list"]:
                            raise ValueError(f"List property in vertex not supported: {path}")
                        fmt_code, _ = _TYPE_MAP[p["type"]]
                        dtype_fields.append((p["name"], endian + fmt_code))
                    data = np.fromfile(f, dtype=np.dtype(dtype_fields), count=count)
                    vertices = np.vstack([data["x"], data["y"], data["z"]]).T.astype(np.float32)
            elif name == "face":
                list_prop = next((p for p in props if p["is_list"]), None)
                if list_prop is None:
                    continue
                if fmt == "ascii":
                    for _ in range(count):
                        parts = f.readline().decode("ascii", errors="ignore").strip().split()
                        if not parts:
                            continue
                        n = int(parts[0])
                        if n >= 3:
                            faces.append([int(v) for v in parts[1 : 1 + n]])
                else:
                    count_code, _ = _TYPE_MAP[list_prop["count_type"]]
                    item_code, _ = _TYPE_MAP[list_prop["item_type"]]
                    count_size = struct.calcsize(endian + count_code)
                    item_size = struct.calcsize(endian + item_code)
                    for _ in range(count):
                        raw = f.read(count_size)
                        if not raw:
                            break
                        n = struct.unpack(endian + count_code, raw)[0]
                        raw_items = f.read(item_size * n)
                        if n >= 3:
                            faces.append(list(struct.unpack(endian + item_code * n, raw_items)))
            else:
                if fmt == "ascii":
                    for _ in range(count):
                        f.readline()
        if vertices is None:
            vertices = np.empty((0, 3), dtype=np.float32)
        return vertices, faces


def triangle_iter(faces: list[list[int]]):
    for face in faces:
        if len(face) < 3:
            continue
        v0 = face[0]
        for i in range(1, len(face) - 1):
            yield v0, face[i], face[i + 1]


def rasterize_triangle_to_height(
    vertices: np.ndarray,
    faces: list[list[int]],
    height_map: np.ndarray,
    x_min: float,
    y_min: float,
    cell_size: float,
) -> None:
    h, w = height_map.shape
    for i0, i1, i2 in triangle_iter(faces):
        try:
            v0 = vertices[i0]
            v1 = vertices[i1]
            v2 = vertices[i2]
        except IndexError:
            continue

        p0 = v1 - v0
        p1 = v2 - v0
        normal = np.cross(p0, p1)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-8:
            continue
        x0, y0, z0 = map(float, v0)
        x1, y1, z1 = map(float, v1)
        x2, y2, z2 = map(float, v2)

        px0 = (x0 - x_min) / cell_size
        py0 = (y0 - y_min) / cell_size
        px1 = (x1 - x_min) / cell_size
        py1 = (y1 - y_min) / cell_size
        px2 = (x2 - x_min) / cell_size
        py2 = (y2 - y_min) / cell_size

        min_px = int(max(0, math.floor(min(px0, px1, px2))))
        max_px = int(min(w - 1, math.ceil(max(px0, px1, px2))))
        min_py = int(max(0, math.floor(min(py0, py1, py2))))
        max_py = int(min(h - 1, math.ceil(max(py0, py1, py2))))

        denom = (py1 - py2) * (px0 - px2) + (px2 - px1) * (py0 - py2)
        if abs(denom) < 1e-8:
            continue

        for row in range(min_py, max_py + 1):
            for col in range(min_px, max_px + 1):
                px = col + 0.5
                py = row + 0.5
                w0 = ((py1 - py2) * (px - px2) + (px2 - px1) * (py - py2)) / denom
                w1 = ((py2 - py0) * (px - px2) + (px0 - px2) * (py - py2)) / denom
                w2 = 1.0 - w0 - w1
                if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                    continue
                z = w0 * z0 + w1 * z1 + w2 * z2
                if z > height_map[row, col]:
                    height_map[row, col] = z


def build_height_map_from_meshes(
    meshes_dir: Path,
    map_size_px: int,
    center_xy: tuple[float, float],
    cell_size_m: float,
) -> np.ndarray:
    height_map = np.zeros((map_size_px, map_size_px), dtype=np.float32)
    x_min = center_xy[0] - (map_size_px * cell_size_m) / 2.0
    y_min = center_xy[1] - (map_size_px * cell_size_m) / 2.0

    for ply_path in sorted(meshes_dir.glob("*.ply")):
        if ply_path.name == "Plane.ply":
            continue
        verts, faces = read_vertices_faces(ply_path)
        if verts.size == 0 or not faces:
            continue
        rasterize_triangle_to_height(verts, faces, height_map, x_min=x_min, y_min=y_min, cell_size=cell_size_m)
    return height_map


def world_to_grid(
    x_m: float,
    y_m: float,
    map_size_px: int,
    center_xy: tuple[float, float],
    cell_size_m: float,
) -> tuple[int, int]:
    x_min = center_xy[0] - (map_size_px * cell_size_m) / 2.0
    y_min = center_xy[1] - (map_size_px * cell_size_m) / 2.0
    col = int(math.floor((x_m - x_min) / cell_size_m))
    row = int(math.floor((y_m - y_min) / cell_size_m))
    row = max(0, min(map_size_px - 1, row))
    col = max(0, min(map_size_px - 1, col))
    return row, col


def load_buildings_u8(path: Path, target_h: int, target_w: int) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), copy=True)
    if arr.shape == (target_h, target_w):
        # External PNG building maps are already stored in final image
        # coordinates. Convert them back to the internal simulation
        # coordinates expected by the export pipeline.
        return np.flipud(arr).astype(np.uint8)

    resized = Image.fromarray(arr).resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
    return np.flipud(np.array(resized, copy=True)).astype(np.uint8)


def export_dataset_triplet(
    output_dir: Path,
    scene_id: str,
    tx_id: str,
    buildings_u8: np.ndarray,
    export_cell_size_m: float,
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
    png_root = output_dir / "png"
    buildings_dir = png_root / "buildingsWHeight"
    tx_dir = png_root / "antennasWHeight"
    gain_dir = output_dir / "gain"

    for path in (buildings_dir, tx_dir, gain_dir):
        ensure_dir(path)

    building_mask = buildings_u8 > 0
    tx_u8 = np.zeros_like(buildings_u8, dtype=np.uint8)
    tx_row, tx_col = world_to_grid(
        tx_position[0],
        tx_position[1],
        map_size_px=buildings_u8.shape[0],
        center_xy=(0.0, 0.0),
        cell_size_m=export_cell_size_m,
    )
    tx_u8[tx_row, tx_col] = encode_tx_value(tx_position[2])
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

    # Match the dataset image convention used by the existing pipeline:
    # x stays left->right, while the y axis is flipped before saving.
    buildings_u8 = np.flipud(buildings_u8)
    tx_u8 = np.flipud(tx_u8)
    raw_label_u8 = np.flipud(raw_label_u8)
    label_u8 = np.flipud(label_u8)
    tx_row_img = buildings_u8.shape[0] - 1 - tx_row
    tx_col_img = tx_col

    buildings_path = buildings_dir / f"{scene_id}.png"
    tx_path = tx_dir / f"{scene_id}_{tx_id}.png"
    gain_path = gain_dir / f"{scene_id}_{tx_id}.png"
    Image.fromarray(buildings_u8, mode="L").save(buildings_path)
    Image.fromarray(tx_u8, mode="L").save(tx_path)
    Image.fromarray(label_u8, mode="L").save(gain_path)

    return [
        f"dataset_buildings_path={buildings_path}",
        f"dataset_tx_path={tx_path}",
        f"dataset_gain_path={gain_path}",
        f"dataset_tx_image_rc=({tx_row_img},{tx_col_img})",
        f"label_smooth_sigma={label_smooth_sigma:.2f}",
        f"label_smooth_mix={label_smooth_mix:.2f}",
        f"label_db_offset={label_db_offset:.2f}",
        f"label_post_sigma={label_post_sigma:.2f}",
        f"label_gain={label_gain:.2f}",
        f"label_gamma={label_gamma:.2f}",
        f"label_broad_blend={label_broad_blend:.2f}",
        f"label_top_gain={label_top_gain:.2f}",
        f"label_bottom_gain={label_bottom_gain:.2f}",
    ]


def build_parser() -> argparse.ArgumentParser:
    from paris_paths import OUTPUT_DATASET_DIR, PARIS_SCENE_XML

    default_scene = PARIS_SCENE_XML
    default_out = OUTPUT_DATASET_DIR

    parser = argparse.ArgumentParser(description="Run a dataset-aligned Sionna radio-map simulation.")
    parser.add_argument("--scene", type=Path, default=default_scene, help="Path to Mitsuba XML scene.")
    parser.add_argument("--output_dir", type=Path, default=default_out, help="Directory for outputs.")
    parser.add_argument("--scene_id", type=str, default="paris", help="Dataset scene id for exported png names.")
    parser.add_argument("--tx_id", type=str, default="0", help="Dataset tx id for exported png names.")
    parser.add_argument("--tx_position", type=parse_triplet, default=DEFAULT_TX_POS, help="Tx position x,y,z in meters.")
    parser.add_argument("--rx_position", type=parse_triplet, default=DEFAULT_RX_POS, help="Rx position x,y,z in meters.")
    parser.add_argument("--tx_orientation", type=parse_triplet, default=DEFAULT_ORIENTATION, help="Tx orientation yaw,pitch,roll.")
    parser.add_argument("--rx_orientation", type=parse_triplet, default=DEFAULT_ORIENTATION, help="Rx orientation yaw,pitch,roll.")
    parser.add_argument("--frequency_hz", type=float, default=FREQUENCY_HZ, help="Carrier frequency in Hz.")
    parser.add_argument("--tx_power_dbm", type=float, default=TX_POWER_DBM, help="Transmit power in dBm.")
    parser.add_argument("--bandwidth_hz", type=float, default=BANDWIDTH_HZ, help="Bandwidth in Hz.")
    parser.add_argument("--noise_psd_dbm_per_hz", type=float, default=NOISE_PSD_DBM_PER_HZ, help="Noise PSD in dBm/Hz.")
    parser.add_argument("--noise_figure_db", type=float, default=NOISE_FIGURE_DB, help="Noise figure in dB.")
    parser.add_argument("--cell_size_m", type=float, default=PIXEL_LENGTH_M, help="Final exported dataset grid resolution in meters.")
    parser.add_argument(
        "--sim_cell_size_m",
        type=float,
        default=None,
        help="Coverage-map simulation resolution in meters. Defaults to --cell_size_m. "
             "Set smaller than --cell_size_m (e.g. 0.5) to simulate on a finer grid and downsample.",
    )
    parser.add_argument("--map_size_px", type=int, default=MAP_SIZE_PX, help="Final center crop/pad size.")
    parser.add_argument("--max_depth", type=int, default=2, help="Max number of ray interactions.")
    parser.add_argument("--samples_per_tx", type=int, default=1_000_000, help="Number of rays per Tx.")
    parser.add_argument("--reflection", action=argparse.BooleanOptionalAction, default=True, help="Enable specular reflections in Sionna RT.")
    parser.add_argument("--diffraction", action=argparse.BooleanOptionalAction, default=True, help="Enable diffraction in Sionna RT.")
    parser.add_argument("--scattering", action=argparse.BooleanOptionalAction, default=False, help="Enable scattering in Sionna RT.")
    parser.add_argument("--edge_diffraction", action=argparse.BooleanOptionalAction, default=False, help="Enable edge diffraction in Sionna RT.")
    parser.add_argument("--db_min", type=float, default=DB_MIN, help="Dataset export lower bound.")
    parser.add_argument("--db_max", type=float, default=DB_MAX, help="Dataset export upper bound.")
    parser.add_argument(
        "--building_map_path",
        type=Path,
        default=default_out / "png" / "buildingsWHeight" / "paris_aligned_scene_range.png",
        help="Precomputed grayscale building-height PNG in final image coordinates. "
             "It will be resized to match the exported gain size and internally flipped "
             "back to the simulation grid before masking/export.",
    )
    parser.add_argument("--label_smooth_sigma", type=float, default=DEFAULT_LABEL_SMOOTH_SIGMA, help="Gaussian sigma for non-building gain smoothing.")
    parser.add_argument("--label_smooth_mix", type=float, default=DEFAULT_LABEL_SMOOTH_MIX, help="Blend ratio between raw gain and smoothed gain.")
    parser.add_argument("--label_db_offset", type=float, default=DEFAULT_LABEL_DB_OFFSET, help="Additional dB offset applied before label post-processing.")
    parser.add_argument("--label_post_sigma", type=float, default=DEFAULT_LABEL_POST_SIGMA, help="Second masked Gaussian sigma applied in image space.")
    parser.add_argument("--label_gain", type=float, default=DEFAULT_LABEL_GAIN, help="Image-space gain applied before the final smoothing step.")
    parser.add_argument("--label_gamma", type=float, default=DEFAULT_LABEL_GAMMA, help="Image-space gamma applied before the final smoothing step.")
    parser.add_argument("--label_broad_blend", type=float, default=DEFAULT_LABEL_BROAD_BLEND, help="Blend ratio for an additional broad spatial smoothing field.")
    parser.add_argument("--label_top_gain", type=float, default=DEFAULT_LABEL_TOP_GAIN, help="Vertical gain multiplier applied at the top of the map.")
    parser.add_argument("--label_bottom_gain", type=float, default=DEFAULT_LABEL_BOTTOM_GAIN, help="Vertical gain multiplier applied at the bottom of the map.")
    parser.add_argument("--save_paths", action="store_true", help="Also solve example Tx-Rx paths and save delays/amps.")
    parser.add_argument("--save_render", action="store_true", help="Save scene renderings that show the buildings.")
    parser.add_argument("--camera_position", type=parse_triplet, default=DEFAULT_CAMERA_POS, help="Render camera position x,y,z.")
    parser.add_argument("--camera_look_at", type=parse_triplet, default=DEFAULT_CAMERA_LOOK_AT, help="Render camera look-at x,y,z.")
    parser.add_argument("--render_samples", type=int, default=64, help="Samples per pixel for scene rendering.")
    parser.add_argument("--render_resolution", type=int, nargs=2, default=(1280, 960), help="Render resolution width height.")
    parser.add_argument(
        "--skip_figure_redraw",
        action="store_true",
        help="Skip the follow-up redraw of Paris comparison figures after dataset export.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.output_dir)
    sim_cell_size_m = args.sim_cell_size_m if args.sim_cell_size_m is not None else args.cell_size_m

    from sionna.rt import Camera, PlanarArray, Transmitter, load_scene

    scene = load_scene(str(args.scene))
    set_optional_attr(scene, "frequency", args.frequency_hz)
    set_optional_attr(scene, "bandwidth", args.bandwidth_hz)

    # Dataset uses isotropic antennas rather than a 4x4 TR38901 BS array.
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

    tx = Transmitter(
        name="dataset_tx",
        position=list(args.tx_position),
        orientation=list(args.tx_orientation),
        power_dbm=args.tx_power_dbm,
    )
    set_optional_attr(tx, "power_dbm", args.tx_power_dbm)
    scene.add(tx)

    radio_map = scene.coverage_map(
        cm_center=[0.0, 0.0, RX_HEIGHT_M],
        cm_orientation=[0.0, 0.0, 0.0],
        cm_size=[float(args.map_size_px), float(args.map_size_px)],
        cm_cell_size=[sim_cell_size_m, sim_cell_size_m],
        rx_orientation=list(args.rx_orientation),
        max_depth=args.max_depth,
        num_samples=args.samples_per_tx,
        los=True,
        reflection=args.reflection,
        diffraction=args.diffraction,
        scattering=args.scattering,
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
    noise_power_dbm = args.noise_psd_dbm_per_hz + 10.0 * math.log10(args.bandwidth_hz) + args.noise_figure_db
    snr_db = rx_power_dbm - noise_power_dbm
    buildings_u8 = load_buildings_u8(args.building_map_path, args.map_size_px, args.map_size_px)

    np.save(args.output_dir / "path_gain_db.npy", metric_db)
    np.save(args.output_dir / "rx_power_dbm.npy", rx_power_dbm)
    np.save(args.output_dir / "snr_db.npy", snr_db)
    dataset_lines = export_dataset_triplet(
        args.output_dir,
        scene_id=args.scene_id,
        tx_id=args.tx_id,
        buildings_u8=buildings_u8,
        export_cell_size_m=args.cell_size_m,
        tx_position=args.tx_position,
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
        f"scene_id={args.scene_id}",
        f"tx_id={args.tx_id}",
        f"tx_position_m={args.tx_position}",
        f"tx_power_dbm={args.tx_power_dbm:.2f}",
        f"frequency_hz={args.frequency_hz:.0f}",
        f"bandwidth_hz={args.bandwidth_hz:.0f}",
        f"noise_psd_dbm_per_hz={args.noise_psd_dbm_per_hz:.2f}",
        f"noise_figure_db={args.noise_figure_db:.2f}",
        f"noise_power_dbm={noise_power_dbm:.2f}",
        f"export_cell_size_m={args.cell_size_m:.2f}",
        f"sim_cell_size_m={sim_cell_size_m:.2f}",
        f"building_map_path={args.building_map_path}",
        f"max_depth={args.max_depth}",
        f"samples_per_tx={args.samples_per_tx}",
        f"reflection={args.reflection}",
        f"diffraction={args.diffraction}",
        f"scattering={args.scattering}",
        f"edge_diffraction={args.edge_diffraction}",
        f"final_map_size={metric_db.shape[1]}x{metric_db.shape[0]}",
        metric_stats("building_gray_u8", buildings_u8[buildings_u8 > 0] if np.any(buildings_u8 > 0) else buildings_u8),
        metric_stats("path_gain_db", metric_db),
        metric_stats("rx_power_dbm", rx_power_dbm),
        metric_stats("snr_db", snr_db),
        *dataset_lines,
    ]

    stats_path = args.output_dir / "stats.txt"
    stats_path.write_text("\n".join(stats_lines) + "\n", encoding="utf-8")
    print("\n".join(stats_lines))

    if not args.skip_figure_redraw:
        figure_script = args.scene.parent / "make_paris_figures.py"
        if figure_script.exists():
            env = dict(os.environ)
            env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
            subprocess.run(
                ["python", str(figure_script)],
                check=True,
                env=env,
            )
            print(f"Redrew Paris comparison figures via {figure_script}")
        else:
            print(f"Skipped figure redraw because {figure_script} does not exist")

    if args.save_render:
        camera = Camera(
            name="dataset_cam",
            position=list(args.camera_position),
            look_at=list(args.camera_look_at),
        )
        scene.add(camera)

        fig_scene = scene.render(
            camera=camera,
            num_samples=args.render_samples,
            resolution=list(args.render_resolution),
        )
        fig_scene.savefig(args.output_dir / "scene_render.png", dpi=300, bbox_inches="tight")
        plt.close(fig_scene)

        fig_overlay = scene.render(
            camera=camera,
            coverage_map=radio_map,
            cm_db_scale=True,
            cm_vmin=args.db_min,
            cm_vmax=args.db_max,
            cm_metric="path_gain",
            cm_show_color_bar=True,
            num_samples=args.render_samples,
            resolution=list(args.render_resolution),
        )
        fig_overlay.savefig(args.output_dir / "scene_render_with_coverage.png", dpi=300, bbox_inches="tight")
        plt.close(fig_overlay)
        print(f"Saved renders to {args.output_dir}")

    if args.save_paths:
        from sionna.rt import Receiver

        rx = Receiver(
            name="dataset_rx",
            position=list(args.rx_position),
            orientation=list(args.rx_orientation),
        )
        scene.add(rx)
        paths = scene.compute_paths(
            max_depth=args.max_depth,
            los=True,
            reflection=True,
            diffraction=True,
            scattering=False,
            edge_diffraction=False,
            num_samples=args.samples_per_tx,
        )
        a, tau = paths.cir(
            sampling_frequency=100e6,
            normalize_delays=True,
            out_type="numpy",
        )
        np.save(args.output_dir / "paths_a.npy", np.asarray(a))
        np.save(args.output_dir / "paths_tau.npy", np.asarray(tau))
        print(f"Saved paths: a.shape={np.asarray(a).shape}, tau.shape={np.asarray(tau).shape}")


if __name__ == "__main__":
    main()
# MPLCONFIGDIR=/tmp/matplotlib \
# MPLCONFIGDIR=/tmp/matplotlib \
# /Users/epiphanyer/Miniconda/envs/torch-mps/bin/python \
# /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/blender_scene/paris/france_success_dataset.py \
#   --scene_id paris \
#   --tx_id 0 \
#   --tx_position=-8.36,24.80,23.0 \
#   --tx_power_dbm 23 \
#   --frequency_hz 3.5e9 \
#   --max_depth 2 \
#   --samples_per_tx 1000000 \
#   --label_smooth_sigma 4.0 \
#   --label_smooth_mix 1.0 \
#   --label_post_sigma 6.0 \
#   --label_gain 1.05 \
#   --label_gamma 1.0 \
#   --label_broad_blend 0.80 \
#   --label_top_gain 1.0 \
#   --label_bottom_gain 1.0 \
#   --output_dir /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/blender_scene/paris/output_dataset


# samples_per_tx 先决定原始仿真图质量
# label_smooth_sigma + label_smooth_mix 先把原始散点扩开
# label_post_sigma + label_broad_blend 再把它变成更连续的场
# label_gain + label_gamma 负责整体亮度和对比度
# label_top_gain + label_bottom_gain 只在需要做上下区域校正时用
