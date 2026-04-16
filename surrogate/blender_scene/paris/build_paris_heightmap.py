#!/usr/bin/env python3
"""Build a grayscale building-height map directly from the Paris mesh files.

This script is intentionally decoupled from Sionna simulation. It only rasterizes
the Blender-exported building meshes into a top-view height map and encodes the
result into the RadioMap3DSeer-style grayscale range:

- background / non-building = 0
- building height in [6.6 m, 19.8 m] -> [1, 255]
- buildings taller than 19.8 m are saturated to 255
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_fill_holes
from paris_paths import MESHES_DIR, PNG_DIR

from france_success_dataset import (
    BUILDING_HEIGHT_MAX_M,
    BUILDING_HEIGHT_MIN_M,
    build_height_map_from_meshes,
    ensure_dir,
    height_to_uint8,
    read_vertices_faces,
)


DEFAULT_MESHES_DIR = MESHES_DIR
DEFAULT_OUTPUT_PATH = PNG_DIR / "buildingsWHeight" / "paris_full_bbox.png"


def parse_pair(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected two comma-separated values: x,y")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid numeric pair: {text}") from exc


def scene_bbox(meshes_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    for ply_path in sorted(meshes_dir.glob("*.ply")):
        if ply_path.name == "Plane.ply":
            continue
        verts, _ = read_vertices_faces(ply_path)
        if verts.size == 0:
            continue
        mins.append(verts.min(axis=0))
        maxs.append(verts.max(axis=0))

    if not mins:
        raise ValueError(f"No mesh vertices found under {meshes_dir}")

    return np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0)


def height_to_uint8_with_floor(
    height_map_m: np.ndarray,
    h_min: float | None,
    h_max: float | None,
    min_gray: int,
) -> np.ndarray:
    out = np.zeros_like(height_map_m, dtype=np.uint8)
    mask = height_map_m > 0.0
    if not np.any(mask):
        return out

    heights = height_map_m[mask].astype(np.float32)
    local_min = float(heights.min()) if h_min is None else float(h_min)
    local_max = float(heights.max()) if h_max is None else float(h_max)
    min_gray = int(min(max(min_gray, 1), 254))
    if np.isclose(local_min, local_max):
        out[mask] = 255
        return out

    clipped = np.clip(heights, local_min, local_max)
    scaled = (clipped - local_min) / (local_max - local_min)
    out[mask] = np.round(min_gray + scaled * (255 - min_gray)).astype(np.uint8)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rasterize Paris building meshes into a grayscale height map.")
    parser.add_argument("--meshes_dir", type=Path, default=DEFAULT_MESHES_DIR, help="Directory containing mesh PLY files.")
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH, help="PNG output path.")
    parser.add_argument("--map_size_px", type=int, default=256, help="Square output size in pixels.")
    parser.add_argument(
        "--cell_size_m",
        type=float,
        default=1.0,
        help="World-space meters per pixel for fixed-window export.",
    )
    parser.add_argument(
        "--center_xy",
        type=parse_pair,
        default=(0.0, 0.0),
        help="Window center x,y in meters for fixed-window export.",
    )
    parser.add_argument(
        "--fit_scene_bbox",
        action="store_true",
        help="Fit the entire mesh bbox into the output square instead of using a fixed center/cell size.",
    )
    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="Also save the meter-valued height map as a .npy file next to the PNG.",
    )
    parser.add_argument(
        "--encoding",
        choices=("scene_range", "dataset_range"),
        default="scene_range",
        help="How to map building height to grayscale. "
             "`scene_range` uses the current map's min/max building height; "
             "`dataset_range` uses the fixed [6.6, 19.8] m interval.",
    )
    parser.add_argument(
        "--min_building_gray",
        type=int,
        default=32,
        help="Minimum grayscale value assigned to nonzero building height. Background remains 0.",
    )
    parser.add_argument(
        "--binary_mask_path",
        type=Path,
        default=None,
        help="Optional output path for a binary building mask PNG (background=0, building=255).",
    )
    parser.add_argument(
        "--mask_closing_iterations",
        type=int,
        default=1,
        help="Binary closing iterations used before exporting the optional building mask.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.output_path.parent)

    bbox_min, bbox_max = scene_bbox(args.meshes_dir)
    bbox_center_xy = ((bbox_min[0] + bbox_max[0]) / 2.0, (bbox_min[1] + bbox_max[1]) / 2.0)
    bbox_span_x = float(bbox_max[0] - bbox_min[0])
    bbox_span_y = float(bbox_max[1] - bbox_min[1])

    if args.fit_scene_bbox:
        side_m = max(bbox_span_x, bbox_span_y)
        cell_size_m = side_m / float(args.map_size_px)
        center_xy = bbox_center_xy
        mode = "fit_scene_bbox"
    else:
        cell_size_m = args.cell_size_m
        center_xy = args.center_xy
        mode = "fixed_window"

    height_map_m = build_height_map_from_meshes(
        args.meshes_dir,
        map_size_px=args.map_size_px,
        center_xy=center_xy,
        cell_size_m=cell_size_m,
    )
    if args.encoding == "scene_range":
        height_map_u8 = np.flipud(
            height_to_uint8_with_floor(
                height_map_m,
                h_min=None,
                h_max=None,
                min_gray=args.min_building_gray,
            )
        )
    else:
        height_map_u8 = np.flipud(
            height_to_uint8_with_floor(
                height_map_m,
                h_min=BUILDING_HEIGHT_MIN_M,
                h_max=BUILDING_HEIGHT_MAX_M,
                min_gray=args.min_building_gray,
            )
        )
    Image.fromarray(height_map_u8, mode="L").save(args.output_path)

    if args.binary_mask_path is not None:
        ensure_dir(args.binary_mask_path.parent)
        mask = height_map_m > 0.0
        if args.mask_closing_iterations > 0:
            mask = binary_closing(mask, iterations=args.mask_closing_iterations)
        mask = binary_fill_holes(mask)
        mask_u8 = np.flipud(mask.astype(np.uint8) * 255)
        Image.fromarray(mask_u8, mode="L").save(args.binary_mask_path)

    if args.save_npy:
        np.save(args.output_path.with_suffix(".npy"), height_map_m)

    finite = height_map_m[height_map_m > 0.0]
    print(f"mode={mode}")
    print(f"meshes_dir={args.meshes_dir}")
    print(f"output_path={args.output_path}")
    print(f"map_size_px={args.map_size_px}")
    print(f"center_xy=({center_xy[0]:.3f},{center_xy[1]:.3f})")
    print(f"cell_size_m={cell_size_m:.6f}")
    print(f"scene_bbox_min=({bbox_min[0]:.3f},{bbox_min[1]:.3f},{bbox_min[2]:.3f})")
    print(f"scene_bbox_max=({bbox_max[0]:.3f},{bbox_max[1]:.3f},{bbox_max[2]:.3f})")
    print(f"scene_bbox_span_xy=({bbox_span_x:.3f},{bbox_span_y:.3f})")
    print(f"min_building_gray={args.min_building_gray}")
    if args.binary_mask_path is not None:
        print(f"binary_mask_path={args.binary_mask_path}")
    print(f"nonzero_pixels={int((height_map_m > 0.0).sum())}")
    if finite.size > 0:
        if args.encoding == "scene_range":
            print(
                "encoding_height_range_m="
                f"({float(finite.min()):.3f},{float(finite.max()):.3f})"
            )
        else:
            print(f"encoding_height_range_m=({BUILDING_HEIGHT_MIN_M:.1f},{BUILDING_HEIGHT_MAX_M:.1f})")
        print(
            "height_stats_m="
            f"min={float(finite.min()):.3f}, "
            f"median={float(np.median(finite)):.3f}, "
            f"max={float(finite.max()):.3f}"
        )


if __name__ == "__main__":
    main()
