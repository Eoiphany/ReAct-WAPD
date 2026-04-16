#!/usr/bin/env python3
"""Build a dataset-aligned grayscale building-height map from WinProp ODA polygons."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from paris_paths import ODB_DIR, PNG_DIR

DEFAULT_ODA_PATH = ODB_DIR / "map.oda"
DEFAULT_ASC_PATH = ODB_DIR / "15.asc"
DEFAULT_OUTPUT_PATH = PNG_DIR / "buildingsWHeight" / "paris_oda_height.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rasterize WinProp ODA buildings into a grayscale height map.")
    parser.add_argument("--oda_path", type=Path, default=DEFAULT_ODA_PATH, help="Input ODA file.")
    parser.add_argument("--asc_path", type=Path, default=DEFAULT_ASC_PATH, help="ASC file used to infer the exact ROI.")
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output grayscale PNG path.")
    parser.add_argument("--map_size_px", type=int, default=256, help="Output square size.")
    parser.add_argument(
        "--supersample",
        type=int,
        default=4,
        help="Rasterize on a finer grid first, then downsample to preserve small holes and thin structures.",
    )
    parser.add_argument(
        "--encoding",
        choices=("scene_range", "fixed_range"),
        default="fixed_range",
        help="Map ODA heights using either scene min/max or a fixed interval. "
        "Default uses the dataset-aligned fixed interval.",
    )
    parser.add_argument(
        "--height_range_m",
        type=float,
        nargs=2,
        default=(6.6, 19.8),
        metavar=("MIN_H", "MAX_H"),
        help="Fixed height interval when --encoding fixed_range is used. "
        "Dataset default is 6.6-19.8 m.",
    )
    parser.add_argument(
        "--min_building_gray",
        type=int,
        default=32,
        help="Minimum grayscale value assigned to buildings. Background stays 0.",
    )
    return parser


def parse_oda_buildings(path: Path) -> tuple[tuple[float, float, float, float], list[tuple[list[tuple[float, float]], float]]]:
    settings = None
    buildings: list[tuple[list[tuple[float, float]], float]] = []
    in_buildings = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("SETTINGS "):
                parts = line.split()
                settings = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
                continue

            if line == "BEGIN_BUILDINGS":
                in_buildings = True
                continue
            if line == "END_BUILDINGS":
                break
            if not in_buildings or line.startswith("*"):
                continue

            parts = line.split()
            if len(parts) < 5:
                continue
            n_pts = int(parts[1])
            coord_tokens = parts[2 : 2 + n_pts]
            if len(coord_tokens) != n_pts:
                continue
            polygon: list[tuple[float, float]] = []
            for token in coord_tokens:
                x_str, y_str = token.split(",")
                polygon.append((float(x_str), float(y_str)))
            if len(polygon) < 3:
                continue
            height_m = float(parts[2 + n_pts])
            buildings.append((polygon, height_m))

    if settings is None:
        raise ValueError(f"No SETTINGS line found in {path}")
    return settings, buildings


def roi_from_asc(path: Path) -> tuple[float, float, float, float]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header = {}
        for _ in range(6):
            k, v = f.readline().split()[:2]
            header[k.lower()] = float(v)
        data = np.loadtxt(f, dtype=np.float32)

    nodata = header["nodata_value"]
    data[data == nodata] = np.nan
    rows, cols = np.where(np.isfinite(data))
    rmin, rmax, cmin, cmax = map(int, (rows.min(), rows.max(), cols.min(), cols.max()))
    x0 = header["xllcorner"]
    y0 = header["yllcorner"]
    cell = header["cellsize"]
    nrows = int(header["nrows"])

    xmin = x0 + cmin * cell
    xmax = x0 + (cmax + 1) * cell
    ymin = y0 + (nrows - 1 - rmax) * cell
    ymax = y0 + (nrows - rmin) * cell
    return xmin, ymin, xmax, ymax


def point_in_polygon(point: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = (x2 - x1) * (y - y1) / max((y2 - y1), 1e-12) + x1
            if x < x_cross:
                inside = not inside
    return inside


def polygon_bbox(poly: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_contains(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def split_outer_and_holes(
    polygons: list[tuple[list[tuple[float, float]], float]]
) -> tuple[list[tuple[list[tuple[float, float]], float]], list[list[tuple[float, float]]]]:
    bboxes = [polygon_bbox(poly) for poly, _ in polygons]
    outers: list[tuple[list[tuple[float, float]], float]] = []
    holes: list[list[tuple[float, float]]] = []

    for i, (poly, height_m) in enumerate(polygons):
        if height_m > 0.0:
            outers.append((poly, height_m))
            continue

        probe = poly[0]
        is_hole = False
        for j, (other_poly, other_height_m) in enumerate(polygons):
            if i == j or other_height_m <= 0.0:
                continue
            if not bbox_contains(bboxes[j], bboxes[i]):
                continue
            if point_in_polygon(probe, other_poly):
                is_hole = True
                break

        if is_hole:
            holes.append(poly)

    return outers, holes


def height_to_gray(height_m: float, h_min: float, h_max: float, min_gray: int) -> int:
    min_gray = int(min(max(min_gray, 1), 254))
    if h_max <= h_min:
        return 255
    clipped = min(max(height_m, h_min), h_max)
    scaled = (clipped - h_min) / (h_max - h_min)
    return int(round(min_gray + scaled * (255 - min_gray)))


def main() -> None:
    args = build_parser().parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    (db_x0, db_y0, db_w, db_h), buildings = parse_oda_buildings(args.oda_path)
    roi_xmin, roi_ymin, roi_xmax, roi_ymax = roi_from_asc(args.asc_path)
    outer_buildings, hole_polygons = split_outer_and_holes(buildings)

    heights = np.array([height for _, height in outer_buildings], dtype=np.float32)
    positive_heights = heights[heights > 0.0]
    if positive_heights.size == 0:
        raise ValueError(f"No building polygons found in {args.oda_path}")

    if args.encoding == "scene_range":
        h_min = float(positive_heights.min())
        h_max = float(positive_heights.max())
    else:
        h_min, h_max = map(float, args.height_range_m)

    ss = max(1, int(args.supersample))
    canvas_size = args.map_size_px * ss

    # Draw heights and holes on separate canvases, then subtract holes after downsampling.
    height_canvas = Image.new("L", (canvas_size, canvas_size), 0)
    hole_canvas = Image.new("L", (canvas_size, canvas_size), 0)
    height_draw = ImageDraw.Draw(height_canvas)
    hole_draw = ImageDraw.Draw(hole_canvas)

    rasterized_outer = 0
    for polygon, height_m in sorted(outer_buildings, key=lambda item: item[1]):
        gray = height_to_gray(height_m, h_min=h_min, h_max=h_max, min_gray=args.min_building_gray)
        pix = []
        for x_local, y_local in polygon:
            x_world = db_x0 + x_local
            y_world = db_y0 + y_local
            x = (x_world - roi_xmin) / max(roi_xmax - roi_xmin, 1e-12) * (canvas_size - 1)
            y = (roi_ymax - y_world) / max(roi_ymax - roi_ymin, 1e-12) * (canvas_size - 1)
            pix.append((x, y))
        if len(pix) >= 3:
            height_draw.polygon(pix, fill=gray)
            rasterized_outer += 1

    rasterized_holes = 0
    for polygon in hole_polygons:
        pix = []
        for x_local, y_local in polygon:
            x_world = db_x0 + x_local
            y_world = db_y0 + y_local
            x = (x_world - roi_xmin) / max(roi_xmax - roi_xmin, 1e-12) * (canvas_size - 1)
            y = (roi_ymax - y_world) / max(roi_ymax - roi_ymin, 1e-12) * (canvas_size - 1)
            pix.append((x, y))
        if len(pix) >= 3:
            hole_draw.polygon(pix, fill=255)
            rasterized_holes += 1

    if ss > 1:
        height_canvas = height_canvas.resize((args.map_size_px, args.map_size_px), resample=Image.Resampling.BOX)
        hole_canvas = hole_canvas.resize((args.map_size_px, args.map_size_px), resample=Image.Resampling.BOX)

    out = np.array(height_canvas, dtype=np.uint8)
    hole_mask = np.array(hole_canvas, dtype=np.uint8) >= 128
    out[hole_mask] = 0

    Image.fromarray(out, mode="L").save(args.output_path)
    print(f"oda_path={args.oda_path}")
    print(f"asc_path={args.asc_path}")
    print(f"output_path={args.output_path}")
    print(f"db_origin_world=({db_x0},{db_y0})")
    print(f"db_span_xy=({db_w},{db_h})")
    print(f"roi_world=({roi_xmin},{roi_ymin}) -> ({roi_xmax},{roi_ymax})")
    print(f"supersample={ss}")
    print(f"outer_polygons_rasterized={rasterized_outer}")
    print(f"hole_polygons_rasterized={rasterized_holes}")
    print(f"encoding={args.encoding}")
    print(f"height_range_m=({h_min:.3f},{h_max:.3f})")
    print(
        "raw_positive_height_stats_m="
        f"min={float(positive_heights.min()):.3f}, "
        f"median={float(np.median(positive_heights)):.3f}, "
        f"max={float(positive_heights.max()):.3f}"
    )
    print(f"nonzero_ratio={float((out > 0).mean()):.6f}")


if __name__ == "__main__":
    main()
