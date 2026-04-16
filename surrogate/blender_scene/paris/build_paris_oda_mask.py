#!/usr/bin/env python3
"""Build a binary building mask from WinProp ODA outdoor database."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from paris_paths import ODB_DIR, PNG_DIR

DEFAULT_ODA_PATH = ODB_DIR / "map.oda"
DEFAULT_ASC_PATH = ODB_DIR / "paris_pathloss.asc"
DEFAULT_OUTPUT_PATH = PNG_DIR / "buildingsMask" / "paris_oda_mask.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rasterize WinProp ODA buildings into a binary mask.")
    parser.add_argument("--oda_path", type=Path, default=DEFAULT_ODA_PATH, help="Input ODA file.")
    parser.add_argument("--asc_path", type=Path, default=DEFAULT_ASC_PATH, help="ASC file used to infer the exact ROI.")
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output binary mask PNG.")
    parser.add_argument("--map_size_px", type=int, default=256, help="Output square size.")
    parser.add_argument(
        "--supersample",
        type=int,
        default=4,
        help="Rasterize on a finer grid first, then downsample to preserve small holes and thin structures.",
    )
    parser.add_argument(
        "--exclude_polygon_ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional ODA polygon ids to exclude from the final building mask.",
    )
    return parser


def parse_oda_buildings(
    path: Path,
) -> tuple[tuple[float, float, float, float], list[tuple[int, list[tuple[float, float]], float]]]:
    settings = None
    polygons: list[tuple[int, list[tuple[float, float]], float]] = []
    in_buildings = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("SETTINGS "):
                parts = line.split()
                # SETTINGS <unknown> <x0> <y0> <w> <h> WINPROP
                settings = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
                continue

            if line == "BEGIN_BUILDINGS":
                in_buildings = True
                continue
            if line == "END_BUILDINGS":
                in_buildings = False
                continue
            if not in_buildings or line.startswith("*"):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue
            polygon_id = int(parts[0])
            n_pts = int(parts[1])
            coord_tokens = parts[2 : 2 + n_pts]
            polygon: list[tuple[float, float]] = []
            for token in coord_tokens:
                x_str, y_str = token.split(",")
                polygon.append((float(x_str), float(y_str)))
            if len(polygon) >= 3:
                height_m = float(parts[2 + n_pts]) if len(parts) > 2 + n_pts else 0.0
                polygons.append((polygon_id, polygon, height_m))

    if settings is None:
        raise ValueError(f"No SETTINGS line found in {path}")
    return settings, polygons


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
    polygons: list[tuple[int, list[tuple[float, float]], float]],
    excluded_ids: set[int] | None = None,
) -> tuple[list[tuple[int, list[tuple[float, float]]]], list[list[tuple[float, float]]]]:
    excluded_ids = excluded_ids or set()
    bboxes = [polygon_bbox(poly) for _, poly, _ in polygons]
    outers: list[tuple[int, list[tuple[float, float]]]] = []
    holes: list[list[tuple[float, float]]] = []

    for i, (polygon_id, poly, height_m) in enumerate(polygons):
        if polygon_id in excluded_ids:
            continue
        if height_m > 0.0:
            outers.append((polygon_id, poly))
            continue

        probe = poly[0]
        is_hole = False
        for j, (other_id, other_poly, other_height_m) in enumerate(polygons):
            if i == j or other_height_m <= 0.0 or other_id in excluded_ids:
                continue
            if not bbox_contains(bboxes[j], bboxes[i]):
                continue
            if point_in_polygon(probe, other_poly):
                is_hole = True
                break

        if is_hole:
            holes.append(poly)

    return outers, holes


def main() -> None:
    args = build_parser().parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    (db_x0, db_y0, db_w, db_h), polygons = parse_oda_buildings(args.oda_path)
    roi_xmin, roi_ymin, roi_xmax, roi_ymax = roi_from_asc(args.asc_path)
    excluded_ids = set(args.exclude_polygon_ids or [])
    outer_polygons, hole_polygons = split_outer_and_holes(polygons, excluded_ids=excluded_ids)

    ss = max(1, int(args.supersample))
    canvas_size = args.map_size_px * ss
    outer_canvas = Image.new("L", (canvas_size, canvas_size), 0)
    hole_canvas = Image.new("L", (canvas_size, canvas_size), 0)
    outer_draw = ImageDraw.Draw(outer_canvas)
    hole_draw = ImageDraw.Draw(hole_canvas)

    rasterized_outer = 0
    for _, polygon in outer_polygons:
        pix = []
        for x_local, y_local in polygon:
            x_world = db_x0 + x_local
            y_world = db_y0 + y_local
            x = (x_world - roi_xmin) / max(roi_xmax - roi_xmin, 1e-12) * (canvas_size - 1)
            y = (roi_ymax - y_world) / max(roi_ymax - roi_ymin, 1e-12) * (canvas_size - 1)
            pix.append((x, y))
        if len(pix) >= 3:
            outer_draw.polygon(pix, fill=255)
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
        outer_canvas = outer_canvas.resize((args.map_size_px, args.map_size_px), resample=Image.Resampling.BOX)
        hole_canvas = hole_canvas.resize((args.map_size_px, args.map_size_px), resample=Image.Resampling.BOX)

    outer_arr = np.array(outer_canvas, dtype=np.uint8)
    hole_arr = np.array(hole_canvas, dtype=np.uint8)
    mask = outer_arr >= 128
    holes = hole_arr >= 128
    mask[holes] = False
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")

    image.save(args.output_path)
    print(f"oda_path={args.oda_path}")
    print(f"asc_path={args.asc_path}")
    print(f"output_path={args.output_path}")
    print(f"db_origin_world=({db_x0},{db_y0})")
    print(f"db_span_xy=({db_w},{db_h})")
    print(f"roi_world=({roi_xmin},{roi_ymin}) -> ({roi_xmax},{roi_ymax})")
    print(f"supersample={ss}")
    print(f"excluded_polygon_ids={sorted(excluded_ids)}")
    print(f"outer_polygons_rasterized={rasterized_outer}")
    print(f"hole_polygons_rasterized={rasterized_holes}")


if __name__ == "__main__":
    main()
