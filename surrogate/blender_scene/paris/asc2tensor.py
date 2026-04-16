"""
把 WinProp 导出的 ASC 栅格直接转换成数据集 gain 标签 PNG。

默认约定：
- 输入 `paris_pathloss.asc` 包含 zoom 区域及其外圈 NODATA
- 自动裁掉四周 NODATA，只保留有效区域
- 输入数值默认为可直接编码的 `dB` 标签图
- 读取建筑高度图生成 mask，建筑区域强制置黑
- 默认不翻转 y 轴，保持当前 WinProp 视图方向
- 默认输出为 256x256 的 8-bit grayscale PNG
- 正式标签默认使用当前图自身分位数拉伸后的灰度版本
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from paris_paths import GAIN_DIR, ODB_DIR, PNG_DIR

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


DB_MIN = -162.0
DB_MAX = -75.0
DEFAULT_TARGET_SIZE = 256
DEFAULT_ASC_PATH = ODB_DIR / "paris_pathloss.asc"
DEFAULT_OUTPUT_PATH = GAIN_DIR / "paris_0.png"
DEFAULT_BUILDING_MAP_PATH = PNG_DIR / "buildingsWHeight" / "paris_aligned_scene_range.png"
DEFAULT_BUILDING_MASK_PATH = PNG_DIR / "buildingsMask" / "paris_mask.png"
DEFAULT_REFERENCE_GAIN_PATH = None
DEFAULT_REFERENCE_BUILDING_MAP_PATH = None


def load_asc(filepath: str | Path) -> tuple[dict[str, float | int], np.ndarray]:
    """
    读取 ESRI ASCII Grid / WinProp ASC 文件。
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    header: dict[str, float | int] = {}
    required_keys = {"ncols", "nrows", "cellsize"}

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in range(6):
            line = f.readline()
            if not line:
                raise ValueError("文件头不足 6 行，可能不是标准 ASC Grid 文件。")

            parts = line.strip().split()
            if len(parts) < 2:
                raise ValueError(f"无法解析头部行: {line}")

            key = parts[0].lower()
            value = parts[1]
            if key in {"ncols", "nrows"}:
                header[key] = int(float(value))
            else:
                header[key] = float(value)

        missing = required_keys - set(header.keys())
        if missing:
            raise ValueError(f"ASC 头部缺少必要字段: {missing}")

        data = np.loadtxt(f, dtype=np.float32)

    expected_shape = (int(header["nrows"]), int(header["ncols"]))
    if data.shape != expected_shape:
        raise ValueError(f"数据尺寸不匹配: 文件头声明 {expected_shape}, 实际读到 {data.shape}")

    return header, data


def replace_nodata_with_nan(data: np.ndarray, header: dict[str, float | int]) -> np.ndarray:
    """
    把 NODATA_value 替换成 np.nan。
    """
    nodata_val = header.get("nodata_value")
    out = data.astype(np.float32, copy=True)
    if nodata_val is not None:
        out[np.isclose(out, float(nodata_val))] = np.nan
    return out


def get_origin(header: dict[str, float | int]) -> tuple[float, float]:
    """
    返回网格左下角坐标，兼容 xllcorner / xllcenter 两种写法。
    """
    cellsize = float(header["cellsize"])

    if "xllcorner" in header:
        x0 = float(header["xllcorner"])
    elif "xllcenter" in header:
        x0 = float(header["xllcenter"]) - cellsize / 2.0
    else:
        raise ValueError("头部缺少 xllcorner/xllcenter")

    if "yllcorner" in header:
        y0 = float(header["yllcorner"])
    elif "yllcenter" in header:
        y0 = float(header["yllcenter"]) - cellsize / 2.0
    else:
        raise ValueError("头部缺少 yllcorner/yllcenter")

    return x0, y0


def grid_to_coord(row: int, col: int, header: dict[str, float | int], center: bool = True) -> tuple[float, float]:
    """
    行列号 -> 实际坐标。
    """
    nrows = int(header["nrows"])
    ncols = int(header["ncols"])
    cellsize = float(header["cellsize"])
    x0, y0 = get_origin(header)

    if not (0 <= row < nrows and 0 <= col < ncols):
        raise IndexError(f"row/col 越界: row={row}, col={col}, shape=({nrows},{ncols})")

    offset = 0.5 if center else 0.0
    x = x0 + (col + offset) * cellsize
    y = y0 + (nrows - row - 1 + offset) * cellsize
    return x, y


def coord_to_grid(x: float, y: float, header: dict[str, float | int], center: bool = True) -> tuple[int, int]:
    """
    实际坐标 -> 行列号（0-based）。
    """
    nrows = int(header["nrows"])
    ncols = int(header["ncols"])
    cellsize = float(header["cellsize"])
    x0, y0 = get_origin(header)

    offset = 0.5 if center else 0.0
    col = int(np.floor((x - x0) / cellsize - offset + 1e-9))
    row_from_bottom = int(np.floor((y - y0) / cellsize - offset + 1e-9))
    row = nrows - 1 - row_from_bottom

    if not (0 <= row < nrows and 0 <= col < ncols):
        raise IndexError(f"坐标超出网格范围: x={x}, y={y}, 计算得到 row={row}, col={col}")

    return row, col


def db_to_uint8(db_map: np.ndarray, db_min: float, db_max: float) -> np.ndarray:
    """
    把 dB 图映射到数据集标签使用的 uint8 灰度图。
    """
    out = np.zeros_like(db_map, dtype=np.float32)
    valid = np.isfinite(db_map)
    out[valid] = np.clip((db_map[valid] - db_min) / (db_max - db_min), 0.0, 1.0) * 255.0
    return np.round(out).astype(np.uint8)


def db_to_uint8_percentile(
    db_map: np.ndarray,
    low_q: float = 0.01,
    high_q: float = 0.99,
    db_min_offset: float = 0.0,
    db_max_offset: float = 0.0,
) -> tuple[np.ndarray, float | None, float | None]:
    """
    用当前图自身的分位数范围做灰度拉伸，便于可视化。
    """
    valid = np.isfinite(db_map)
    if not np.any(valid):
        return np.zeros_like(db_map, dtype=np.uint8), None, None

    vals = db_map[valid].astype(np.float32)
    db_min = float(np.quantile(vals, low_q)) + float(db_min_offset)
    db_max = float(np.quantile(vals, high_q)) + float(db_max_offset)
    if np.isclose(db_min, db_max):
        db_min = float(vals.min())
        db_max = float(vals.max())
    if np.isclose(db_min, db_max):
        out = np.zeros_like(db_map, dtype=np.uint8)
        out[valid] = 255
        return out, db_min, db_max
    return db_to_uint8(db_map, db_min=db_min, db_max=db_max), db_min, db_max


def crop_to_valid_bbox(db_map: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    裁剪到非 NaN 有效区域的最小外接矩形。
    注意，row是倒过来的，即y轴向下增加。计算时要使用nrows - row + 1来得到真正的row。
    返回:
        cropped_map
        (row_min, row_max, col_min, col_max)
    """
    valid = np.isfinite(db_map)
    if not np.any(valid):
        raise ValueError("ASC 中没有有效值，无法裁剪 zoom 区域。")

    rows, cols = np.where(valid)
    row_min = int(rows.min())
    row_max = int(rows.max())
    col_min = int(cols.min())
    col_max = int(cols.max())
    cropped = db_map[row_min : row_max + 1, col_min : col_max + 1]
    return cropped, (row_min, row_max, col_min, col_max)


def resize_uint8_map(image_u8: np.ndarray, target_size: int) -> np.ndarray:
    """
    用双线性插值 resize 到方形输出。
    """
    if image_u8.shape == (target_size, target_size):
        return image_u8.copy()

    pil_image = Image.fromarray(image_u8)
    resized = pil_image.resize((target_size, target_size), resample=Image.Resampling.BILINEAR)
    return np.array(resized, dtype=np.uint8, copy=True)


def load_building_mask(building_map_path: str | Path, target_size: int) -> np.ndarray:
    """
    从 buildingsWHeight 灰度图构建建筑 mask。
    非零像素视为建筑。
    """
    arr = np.array(Image.open(building_map_path).convert("L"), dtype=np.uint8, copy=True)
    if arr.shape != (target_size, target_size):
        resized = Image.fromarray(arr).resize((target_size, target_size), resample=Image.Resampling.NEAREST)
        arr = np.array(resized, dtype=np.uint8, copy=True)
    return arr > 0


def load_gray_image(path: str | Path, target_size: int) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8, copy=True)
    if arr.shape != (target_size, target_size):
        resized = Image.fromarray(arr).resize((target_size, target_size), resample=Image.Resampling.NEAREST)
        arr = np.array(resized, dtype=np.uint8, copy=True)
    return arr


def match_nonbuilding_distribution(
    source_u8: np.ndarray,
    source_building_mask: np.ndarray,
    reference_gain_path: str | Path,
    reference_building_map_path: str | Path,
    target_size: int,
) -> np.ndarray:
    """
    用参考 gain 图的非建筑灰度分布，对当前图的非建筑正值区域做分位数匹配。
    0 值保持 0，不破坏建筑全黑与无效区域全黑。
    """
    ref_gain = load_gray_image(reference_gain_path, target_size=target_size)
    ref_building_mask = load_building_mask(reference_building_map_path, target_size=target_size)

    src_pos = (~source_building_mask) & (source_u8 > 0)
    ref_pos = (~ref_building_mask) & (ref_gain > 0)
    if not np.any(src_pos) or not np.any(ref_pos):
        return source_u8.copy()

    src_vals = source_u8[src_pos].astype(np.float32)
    ref_vals = ref_gain[ref_pos].astype(np.float32)

    q = np.linspace(0.0, 1.0, 1024)
    src_q = np.quantile(src_vals, q)
    ref_q = np.quantile(ref_vals, q)

    matched = source_u8.astype(np.float32, copy=True)
    matched[src_pos] = np.interp(src_vals, src_q, ref_q)
    return np.clip(np.round(matched), 0, 255).astype(np.uint8)


def asc_to_dataset_gain(
    asc_path: str | Path,
    output_path: str | Path,
    building_map_path: str | Path | None = DEFAULT_BUILDING_MAP_PATH,
    building_mask_path: str | Path | None = DEFAULT_BUILDING_MASK_PATH,
    reference_gain_path: str | Path | None = DEFAULT_REFERENCE_GAIN_PATH,
    reference_building_map_path: str | Path | None = DEFAULT_REFERENCE_BUILDING_MAP_PATH,
    db_min: float = DB_MIN,
    db_max: float = DB_MAX,
    target_size: int = DEFAULT_TARGET_SIZE,
    crop_valid_bbox: bool = True,
    flip_y: bool = False,
    save_npy: bool = False,
    save_tensor: bool = False,
    use_percentile_label: bool = True,
    percentile_low: float = 0.01,
    percentile_high: float = 0.99,
    percentile_db_min_offset: float = 0.0,
    percentile_db_max_offset: float = 8.0,
) -> dict[str, object]:
    """
    读取 ASC 并导出数据集兼容的 gain PNG。
    """
    header, data_raw = load_asc(asc_path)
    data_db = replace_nodata_with_nan(data_raw, header)
    crop_bbox = None
    if crop_valid_bbox:
        data_db, crop_bbox = crop_to_valid_bbox(data_db)

    vis_u8, vis_db_min, vis_db_max = db_to_uint8_percentile(
        data_db,
        low_q=percentile_low,
        high_q=percentile_high,
        db_min_offset=percentile_db_min_offset,
        db_max_offset=percentile_db_max_offset,
    )
    data_u8 = vis_u8.copy() if use_percentile_label else db_to_uint8(data_db, db_min=db_min, db_max=db_max)

    if flip_y:
        data_u8 = np.flipud(data_u8)
        vis_u8 = np.flipud(vis_u8)
    data_u8 = resize_uint8_map(data_u8, target_size=target_size)
    vis_u8 = resize_uint8_map(vis_u8, target_size=target_size)

    building_ratio = None
    building_mask = None
    if building_mask_path is not None and Path(building_mask_path).exists():
        building_mask = load_building_mask(building_mask_path, target_size=target_size)
    elif building_map_path is not None:
        building_mask = load_building_mask(building_map_path, target_size=target_size)

    if building_mask is not None:
        data_u8[building_mask] = 0
        vis_u8[building_mask] = 0
        building_ratio = float(building_mask.mean())

    if (
        reference_gain_path is not None
        and reference_building_map_path is not None
        and building_mask is not None
    ):
        data_u8 = match_nonbuilding_distribution(
            data_u8,
            source_building_mask=building_mask,
            reference_gain_path=reference_gain_path,
            reference_building_map_path=reference_building_map_path,
            target_size=target_size,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(data_u8).save(output_path)

    results: dict[str, object] = {
        "header": header,
        "shape_raw": data_db.shape,
        "shape_png": data_u8.shape,
        "output_path": str(output_path),
        "valid_min_db": float(np.nanmin(data_db)),
        "valid_max_db": float(np.nanmax(data_db)),
        "valid_mean_db": float(np.nanmean(data_db)),
        "valid_ratio": float(np.isfinite(data_db).mean()),
        "crop_bbox_rowcol": crop_bbox,
        "flip_y": flip_y,
        "building_map_path": None if building_map_path is None else str(building_map_path),
        "building_mask_path": None if building_mask_path is None else str(building_mask_path),
        "building_ratio": building_ratio,
        "reference_gain_path": None if reference_gain_path is None else str(reference_gain_path),
        "reference_building_map_path": None if reference_building_map_path is None else str(reference_building_map_path),
        "zero_ratio_after_mask": float((data_u8 == 0).mean()),
        "use_percentile_label": use_percentile_label,
        "percentile_low": percentile_low,
        "percentile_high": percentile_high,
        "percentile_db_min_offset": percentile_db_min_offset,
        "percentile_db_max_offset": percentile_db_max_offset,
    }

    if save_npy:
        npy_path = output_path.with_suffix(".npy")
        np.save(npy_path, data_db)
        results["npy_path"] = str(npy_path)

    if save_tensor:
        if not TORCH_AVAILABLE:
            raise ImportError("未安装 torch，无法导出 tensor。")
        tensor_path = output_path.with_suffix(".pt")
        tensor = torch.tensor(data_db, dtype=torch.float32).unsqueeze(0)
        torch.save(tensor, tensor_path)
        results["tensor_path"] = str(tensor_path)

    return results


def plot_map(
    data: np.ndarray,
    title: str = "ASC Data",
    cmap: str = "jet",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    可视化二维网格。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("未安装 matplotlib，无法保存预览图。") from exc

    plt.figure(figsize=(6, 6))
    img = plt.imshow(data, cmap=cmap)
    plt.colorbar(img)
    plt.title(title)
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05, dpi=300)
        print(f"[OK] 已保存图片: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def print_header_info(header: dict[str, float | int]) -> None:
    """
    打印头部和范围信息。
    """
    print("===== ASC Header =====")
    for key, value in header.items():
        print(f"{key}: {value}")

    x0, y0 = get_origin(header)
    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    cellsize = float(header["cellsize"])
    xmax = x0 + ncols * cellsize
    ymax = y0 + nrows * cellsize

    print("===== Derived Extent =====")
    print(f"xmin: {x0}")
    print(f"ymin: {y0}")
    print(f"xmax: {xmax}")
    print(f"ymax: {ymax}")
    print(f"width:  {xmax - x0}")
    print(f"height: {ymax - y0}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert WinProp ASC to dataset-style gain PNG.")
    parser.add_argument("--asc_path", type=Path, default=DEFAULT_ASC_PATH, help="输入 ASC 文件路径。")
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH, help="输出 gain PNG 路径。")
    parser.add_argument(
        "--building_map_path",
        type=Path,
        default=DEFAULT_BUILDING_MAP_PATH,
        help="建筑高度图路径。非零像素会在 gain 标签中被置黑。",
    )
    parser.add_argument(
        "--building_mask_path",
        type=Path,
        default=DEFAULT_BUILDING_MASK_PATH,
        help="可选：独立建筑二值 mask 路径。若存在，优先于 building_map_path 用作建筑遮罩。",
    )
    parser.add_argument(
        "--reference_gain_path",
        type=Path,
        default=DEFAULT_REFERENCE_GAIN_PATH,
        help="参考 gain 图路径，用于把非建筑灰度分布校准到现有数据集风格。",
    )
    parser.add_argument(
        "--reference_building_map_path",
        type=Path,
        default=DEFAULT_REFERENCE_BUILDING_MAP_PATH,
        help="参考 gain 图对应的建筑图路径。",
    )
    parser.add_argument("--db_min", type=float, default=DB_MIN, help="映射下界，对应灰度 0。")
    parser.add_argument("--db_max", type=float, default=DB_MAX, help="映射上界，对应灰度 255。")
    parser.add_argument("--target_size", type=int, default=DEFAULT_TARGET_SIZE, help="输出 PNG 边长。")
    parser.add_argument("--no_crop_valid_bbox", action="store_true", help="不要先裁掉 ASC 四周的 NODATA 外框。")
    parser.add_argument("--flip_y", action="store_true", help="在保存前做上下翻转。")
    parser.add_argument("--no_building_mask", action="store_true", help="不使用建筑图做遮罩。")
    parser.add_argument("--no_reference_match", action="store_true", help="不做参考 gain 分布匹配。")
    parser.add_argument("--save_npy", action="store_true", help="额外保存 ASC 的原始 dB 数组为 .npy。")
    parser.add_argument("--save_tensor", action="store_true", help="额外保存 ASC 的原始 dB tensor 为 .pt。")
    parser.add_argument(
        "--use_percentile_label",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="正式标签是否使用自适应分位数拉伸。默认启用。",
    )
    parser.add_argument("--percentile_low", type=float, default=0.01, help="自适应映射下分位数。")
    parser.add_argument("--percentile_high", type=float, default=0.99, help="自适应映射上分位数。")
    parser.add_argument(
        "--percentile_db_min_offset",
        type=float,
        default=0.0,
        help="在下分位数对应 dB 基础上再加的偏移量。",
    )
    parser.add_argument(
        "--percentile_db_max_offset",
        type=float,
        default=8.0,
        help="在上分位数对应 dB 基础上再加的偏移量。调大通常会让图整体更暗。",
    )
    parser.add_argument("--check_row", type=int, default=103, help="用于打印检查的示例 row。")
    parser.add_argument("--check_col", type=int, default=119, help="用于打印检查的示例 col。")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    header, data_raw = load_asc(args.asc_path)
    data_db = replace_nodata_with_nan(data_raw, header)

    print_header_info(header)
    print("===== Data Info =====")
    print("shape:", data_db.shape)
    print("dtype:", data_db.dtype)
    print("min_db:", float(np.nanmin(data_db)))
    print("max_db:", float(np.nanmax(data_db)))
    print("mean_db:", float(np.nanmean(data_db)))
    print("valid_ratio:", float(np.isfinite(data_db).mean()))

    row = args.check_row
    col = args.check_col
    value = data_db[row, col]
    x, y = grid_to_coord(row, col, header, center=True)
    row2, col2 = coord_to_grid(x, y, header, center=True)

    print("===== Example Pixel =====")
    print(f"row={row}, col={col}")
    print(f"value={value}")
    print(f"x={x}, y={y}")
    print("===== Reverse Check =====")
    print(f"(x,y) -> row={row2}, col={col2}")

    results = asc_to_dataset_gain(
        asc_path=args.asc_path,
        output_path=args.output_path,
        building_map_path=None if args.no_building_mask else args.building_map_path,
        building_mask_path=None if args.no_building_mask else args.building_mask_path,
        reference_gain_path=None if args.no_reference_match else args.reference_gain_path,
        reference_building_map_path=None if args.no_reference_match else args.reference_building_map_path,
        db_min=args.db_min,
        db_max=args.db_max,
        target_size=args.target_size,
        crop_valid_bbox=not args.no_crop_valid_bbox,
        flip_y=args.flip_y,
        save_npy=args.save_npy,
        save_tensor=args.save_tensor,
        use_percentile_label=args.use_percentile_label,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        percentile_db_min_offset=args.percentile_db_min_offset,
        percentile_db_max_offset=args.percentile_db_max_offset,
    )

    print("===== Export Result =====")
    for key, value in results.items():
        if key == "header":
            continue
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
