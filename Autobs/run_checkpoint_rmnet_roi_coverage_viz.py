"""注释
命令示例:
python -m Autobs.run_checkpoint_rmnet_roi_coverage_viz \
  --image dataset/png/buildingsWHeight/82.png \
  --overlay-points-from-image /abs/path/to/current_deploy.png \
  --checkpoint Autobs/checkpoints \
  --model-path Autobs/models/RMNet.pt \
  --network-type rmnet \
  --output-dir Autobs/outputs/checkpoint_roi_coverage_viz

参数含义:
- `--image`: 输入的灰度高度图路径，通常使用 `dataset/png/buildingsWHeight/*.png`。
- `--checkpoint`: PPO checkpoint 目录；默认读取 `Autobs/checkpoints`。
- `--model-path`: RMNet 权重路径；默认读取 `Autobs/models/RMNet.pt`。
- `--network-type`: RMNet 变体，保持与权重一致。
- `--device`: 推理设备，可选 `auto/cpu/cuda/mps`。
- `--output-dir`: 结果输出目录；本脚本会在其中保存单张 RoI coverage 二值图和 JSON 摘要。
- `--explore`: 是否使用带随机性的 exploratory action；默认关闭，使用确定性动作。
- `--overlay-points`: 以 `(x,y)` 指定的部署站点列表；若提供，则脚本用这些站点重新计算多站点 coverage，并在图中标出。
- `--overlay-points-from-image`: 从“当前部署效果图”中自动提取红叉站点，并映射回原始 `256x256` 坐标后参与计算。

用途:
- 对单张地图构造 PPO checkpoint 所需 observation。
- 使用 checkpoint 选取单站点部署位置。
- 使用 RMNet 代理模型预测 radiomap，并计算 coverage / spectral efficiency / score。
- 仅输出 RoI coverage 二值图，也就是原始 `run_checkpoint_rmnet_viz.py` 中最后一张图。

输出示例:
- `*_checkpoint_rmnet_roi_coverage.png`: 单张 RoI coverage 二值图。
- `*_checkpoint_rmnet_roi_coverage_summary.json`: 动作、像素坐标与指标摘要。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from Autobs.env.utils import calc_upsampling_loc, get_stats, load_map_normalized
from Autobs.paths import CHECKPOINT_DIR, DEFAULT_RMNET_WEIGHTS, PACKAGE_ROOT
from Autobs.run_checkpoint_rmnet_viz import (
    CHINESE_FONT_CANDIDATES,
    LATIN_FONT_FAMILY,
    RMNetPredictor,
    build_policy_observation,
    compute_checkpoint_action,
    get_device,
    load_checkpoint_agent,
)


DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs" / "checkpoint_roi_coverage_viz"
COVERAGE_THRESHOLD_DB = -117.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one PPO checkpoint on one image and save RoI coverage only")
    parser.add_argument("--image", required=True, type=str, help="Input grayscale height map")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR), type=str, help="RLlib checkpoint directory")
    parser.add_argument("--model-path", default=str(DEFAULT_RMNET_WEIGHTS), type=str, help="RMNet weights path")
    parser.add_argument("--network-type", default="rmnet", choices=["rmnet", "rmnet_v3"], help="RMNet variant")
    parser.add_argument("--device", default="auto", type=str, help="Inference device: auto/cpu/cuda/mps")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), type=str, help="Directory for PNG/JSON outputs")
    parser.add_argument("--explore", action="store_true", help="Use exploratory sampling instead of deterministic action")
    parser.add_argument(
        "--overlay-points-from-image",
        type=str,
        default="",
        help="Extract red-x deployment points from an existing deployment image",
    )
    parser.add_argument(
        "--overlay-points",
        nargs="*",
        default=[],
        help="Optional overlay points in x,y format, e.g. 152,55 52,112",
    )
    return parser


def parse_overlay_points_arg(raw_points: list[str]) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for raw in raw_points:
        try:
            x_text, y_text = raw.split(",", 1)
            x = int(x_text.strip())
            y = int(y_text.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid overlay point `{raw}`. Expected x,y.") from exc
        points.append((x, y))
    return points


def overlay_points_xy_to_tx_locs(points_xy: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(int(y), int(x)) for x, y in points_xy]


def extract_overlay_points_xy_from_deployment_image(
    image_path: str | Path,
    target_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    target_h, target_w = target_shape
    red_mask = np.zeros((height, width), dtype=bool)
    nonwhite_mask = np.zeros((height, width), dtype=bool)

    for y in range(height):
        for x in range(width):
            r, g, b = image.getpixel((x, y))
            if (r, g, b) != (255, 255, 255):
                nonwhite_mask[y, x] = True
            if r > 220 and g < 80 and b < 80:
                red_mask[y, x] = True

    nonwhite_cols = np.where(nonwhite_mask.any(axis=0))[0]
    if nonwhite_cols.size == 0:
        return []
    col_clusters: list[tuple[int, int]] = []
    start = prev = int(nonwhite_cols[0])
    for col in nonwhite_cols[1:]:
        col = int(col)
        if col == prev + 1:
            prev = col
            continue
        col_clusters.append((start, prev))
        start = prev = col
    col_clusters.append((start, prev))
    map_x0, map_x1 = max(col_clusters, key=lambda item: item[1] - item[0])

    map_nonwhite_rows = np.where(nonwhite_mask[:, map_x0 : map_x1 + 1].any(axis=1))[0]
    if map_nonwhite_rows.size == 0:
        return []
    map_y0 = int(map_nonwhite_rows[0])
    map_y1 = int(map_nonwhite_rows[-1])
    map_width = max(map_x1 - map_x0, 1)
    map_height = max(map_y1 - map_y0, 1)

    visited = np.zeros_like(red_mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for y in range(height):
        for x in range(width):
            if not red_mask[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            component: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                component.append((cx, cy))
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    for ny in range(max(0, cy - 1), min(height, cy + 2)):
                        if red_mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((nx, ny))
            if len(component) >= 20:
                components.append(component)

    points_xy: list[tuple[int, int]] = []
    for component in components:
        xs = [point[0] for point in component]
        ys = [point[1] for point in component]
        raw_center_x = sum(xs) / len(xs)
        raw_center_y = sum(ys) / len(ys)
        center_x = round((raw_center_x - map_x0) * (target_w - 1) / map_width)
        center_y = round((raw_center_y - map_y0) * (target_h - 1) / map_height)
        points_xy.append((int(center_x), int(center_y)))

    points_xy.sort(key=lambda point: (point[1], point[0]))
    return points_xy


def snap_overlay_points_to_valid_tx_pixels(
    pixel_map: np.ndarray,
    points_xy: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    tx_mask = pixel_map <= 0.01
    legal_rc = np.argwhere(tx_mask)
    if legal_rc.size == 0:
        raise ValueError("No legal tx pixels found in the input map.")

    snapped_points: list[tuple[int, int]] = []
    for x, y in points_xy:
        x = int(x)
        y = int(y)
        if 0 <= y < tx_mask.shape[0] and 0 <= x < tx_mask.shape[1] and bool(tx_mask[y, x]):
            snapped_points.append((x, y))
            continue
        deltas = legal_rc - np.array([[y, x]])
        best_idx = int(np.argmin(np.sum(deltas * deltas, axis=1)))
        row, col = legal_rc[best_idx]
        snapped_points.append((int(col), int(row)))
    return snapped_points


def render_roi_coverage_visualization(
    pathgain_db: np.ndarray,
    tx_loc: tuple[int, int],
    metrics: dict[str, float],
    output_path: str | Path,
    overlay_points: list[tuple[int, int]] | None = None,
) -> None:
    mpl_config_dir = PACKAGE_ROOT / ".mplconfig"
    cache_dir = PACKAGE_ROOT / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    chinese_font = next((name for name in CHINESE_FONT_CANDIDATES if name in available_fonts), CHINESE_FONT_CANDIDATES[0])
    matplotlib.rcParams["font.family"] = [LATIN_FONT_FAMILY, chinese_font]
    matplotlib.rcParams["axes.unicode_minus"] = False

    coverage_map = pathgain_db >= COVERAGE_THRESHOLD_DB
    overlay_points = overlay_points or []

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.imshow(coverage_map, cmap="magma", vmin=0.0, vmax=1.0)
    if overlay_points:
        overlay_cols = [point[0] for point in overlay_points]
        overlay_rows = [point[1] for point in overlay_points]
        ax.scatter(overlay_cols, overlay_rows, c="red", s=72, marker="x", linewidths=1.6)
    else:
        row, col = tx_loc
        ax.scatter([col], [row], c="cyan", s=36, marker="x")
    ax.set_title(
        "RoI Coverage\n"
        f"coverage={metrics['coverage']:.4f}  "
        f"se={metrics['spectral_efficiency']:.4f}  "
        f"score={metrics['score']:.4f}"
    )
    ax.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def evaluate_checkpoint(
    image_path: str | Path,
    checkpoint_path: str | Path,
    model_path: str | Path,
    network_type: str,
    device_name: str,
    output_dir: str | Path,
    explore: bool = False,
    overlay_points_from_image: str | Path | None = None,
    overlay_points: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    image_path = Path(image_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pixel_map = load_map_normalized(image_path)
    observation = build_policy_observation(pixel_map)
    agent = load_checkpoint_agent(checkpoint_path)
    predictor = RMNetPredictor(model_path=model_path, network_type=network_type, device_name=device_name)

    try:
        action = compute_checkpoint_action(agent, observation, explore=explore)
    finally:
        stop = getattr(agent, "stop", None)
        if callable(stop):
            stop()

    tx_loc = calc_upsampling_loc(action, pixel_map)
    overlay_points = overlay_points or []
    if overlay_points_from_image:
        overlay_points = extract_overlay_points_xy_from_deployment_image(
            image_path=overlay_points_from_image,
            target_shape=pixel_map.shape,
        )
    snapped_overlay_points = snap_overlay_points_to_valid_tx_pixels(pixel_map, overlay_points) if overlay_points else []
    tx_locs = overlay_points_xy_to_tx_locs(snapped_overlay_points) if snapped_overlay_points else [tx_loc]
    pathgain_db, metrics = get_stats(pixel_map, tx_locs, pmnet=predictor)

    stem = image_path.stem
    png_path = output_dir / f"{stem}_checkpoint_rmnet_roi_coverage.png"
    json_path = output_dir / f"{stem}_checkpoint_rmnet_roi_coverage_summary.json"
    render_roi_coverage_visualization(
        pathgain_db=pathgain_db,
        tx_loc=tx_loc,
        metrics=metrics,
        output_path=png_path,
        overlay_points=snapped_overlay_points,
    )

    summary = {
        "image": str(image_path),
        "checkpoint": str(checkpoint_path),
        "model_path": str(Path(model_path).expanduser().resolve()),
        "network_type": network_type,
        "device": str(get_device(device_name)),
        "coverage_threshold_db": float(COVERAGE_THRESHOLD_DB),
        "action": int(action),
        "tx_row": int(tx_loc[0]),
        "tx_col": int(tx_loc[1]),
        "overlay_points_xy": [
            {"x": int(point[0]), "y": int(point[1])} for point in (overlay_points or [])
        ],
        "snapped_overlay_points_xy": [
            {"x": int(point[0]), "y": int(point[1])} for point in snapped_overlay_points
        ],
        "tx_locs_used_rc": [
            {"row": int(point[0]), "col": int(point[1])} for point in tx_locs
        ],
        "metrics": {key: float(value) for key, value in metrics.items()},
        "visualization": str(png_path),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    overlay_points = parse_overlay_points_arg(args.overlay_points)
    summary = evaluate_checkpoint(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        model_path=args.model_path,
        network_type=args.network_type,
        device_name=args.device,
        output_dir=args.output_dir,
        explore=args.explore,
        overlay_points_from_image=args.overlay_points_from_image or None,
        overlay_points=overlay_points,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
