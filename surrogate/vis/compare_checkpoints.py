"""
用途:
  对比同一模型结构在 USC 权重和 RadioMap3DSeer 权重下，对同一样本的推理结果。

示例命令:
  python surrogate/vis/compare_checkpoints.py \
    --model-type pmnet \
    --usc-model surrogate/checkpoints/pmnet_usc.pt \
    --radiomap-model surrogate/checkpoints/pmnet_radiomap3dseer.pt \
    --map-path /path/to/map.png \
    --tx-path /path/to/tx.png \
    --label-path /path/to/label.png

参数说明:
  --model-type: 模型类型，pmnet 或 rmnet。
  --usc-model: USC 训练得到的 checkpoint。
  --radiomap-model: RadioMap3DSeer 训练得到的 checkpoint。
  --output-stride: 模型输出步长。
  --output-dir: 输出图片目录。
  --antenna: 用输入目录模式时的天线子目录类型。
  --city-map: 用输入目录模式时的地图子目录类型。
  --input-dir: 数据集 png 根目录。
  --gt-dir: 标签目录。
  --scene-id/--tx-id: 用目录模式定位样本。
  --map-path/--tx-path/--label-path: 直接指定单样本路径。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model_pmnet import build_pmnet
from model_rmnet import build_rmnet
from runtime_surrogate import get_device, load_checkpoint


DB_MIN = -162.0
DB_MAX = -75.0
DEFAULT_OUTPUT_DIR = ROOT_DIR / "vis" / "outputs" / "compare_usc_vs_radiomap"
DEFAULT_PMNET_USC_MODEL = ROOT_DIR / "checkpoints" / "pmnet_usc.pt"
DEFAULT_PMNET_RADIOMAP_MODEL = ROOT_DIR / "checkpoints" / "pmnet_radiomap3dseer.pt"
DEFAULT_RMNET_USC_MODEL = ROOT_DIR / "checkpoints" / "rmnet_usc.pt"
DEFAULT_RMNET_RADIOMAP_MODEL = ROOT_DIR / "checkpoints" / "rmnet_radiomap3dseer.pt"


def build_model(model_type: str, output_stride: int):
    if model_type == "pmnet":
        return build_pmnet(output_stride=output_stride)
    return build_rmnet(output_stride=output_stride)


def resolve_dirs(input_dir: str, antenna: str, city_map: str):
    if city_map == "complete":
        map_dir = os.path.join(input_dir, "buildings_complete")
    elif city_map == "height":
        map_dir = os.path.join(input_dir, "buildingsWHeight")
    else:
        raise ValueError(f"Unsupported city_map: {city_map}")

    if antenna == "complete":
        tx_dir = os.path.join(input_dir, "antennas")
    elif antenna == "height":
        tx_dir = os.path.join(input_dir, "antennasWHeight")
    elif antenna == "building":
        tx_dir = os.path.join(input_dir, "antennasBuildings")
    else:
        raise ValueError(f"Unsupported antenna: {antenna}")
    return map_dir, tx_dir


def build_paths_from_ids(input_dir: str, gt_dir: str | None, scene_id: str, tx_id: str, antenna: str, city_map: str):
    map_dir, tx_dir = resolve_dirs(input_dir, antenna, city_map)
    label_name = f"{scene_id}_{tx_id}.png"
    return (
        os.path.join(map_dir, f"{scene_id}.png"),
        os.path.join(tx_dir, label_name),
        os.path.join(gt_dir, label_name) if gt_dir else None,
        label_name,
    )


def to_db_scale(array: np.ndarray) -> np.ndarray:
    array = np.clip(array, 0.0, 1.0)
    return array * (DB_MAX - DB_MIN) + DB_MIN


def load_sample(map_path: str, tx_path: str, label_path: str | None, device: torch.device):
    image_map = np.asarray(Image.open(map_path))
    image_tx = np.asarray(Image.open(tx_path))
    inputs = np.stack([image_map, image_tx], axis=2)
    inputs = transforms.ToTensor()(inputs).float().unsqueeze(0).to(device)

    label_tensor = None
    if label_path:
        image_label = np.asarray(Image.open(label_path))
        label_tensor = transforms.ToTensor()(image_label).float()
    return inputs, label_tensor


def predict(model: torch.nn.Module, inputs: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        pred = torch.clamp(model(inputs), 0, 1)
    return pred[0, 0].detach().cpu().numpy()


def save_panel(pred_a: np.ndarray, pred_b: np.ndarray, label: np.ndarray | None, output_path: Path) -> None:
    ncols = 3 if label is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    if ncols == 2:
        axes = list(axes)

    axes[0].imshow(to_db_scale(pred_a), cmap="viridis", vmin=DB_MIN, vmax=DB_MAX)
    axes[0].set_title("USC")
    axes[0].axis("off")

    axes[1].imshow(to_db_scale(pred_b), cmap="viridis", vmin=DB_MIN, vmax=DB_MAX)
    axes[1].set_title("RadioMap3DSeer")
    axes[1].axis("off")

    if label is not None:
        axes[2].imshow(label, cmap="gray", vmin=0.0, vmax=1.0)
        axes[2].set_title("Label")
        axes[2].axis("off")

    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="pmnet", choices=["pmnet", "rmnet"])
    parser.add_argument("--usc-model", default=None)
    parser.add_argument("--radiomap-model", default=None)
    parser.add_argument("--output-stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("-o", "--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--antenna", default="height")
    parser.add_argument("--city-map", default="height")
    parser.add_argument("-d", "--input-dir", type=str)
    parser.add_argument("-g", "--gt-dir", type=str)
    parser.add_argument("--scene-id", type=str)
    parser.add_argument("--tx-id", type=str)
    parser.add_argument("--map-path", type=str)
    parser.add_argument("--tx-path", type=str)
    parser.add_argument("--label-path", type=str)
    args = parser.parse_args()

    if args.usc_model is None:
        args.usc_model = str(DEFAULT_PMNET_USC_MODEL if args.model_type == "pmnet" else DEFAULT_RMNET_USC_MODEL)
    if args.radiomap_model is None:
        args.radiomap_model = str(
            DEFAULT_PMNET_RADIOMAP_MODEL if args.model_type == "pmnet" else DEFAULT_RMNET_RADIOMAP_MODEL
        )

    if args.map_path and args.tx_path:
        map_path = args.map_path
        tx_path = args.tx_path
        label_path = args.label_path
        label_name = os.path.basename(label_path) if label_path else "sample"
    elif args.input_dir and args.scene_id is not None and args.tx_id is not None:
        map_path, tx_path, label_path, label_name = build_paths_from_ids(
            args.input_dir,
            args.gt_dir,
            str(args.scene_id),
            str(args.tx_id),
            args.antenna,
            args.city_map,
        )
    else:
        raise ValueError("Provide either --map-path/--tx-path or --input-dir/--scene-id/--tx-id")

    for path in [map_path, tx_path, args.usc_model, args.radiomap_model]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    if label_path and not os.path.exists(label_path):
        raise FileNotFoundError(label_path)

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs, label_tensor = load_sample(map_path, tx_path, label_path, device)
    usc_model = build_model(args.model_type, args.output_stride)
    radiomap_model = build_model(args.model_type, args.output_stride)
    load_checkpoint(usc_model, args.usc_model, strict=True)
    load_checkpoint(radiomap_model, args.radiomap_model, strict=True)
    usc_model = usc_model.to(device).eval()
    radiomap_model = radiomap_model.to(device).eval()

    usc_pred = predict(usc_model, inputs)
    radiomap_pred = predict(radiomap_model, inputs)
    label_array = label_tensor[0].detach().cpu().numpy() if label_tensor is not None else None

    stem = Path(label_name).stem
    save_panel(usc_pred, radiomap_pred, label_array, output_dir / f"{stem}_usc_vs_radiomap.png")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
