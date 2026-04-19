"""
用途:
  在同一样本上对比 PMNet 与 RMNet 的预测图、误差图和指标。

示例命令:
  python surrogate/vis/compare_model.py \
    --pmnet-model surrogate/checkpoints/pmnet_radiomap3dseer.pt \
    --rmnet-model surrogate/checkpoints/rmnet_radiomap3dseer.pt \
    --map-path /path/to/map.png \
    --tx-path /path/to/tx.png \
    --label-path /path/to/label.png

参数说明:
  --pmnet-model: PMNet checkpoint 路径。
  --rmnet-model: RMNet checkpoint 路径。
  --pmnet-output-stride/--rmnet-output-stride: 两个模型各自的输出步长。
  --output-dir: 可视化输出目录。
  --antenna/--city-map/--input-dir/--gt-dir/--scene-id/--tx-id: 目录模式定位样本所需参数。
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
DEFAULT_OUTPUT_DIR = ROOT_DIR / "vis" / "outputs" / "compare_pmnet_vs_rmnet"
DEFAULT_PMNET_MODEL = ROOT_DIR / "checkpoints" / "pmnet_radiomap3dseer.pt"
DEFAULT_RMNET_MODEL = ROOT_DIR / "checkpoints" / "rmnet_radiomap3dseer.pt"


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


def load_sample(map_path: str, tx_path: str, label_path: str | None, device: torch.device):
    image_map = np.array(Image.open(map_path), copy=True)
    image_tx = np.array(Image.open(tx_path), copy=True)
    inputs = np.stack([image_map, image_tx], axis=2)
    inputs = transforms.ToTensor()(inputs).float().unsqueeze(0).to(device)

    label_tensor = None
    if label_path:
        image_label = np.array(Image.open(label_path), copy=True)
        label_tensor = transforms.ToTensor()(image_label).float()
    return inputs, label_tensor


def predict(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.clamp(model(inputs), 0, 1)


def to_db_scale(array: np.ndarray) -> np.ndarray:
    array = np.clip(array, 0.0, 1.0)
    return array * (DB_MAX - DB_MIN) + DB_MIN


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(((pred - target) ** 2).mean().sqrt().item())


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).abs().mean().item())


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    target_mean = target.mean()
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target_mean) ** 2)
    if float(ss_tot.item()) == 0.0:
        return 0.0
    return float((1.0 - ss_res / ss_tot).item())


def save_single_image(array: np.ndarray, output_path: Path, title: str, cmap: str, vmin=None, vmax=None) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def save_compare_grid(
    pmnet_pred: np.ndarray,
    rmnet_pred: np.ndarray,
    label: np.ndarray,
    output_path: Path,
    pmnet_metrics: tuple[float, float, float],
    rmnet_metrics: tuple[float, float, float],
) -> None:
    pmnet_error = np.abs(pmnet_pred - label)
    rmnet_error = np.abs(rmnet_pred - label)
    err_vmax = max(float(pmnet_error.max()), float(rmnet_error.max()), 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.3), constrained_layout=True)
    gt_im = axes[0, 0].imshow(to_db_scale(label), cmap="viridis", vmin=DB_MIN, vmax=DB_MAX)
    axes[0, 0].set_title("Ground Truth")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(to_db_scale(label), cmap="viridis", vmin=DB_MIN, vmax=DB_MAX)
    axes[1, 0].set_title("Ground Truth")
    axes[1, 0].axis("off")

    axes[0, 1].imshow(to_db_scale(pmnet_pred), cmap="viridis", vmin=DB_MIN, vmax=DB_MAX)
    axes[0, 1].set_title(f"PMNet\nRMSE={pmnet_metrics[0]:.4f}, MAE={pmnet_metrics[1]:.4f}, R^2={pmnet_metrics[2]:.4f}")
    axes[0, 1].axis("off")

    axes[1, 1].imshow(to_db_scale(rmnet_pred), cmap="viridis", vmin=DB_MIN, vmax=DB_MAX)
    axes[1, 1].set_title(f"RMNet\nRMSE={rmnet_metrics[0]:.4f}, MAE={rmnet_metrics[1]:.4f}, R^2={rmnet_metrics[2]:.4f}")
    axes[1, 1].axis("off")

    err_im = axes[0, 2].imshow(pmnet_error, cmap="magma", vmin=0.0, vmax=err_vmax)
    axes[0, 2].set_title("Absolute Error of PMNet")
    axes[0, 2].axis("off")
    axes[1, 2].imshow(rmnet_error, cmap="magma", vmin=0.0, vmax=err_vmax)
    axes[1, 2].set_title("Absolute Error of RMNet")
    axes[1, 2].axis("off")

    fig.colorbar(gt_im, ax=[axes[0, 0], axes[1, 0], axes[0, 1], axes[1, 1]], fraction=0.024, pad=0.02, label="Path Gain (dB)")
    fig.colorbar(err_im, ax=[axes[0, 2], axes[1, 2]], fraction=0.030, pad=0.02, label="Absolute Error")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PMNet and RMNet on one sample.")
    parser.add_argument("--pmnet-model", default=str(DEFAULT_PMNET_MODEL))
    parser.add_argument("--rmnet-model", default=str(DEFAULT_RMNET_MODEL))
    parser.add_argument("--pmnet-output-stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--rmnet-output-stride", type=int, default=16, choices=[8, 16])
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

    if args.map_path and args.tx_path:
        map_path = args.map_path
        tx_path = args.tx_path
        label_path = args.label_path
        sample_name = Path(label_path).stem if label_path else Path(tx_path).stem
    elif args.input_dir and args.scene_id is not None and args.tx_id is not None:
        map_path, tx_path, label_path, label_name = build_paths_from_ids(
            args.input_dir,
            args.gt_dir,
            str(args.scene_id),
            str(args.tx_id),
            args.antenna,
            args.city_map,
        )
        sample_name = Path(label_name).stem
    else:
        raise ValueError("Provide either --map-path/--tx-path or --input-dir/--scene-id/--tx-id")

    required_paths = [map_path, tx_path, args.pmnet_model, args.rmnet_model]
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    if label_path and not os.path.exists(label_path):
        raise FileNotFoundError(label_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    inputs, label_tensor = load_sample(map_path, tx_path, label_path, device)
    pmnet = build_pmnet(args.pmnet_output_stride)
    rmnet = build_rmnet(args.rmnet_output_stride)
    load_checkpoint(pmnet, args.pmnet_model, strict=True)
    load_checkpoint(rmnet, args.rmnet_model, strict=True)
    pmnet = pmnet.to(device).eval()
    rmnet = rmnet.to(device).eval()

    pmnet_pred = predict(pmnet, inputs)[0, 0].detach().cpu()
    rmnet_pred = predict(rmnet, inputs)[0, 0].detach().cpu()
    label = label_tensor[0].detach().cpu() if label_tensor is not None else None

    save_single_image(to_db_scale(pmnet_pred.numpy()), output_dir / f"{sample_name}_pmnet_pred.png", "PMNet Prediction", "viridis", DB_MIN, DB_MAX)
    save_single_image(to_db_scale(rmnet_pred.numpy()), output_dir / f"{sample_name}_rmnet_pred.png", "RMNet Prediction", "viridis", DB_MIN, DB_MAX)

    if label is not None:
        save_single_image(to_db_scale(label.numpy()), output_dir / f"{sample_name}_label.png", "Label", "viridis", DB_MIN, DB_MAX)
        pmnet_metrics = (rmse(pmnet_pred, label), mae(pmnet_pred, label), r2_score(pmnet_pred, label))
        rmnet_metrics = (rmse(rmnet_pred, label), mae(rmnet_pred, label), r2_score(rmnet_pred, label))
        save_compare_grid(
            pmnet_pred.numpy(),
            rmnet_pred.numpy(),
            label.numpy(),
            output_dir / f"{sample_name}_pmnet_vs_rmnet.png",
            pmnet_metrics,
            rmnet_metrics,
        )

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
