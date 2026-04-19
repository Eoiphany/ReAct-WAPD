"""
用途:
  统一评估 USC 或 RadioMap3DSeer 上的 PMNet / RMNet 权重，并可选保存少量预测预览图。

示例命令:
  评估 USC:
    python surrogate/vis/evaluate_surrogate.py \
      --dataset usc \
      --model-type pmnet \
      --data-root /path/to/usc \
      --checkpoint surrogate/checkpoints/pmnet_usc.pt
  评估 RadioMap3DSeer:
    python surrogate/vis/evaluate_surrogate.py \
      --dataset radiomap3dseer \
      --model-type rmnet \
      --data-root /path/to/RadioMap3DSeer \
      --csv-file /path/to/eval_pairs.csv \
      --checkpoint surrogate/checkpoints/rmnet_radiomap3dseer.pt \
      --output-dir surrogate/eval_outputs \
      --preview-count 8

参数说明:
  --dataset: 数据集类型，usc 或 radiomap3dseer。
  --model-type: 模型类型，pmnet 或 rmnet。
  --data-root: 数据集根目录。
  --csv-file: 可选样本 CSV；为空时自动扫描默认数据目录。
  --checkpoint: 待评估权重路径。
  --output-dir: 可选输出目录；若提供并设置了 --preview-count，则保存部分预测图与标签图。
  --preview-count: 最多保存多少个预览样本。
  --batch-size: batch 大小。
  --num-workers: DataLoader worker 数量。
  --output-stride: 模型输出步长，8 或 16。
  --use-height / --no-use-height: RadioMap3DSeer 是否使用带高度输入。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_surrogate import (
    RadioMap3DSeerDataset,
    USCDataset,
    resolve_radiomap_sample_pairs,
    resolve_usc_sample_ids,
)
from model_pmnet import build_pmnet
from model_rmnet import build_rmnet
from utils import MAE, R2, RMSE, get_device, load_checkpoint


class USCPreviewDataset(Dataset):
    def __init__(self, data_root: str, csv_file: str | None):
        self.sample_ids = resolve_usc_sample_ids(data_root, csv_file)
        self.dataset = USCDataset(data_root, self.sample_ids)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs, target = self.dataset[idx]
        return inputs, target, self.sample_ids[idx]


class RadioMapPreviewDataset(Dataset):
    def __init__(self, data_root: str, csv_file: str | None, use_height: bool):
        self.sample_pairs = resolve_radiomap_sample_pairs(data_root, csv_file)
        self.dataset = RadioMap3DSeerDataset(data_root, self.sample_pairs, use_height=use_height)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs, target, scene_id, tx_id = self.dataset[idx]
        return inputs, target, f"{scene_id}_{tx_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PMNet/RMNet on USC or RadioMap3DSeer.")
    parser.add_argument("--dataset", required=True, choices=["usc", "radiomap3dseer"])
    parser.add_argument("--model-type", required=True, choices=["pmnet", "rmnet"])
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--csv-file", type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--preview-count", default=0, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--output-stride", default=16, choices=[8, 16], type=int)
    parser.add_argument("--use-height", dest="use_height", action="store_true")
    parser.add_argument("--no-use-height", dest="use_height", action="store_false")
    parser.set_defaults(use_height=True)
    return parser.parse_args()


def build_model(model_type: str, output_stride: int):
    if model_type == "pmnet":
        return build_pmnet(output_stride=output_stride)
    return build_rmnet(output_stride=output_stride)


def save_preview(pred: torch.Tensor, target: torch.Tensor, stem: str, output_dir: Path, model_type: str) -> None:
    pred_array = pred.detach().cpu().squeeze().numpy()
    target_array = target.detach().cpu().squeeze().numpy()

    fig = plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(pred_array, cmap="viridis")
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(target_array, cmap="viridis")
    plt.title("Label")
    plt.axis("off")

    plt.tight_layout()
    fig.savefig(output_dir / f"{stem}_{model_type}_preview.png", dpi=300)
    plt.close(fig)


def build_loader(args: argparse.Namespace) -> DataLoader:
    if args.dataset == "usc":
        dataset = USCPreviewDataset(args.data_root, args.csv_file)
    else:
        dataset = RadioMapPreviewDataset(args.data_root, args.csv_file, args.use_height)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )


def main() -> None:
    args = parse_args()
    device = get_device()
    # print("device: ", device)
    model = build_model(args.model_type, args.output_stride)
    load_checkpoint(model, args.checkpoint, strict=True)
    model = model.to(device)
    loader = build_loader(args)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    total_rmse = 0.0
    total_mae = 0.0
    total_r2 = 0.0
    total_samples = 0
    saved_count = 0

    with torch.no_grad():
        for inputs, targets, names in tqdm(loader, desc="Evaluate", leave=True):
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = torch.clamp(model(inputs), 0, 1)
            total_rmse += RMSE(preds, targets).item() * inputs.size(0)
            total_mae += MAE(preds, targets).item() * inputs.size(0)
            total_r2 += R2(preds, targets).item() * inputs.size(0)
            total_samples += inputs.size(0)

            if output_dir is not None and saved_count < args.preview_count:
                for local_idx in range(inputs.size(0)):
                    if saved_count >= args.preview_count:
                        break
                    save_preview(preds[local_idx], targets[local_idx], names[local_idx], output_dir, args.model_type)
                    saved_count += 1

    denom = max(total_samples, 1)
    summary = {
        "dataset": args.dataset,
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "rmse": total_rmse / denom,
        "mae": total_mae / denom,
        "r2": total_r2 / denom,
    }
    if output_dir is not None:
        with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
