"""注释
命令示例:
python -m surrogate.vis.visualize_single_surrogate_sample \
  --dataset usc \
  --model-type pmnet \
  --data-root /Users/epiphanyer/Desktop/coding/paper_experiment/usc-data \
  --checkpoint /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/runs/pmnet_usc/16_0.0005_0.5_10/pmnet_usc_best.pt \
  --sample-id 42 \
  --output-path /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/output_pred/usc_42_compare.png

python -m surrogate.vis.visualize_single_surrogate_sample \
  --dataset radiomap3dseer \
  --model-type pmnet \
  --data-root /Users/epiphanyer/Desktop/coding/paper_experiment/dataset \
  --checkpoint /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/runs/pmnet_radiomap3dseer/16_0.0001_0.5_10/pmnet_radiomap3dseer_best.pt \
  --scene-id 348 \
  --tx-id 7 \
  --output-path /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/output_pred/348_7_compare.png


参数说明:
- --dataset: 数据集类型，`usc` 或 `radiomap3dseer`。
- --model-type: 模型类型，沿用 surrogate 统一注册入口。
- --data-root: 数据集根目录。
- --checkpoint: 训练完成后的模型权重路径。
- --output-path: 输出对比图 PNG 路径；同目录会额外生成一个同名 `.json` 指标摘要。
- --sample-id: USC 样本 id；与 `--sample-index` 二选一。
- --scene-id / --tx-id: RadioMap3DSeer 样本键；与 `--sample-index` 二选一。
- --sample-index: 不知道样本名时可直接按解析后的样本列表下标取样。
- --csv-file: 可选样本 CSV；样本解析规则与训练脚本一致。
- --output-stride: 需要时传给对应模型。
- --use-height / --no-use-height: RadioMap3DSeer 是否使用带高度输入。
- --device: 推理设备，`auto/cpu/cuda/mps`。

脚本逻辑:
- 按数据集类型定位单个样本，读取输入张量与标签张量。
- 加载指定 checkpoint，对该样本做一次前向推理并裁剪到 `[0,1]`。
- 输出一张仅包含 `Inference` 与 `Label` 的对比图，并写出该样本的 `rmse/mae/max_abs_error` 摘要。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

try:
    from ..data_surrogate import (
        RadioMap3DSeerDataset,
        USCDataset,
        resolve_radiomap_sample_pairs,
        resolve_usc_sample_ids,
    )
    from ..model_registry import ALL_MODEL_TYPES, build_model, select_prediction
    from ..utils import get_device, load_checkpoint, configure_plot_style
except ImportError:
    from data_surrogate import RadioMap3DSeerDataset, USCDataset, resolve_radiomap_sample_pairs, resolve_usc_sample_ids
    from model_registry import ALL_MODEL_TYPES, build_model, select_prediction
    from utils import get_device, load_checkpoint, configure_plot_style


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize surrogate inference vs label for one sample.")
    parser.add_argument("--dataset", required=True, choices=["usc", "radiomap3dseer"])
    parser.add_argument("--model-type", required=True, choices=ALL_MODEL_TYPES)
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output-path", required=True, type=str)
    parser.add_argument("--sample-id", type=str)
    parser.add_argument("--scene-id", type=str)
    parser.add_argument("--tx-id", type=str)
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--csv-file", type=str)
    parser.add_argument("--output-stride", default=16, choices=[8, 16], type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], type=str)
    parser.add_argument("--use-height", dest="use_height", action="store_true")
    parser.add_argument("--no-use-height", dest="use_height", action="store_false")
    parser.set_defaults(use_height=True)
    return parser


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return get_device()
    return torch.device(device_name)


def _prepare_matplotlib(output_path: Path) -> None:
    cache_dir = output_path.parent / ".mpl-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def select_usc_sample(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str]:
    sample_ids = resolve_usc_sample_ids(args.data_root, args.csv_file)
    if args.sample_id is not None:
        sample_id = args.sample_id
        if sample_id not in sample_ids:
            raise ValueError(f"USC sample-id not found: {sample_id}")
    elif args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= len(sample_ids):
            raise IndexError(f"sample-index out of range: {args.sample_index}")
        sample_id = sample_ids[args.sample_index]
    else:
        raise ValueError("USC requires --sample-id or --sample-index")

    dataset = USCDataset(args.data_root, [sample_id])
    inputs, label = dataset[0]
    return inputs.numpy(), label.squeeze(0).numpy(), sample_id


def select_radiomap_sample(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str]:
    sample_pairs = resolve_radiomap_sample_pairs(args.data_root, args.csv_file)
    if args.scene_id is not None or args.tx_id is not None:
        if args.scene_id is None or args.tx_id is None:
            raise ValueError("RadioMap3DSeer requires both --scene-id and --tx-id")
        sample_pair = (args.scene_id, args.tx_id)
        if sample_pair not in sample_pairs:
            raise ValueError(f"RadioMap3DSeer sample not found: {sample_pair}")
    elif args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= len(sample_pairs):
            raise IndexError(f"sample-index out of range: {args.sample_index}")
        sample_pair = sample_pairs[args.sample_index]
    else:
        raise ValueError("RadioMap3DSeer requires --scene-id/--tx-id or --sample-index")

    dataset = RadioMap3DSeerDataset(args.data_root, [sample_pair], use_height=args.use_height)
    inputs, label, scene_id, tx_id = dataset[0]
    return inputs.numpy(), label.squeeze(0).numpy(), f"{scene_id}_{tx_id}"


def load_single_sample(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str]:
    if args.dataset == "usc":
        return select_usc_sample(args)
    return select_radiomap_sample(args)


def run_inference(
    *,
    inputs: np.ndarray,
    model_type: str,
    checkpoint: str,
    output_stride: int,
    device: torch.device,
) -> np.ndarray:
    model = build_model(model_type=model_type, output_stride=output_stride, in_channels=int(inputs.shape[0]))
    load_checkpoint(model, checkpoint, strict=True)
    model = model.to(device)
    model.eval()

    tensor = torch.from_numpy(inputs).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        prediction = torch.clamp(select_prediction(model(tensor)), 0.0, 1.0)
    return prediction.detach().cpu().squeeze().numpy().astype(np.float32)


def compute_sample_metrics(prediction: np.ndarray, label: np.ndarray) -> dict[str, float]:
    diff = prediction - label
    return {
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
    }


def save_sample_comparison_figure(
    *,
    prediction: np.ndarray,
    label: np.ndarray,
    output_path: Path,
) -> None:
    _prepare_matplotlib(output_path)
    import matplotlib

    matplotlib.use("Agg")
    configure_plot_style()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vmin = min(float(prediction.min()), float(label.min()))
    vmax = max(float(prediction.max()), float(label.max()))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axes[0].imshow(prediction, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Inference")
    axes[0].axis("off")
    axes[1].imshow(label, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Label")
    axes[1].axis("off")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_summary_json(*, summary: dict, output_path: Path) -> None:
    summary_path = output_path.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output_path).expanduser().resolve()
    device = _resolve_device(args.device)

    inputs, label, sample_name = load_single_sample(args)
    prediction = run_inference(
        inputs=inputs,
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        output_stride=args.output_stride,
        device=device,
    )
    metrics = compute_sample_metrics(prediction, label)
    save_sample_comparison_figure(
        prediction=prediction,
        label=label,
        output_path=output_path,
    )

    summary = {
        "dataset": args.dataset,
        "model_type": args.model_type,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "sample_name": sample_name,
        "output_path": str(output_path),
        "device": str(device),
        **metrics,
    }
    save_summary_json(summary=summary, output_path=output_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
