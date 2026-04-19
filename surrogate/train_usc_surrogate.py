"""
用途:
  在 USC 数据集上训练代理模型，并保存验证集上表现最好的权重。

示例命令:
  训练 PMNet:
    python surrogate/train_usc_surrogate.py \
      --model-type pmnet \
      --data-root /path/to/usc \
      --output-root surrogate/runs
  训练 RMNet:
    python surrogate/train_usc_surrogate.py \
      --model-type rmnet \
      --data-root /path/to/usc \
      --output-root surrogate/runs
  训练 U-Net:
    python surrogate/train_usc_surrogate.py \
      --model-type unet \
      --data-root /path/to/usc \
      --output-root surrogate/runs
  训练 TransUNet:
    python surrogate/train_usc_surrogate.py \
      --model-type transunet \
      --data-root /path/to/usc \
      --output-root surrogate/runs
  训练 RadioUNet:
    python surrogate/train_usc_surrogate.py \
      --model-type radiounet \
      --data-root /path/to/usc \
      --output-root surrogate/runs
  只评估已有权重:
    python surrogate/train_usc_surrogate.py \
      --model-type pmnet \
      --data-root /path/to/usc \
      --csv-file /path/to/eval_ids.csv \
      --checkpoint surrogate/checkpoints/pmnet_usc.pt \
      --eval-only

参数说明:
  --model-type: 模型类型，pmnet / rmnet / unet / transunet / radiounet / radionet。
  --data-root: USC 数据集根目录，内部应包含 map/、Tx/、pmap/。
  --csv-file: 可选样本列表 CSV；训练时为空则自动读 Data_coarse_train.csv 或扫描 pmap/。
  --checkpoint: 评估模式或继续训练时使用的已有权重路径。
  --output-root: 训练输出目录，保存日志、配置、历史与 best checkpoint。
  --output-stride: 模型输出步长，8 或 16。
  --batch-size: batch 大小。
  --num-epochs: 训练轮数。
  --val-freq: 每隔多少个 epoch 做一次验证。
  --num-workers: DataLoader worker 数量。
  --train-ratio: 训练集比例；剩余部分作为验证集。
  --lr: Adam 学习率。
  --lr-decay: StepLR 的 gamma。
  --step: StepLR 的 step size。
  --weight-decay: Adam 的 weight decay。
  --log-every: 每多少个 step 打印一次训练 loss。
  --seed: 随机种子。
  --eval-only: 只做评估，不训练；此时建议显式传入 --csv-file 与 --checkpoint。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

try:
    from .data_surrogate import USCDataset, resolve_usc_sample_ids
    from .model_registry import ALL_MODEL_TYPES, build_model, select_prediction
    from .utils import MSE, RMSE, get_device, load_checkpoint, save_checkpoint, save_training_plots, set_seed
except ImportError:
    from data_surrogate import USCDataset, resolve_usc_sample_ids
    from model_registry import ALL_MODEL_TYPES, build_model, select_prediction
    from utils import MSE, RMSE, get_device, load_checkpoint, save_checkpoint, save_training_plots, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate surrogate models on the USC dataset.")
    parser.add_argument("--model-type", required=True, choices=ALL_MODEL_TYPES)
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--csv-file", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--output-root", default="./surrogate_runs", type=str)
    parser.add_argument("--output-stride", default=16, choices=[8, 16], type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-epochs", default=30, type=int)
    parser.add_argument("--val-freq", default=1, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--train-ratio", default=0.9, type=float)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--lr-decay", default=0.5, type=float)
    parser.add_argument("--step", default=10, type=int)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--log-every", default=100, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


@dataclass
class USCTrainConfig:
    model_type: str
    data_root: str
    csv_file: str | None
    checkpoint: str | None
    output_root: str
    output_stride: int
    batch_size: int
    num_epochs: int
    val_freq: int
    num_workers: int
    train_ratio: float
    lr: float
    lr_decay: float
    step: int
    weight_decay: float
    log_every: int
    seed: int
    eval_only: bool

    @property
    def param_str(self) -> str:
        return f"{self.batch_size}_{self.lr}_{self.lr_decay}_{self.step}"

    @property
    def run_name(self) -> str:
        return f"{self.model_type}_usc"


def build_config(args: argparse.Namespace) -> USCTrainConfig:
    return USCTrainConfig(**vars(args))


def prepare_run_dir(cfg: USCTrainConfig) -> Path:
    run_dir = Path(cfg.output_root) / cfg.run_name / cfg.param_str
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_history_csv(history: list[dict], output_path: Path) -> None:
    fieldnames = ["epoch", "train_loss", "val_rmse", "best_val_rmse", "lr"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def evaluate(model, loader, device) -> float:
    model.eval()
    total_rmse = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Eval", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = torch.clamp(select_prediction(model(inputs)), 0, 1)
            batch_rmse = RMSE(preds, targets)
            total_rmse += batch_rmse.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_rmse / max(total_samples, 1)


def build_eval_loader(cfg: USCTrainConfig) -> DataLoader:
    sample_ids = resolve_usc_sample_ids(cfg.data_root, cfg.csv_file)
    dataset = USCDataset(cfg.data_root, sample_ids)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
    )


def build_train_and_val_loaders(cfg: USCTrainConfig):
    sample_ids = resolve_usc_sample_ids(cfg.data_root, cfg.csv_file)
    dataset = USCDataset(cfg.data_root, sample_ids)
    dataset_size = len(dataset)
    train_size = int(dataset_size * cfg.train_ratio)
    val_size = dataset_size - train_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError(f"Invalid split sizes: train={train_size}, val={val_size}, total={dataset_size}")

    split_generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_generator)

    loader_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def main() -> None:
    cfg = build_config(parse_args())
    set_seed(cfg.seed)
    device = get_device()
    run_dir = prepare_run_dir(cfg)

    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2)

    model = build_model(cfg.model_type, output_stride=cfg.output_stride, in_channels=2)
    if cfg.checkpoint:
        load_checkpoint(model, cfg.checkpoint, strict=True)
    model = model.to(device)

    if cfg.eval_only:
        if not cfg.checkpoint:
            raise ValueError("--eval-only requires --checkpoint")
        eval_loader = build_eval_loader(cfg)
        eval_rmse = evaluate(model, eval_loader, device)
        summary = {
            "dataset": "usc",
            "model_type": cfg.model_type,
            "checkpoint": cfg.checkpoint,
            "eval_rmse": eval_rmse,
        }
        with (run_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2))
        return

    train_loader, val_loader = build_train_and_val_loaders(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step, gamma=cfg.lr_decay)

    best_val_rmse = float("inf")
    best_checkpoint_path = run_dir / f"{cfg.model_type}_usc_best.pt"
    history: list[dict] = []
    global_step = 0

    with (run_dir / "train.log").open("w", encoding="utf-8") as log_handle:
        log_handle.write(json.dumps(asdict(cfg), indent=2) + "\n")
        for epoch in range(cfg.num_epochs):
            model.train()
            epoch_loss_total = 0.0
            epoch_samples = 0

            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}", leave=True):
                global_step += 1
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad(set_to_none=True)
                preds = select_prediction(model(inputs))
                loss = MSE(preds, targets)
                loss.backward()
                optimizer.step()

                epoch_loss_total += loss.item() * inputs.size(0)
                epoch_samples += inputs.size(0)

                if global_step % cfg.log_every == 0:
                    message = f"epoch={epoch + 1}, step={global_step}, train_loss={loss.item():.6f}"
                    print(message)
                    log_handle.write(message + "\n")

            train_loss = epoch_loss_total / max(epoch_samples, 1)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            val_rmse = None
            if epoch % cfg.val_freq == 0:
                val_rmse = evaluate(model, val_loader, device)
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    save_checkpoint(model, best_checkpoint_path)

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_rmse": val_rmse,
                    "best_val_rmse": None if best_val_rmse == float("inf") else best_val_rmse,
                    "lr": current_lr,
                }
            )
            save_history_csv(history, run_dir / "history.csv")
            with (run_dir / "history.json").open("w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)
            save_training_plots(
                history=history,
                output_path=run_dir / "training_curves.png",
                title=f"USC / {cfg.model_type}",
                metric_keys=("train_loss", "val_rmse", "best_val_rmse", "lr"),
            )

    summary = {
        "dataset": "usc",
        "model_type": cfg.model_type,
        "best_val_rmse": best_val_rmse,
        "best_checkpoint": str(best_checkpoint_path),
    }
    with (run_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
