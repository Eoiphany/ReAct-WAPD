"""
用途:
  在 RadioMap3DSeer 数据集上训练或评估代理模型，完成训练、验证、测试，并将配置、划分结果、
  训练历史、最优权重与最终指标保存到输出目录中。

示例命令:
  训练 PMNet:
    python autodl-tmp/surrogate/train_radiomap3dseer_surrogate.py \
      --model-type pmnet \
      --data-root autodl-tmp/dataset/ \
      --output-root autodl-tmp/surrogate/runs

  训练 RMNet:
    python autodl-tmp/surrogate/train_radiomap3dseer_surrogate.py \
      --model-type rmnet \
      --data-root autodl-tmp/dataset/ \
      --output-root autodl-tmp/surrogate/runs; shutdown -h now
  训练 U-Net:
    python autodl-tmp/surrogate/train_radiomap3dseer_surrogate.py \
      --model-type unet \
      --data-root autodl-tmp/dataset/ \
      --output-root autodl-tmp/surrogate/runs; shutdown -h now
  训练 TransUNet:
    python autodl-tmp/surrogate/train_radiomap3dseer_surrogate.py \
      --model-type transunet \
      --data-root autodl-tmp/dataset/ \
      --output-root autodl-tmp/surrogate/runs; shutdown -h now
  训练 RadioUNet:
    python autodl-tmp/surrogate/train_radiomap3dseer_surrogate.py \
      --model-type radiounet \
      --data-root autodl-tmp/dataset/ \
      --output-root autodl-tmp/surrogate/runs; shutdown -h now
      
  只评估已有权重:
    python autodl-tmp/surrogate/train_radiomap3dseer_surrogate.py \
      --model-type rmnet \
      --data-root autodl-tmp/dataset/ \
      --checkpoint autodl-tmp/surrogate/checkpoints/rmnet_radiomap3dseer.pt \
      --csv-file /path/to/eval_pairs.csv \
      --eval-only

参数说明:
  --model-type: 模型类型，pmnet / rmnet / unet / transunet / radiounet / radionet。
  --data-root: RadioMap3DSeer 数据集根目录，内部应包含 png/ 与 gain/。
  --csv-file: 可选样本 CSV；为空则自动扫描 gain/。
  --checkpoint: 评估模式或继续训练时使用的已有权重路径。
  --output-root: 输出目录，保存日志、配置、history、best checkpoint 与 summary。
  --output-stride: 模型输出步长，8 或 16。
  --batch-size: batch 大小。
  --epochs: 训练轮数。
  --train-ratio: 训练集比例。
  --test-ratio: 测试集比例；剩余部分作为验证集。
  --lr: Adam 学习率。
  --lr-decay: StepLR 的 gamma。
  --step: StepLR 的 step size。
  --weight-decay: Adam 的 weight decay。
  --val-freq: 每隔多少个 epoch 验证一次。
  --num-workers: DataLoader worker 数量。
  --log-every: 每多少个 step 打印一次训练 loss。
  --seed: 随机种子。
  --eval-only: 只对给定样本集做评估，不训练。
  --use-height / --no-use-height: 是否使用带高度的 buildings / antenna 图。
"""

# 启用延迟解析类型注解，使类型提示在运行时以字符串形式处理，便于书写如 str | None、list[dict] 这类较现代的类型表达
from __future__ import annotations

import argparse
import csv
import json
# 定义配置数据类并转为字典
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

try:
    from .data_surrogate import RadioMap3DSeerDataset, resolve_radiomap_sample_pairs
    from .model_registry import ALL_MODEL_TYPES, build_model, select_prediction
    from .utils import (
        MSE,
        build_prefixed_metric_summary,
        compute_regression_metrics,
        get_device,
        load_checkpoint,
        save_checkpoint,
        save_training_plots,
        set_seed,
    )
except ImportError:
    from data_surrogate import RadioMap3DSeerDataset, resolve_radiomap_sample_pairs
    from model_registry import ALL_MODEL_TYPES, build_model, select_prediction
    from utils import (
        MSE,
        build_prefixed_metric_summary,
        compute_regression_metrics,
        get_device,
        load_checkpoint,
        save_checkpoint,
        save_training_plots,
        set_seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate surrogate models on RadioMap3DSeer.")
    parser.add_argument("--model-type", required=True, choices=ALL_MODEL_TYPES)
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--csv-file", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--output-root", default="./surrogate_runs", type=str)
    parser.add_argument("--output-stride", default=16, choices=[8, 16], type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--train-ratio", default=0.7, type=float)
    parser.add_argument("--test-ratio", default=0.2, type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr-decay", default=0.5, type=float)
    # 一个 batch 记作一个 step
    parser.add_argument("--step", default=10, type=int)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--val-freq", default=1, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--log-every", default=100, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--use-height", dest="use_height", action="store_true")
    parser.add_argument("--no-use-height", dest="use_height", action="store_false")
    parser.set_defaults(use_height=True)
    return parser.parse_args()


@dataclass
class RadioMapTrainConfig:
    model_type: str
    data_root: str
    csv_file: str | None
    checkpoint: str | None
    output_root: str
    output_stride: int
    batch_size: int
    epochs: int
    train_ratio: float
    test_ratio: float
    lr: float
    lr_decay: float
    step: int
    weight_decay: float
    val_freq: int
    num_workers: int
    log_every: int
    seed: int
    eval_only: bool
    use_height: bool

    @property
    def val_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.test_ratio

    @property
    def param_str(self) -> str:
        return f"{self.batch_size}_{self.lr}_{self.lr_decay}_{self.step}"

    @property
    def run_name(self) -> str:
        return f"{self.model_type}_radiomap3dseer"


def build_config(args: argparse.Namespace) -> RadioMapTrainConfig:
    return RadioMapTrainConfig(**vars(args))


def prepare_run_dir(cfg: RadioMapTrainConfig) -> Path:
    run_dir = Path(cfg.output_root) / cfg.run_name / cfg.param_str
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_history_csv(history: list[dict], output_path: Path) -> None:
    # 定义 CSV 列名顺序
    fieldnames = ["epoch", "train_loss", "val_mse", "val_rmse", "val_mae", "val_r2", "best_val_rmse", "lr"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        # 构造基于字典的 CSV 写入器
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # 先写表头
        writer.writeheader()
        # 遍历每一轮训练记录，并逐行写入
        for row in history:
            writer.writerow(row)


def evaluate(model, loader, device) -> tuple[dict[str, float], int]:
    # 关闭 Dropout、使用 BatchNorm
    model.eval()
    totals = {"rmse": 0.0, "mse": 0.0, "mae": 0.0, "r2": 0.0}
    total_samples = 0
    # 关闭梯度计算，减少显存占用并提升推理效率
    with torch.no_grad():
        # _ , _：忽略附带返回的 scene_id 与 tx_id
        for inputs, targets, _, _ in tqdm(loader, desc="Eval", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # torch.clamp 限制输出范围在 [0,1]
            preds = torch.clamp(select_prediction(model(inputs)), 0, 1)
            batch_metrics = compute_regression_metrics(preds, targets)
            batch_size = inputs.size(0)
            for metric_name, metric_value in batch_metrics.items():
                totals[metric_name] += metric_value * batch_size
            total_samples += batch_size
    denom = max(total_samples, 1)
    metrics = {metric_name: metric_total / denom for metric_name, metric_total in totals.items()}
    return metrics, total_samples


def build_eval_loader(cfg: RadioMapTrainConfig) -> DataLoader:
    sample_pairs = resolve_radiomap_sample_pairs(cfg.data_root, cfg.csv_file)
    dataset = RadioMap3DSeerDataset(cfg.data_root, sample_pairs, use_height=cfg.use_height)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
    )


def build_train_val_test_loaders(cfg: RadioMapTrainConfig, run_dir: Path):
    if cfg.val_ratio <= 0:
        raise ValueError("train_ratio + test_ratio must be < 1.0 so the validation split is non-empty.")

    sample_pairs = resolve_radiomap_sample_pairs(cfg.data_root, cfg.csv_file)
    dataset = RadioMap3DSeerDataset(cfg.data_root, sample_pairs, use_height=cfg.use_height)
    dataset_size = len(dataset)
    train_size = int(dataset_size * cfg.train_ratio)
    test_size = int(dataset_size * cfg.test_ratio)
    val_size = dataset_size - train_size - test_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError(
            f"Invalid split sizes: train={train_size}, val={val_size}, test={test_size}, total={dataset_size}"
        )

    # 创建一个带固定随机种子的独立随机数生成器
    split_generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, test_dataset, val_dataset = random_split(
        dataset,
        [train_size, test_size, val_size],
        generator=split_generator,
    )

    def save_split(indices, output_path: Path) -> None:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for item_index in indices:
                scene_id, tx_id = dataset.sample_pairs[item_index]
                writer.writerow([scene_id, tx_id])

    save_split(train_dataset.indices, run_dir / "train_split.csv")
    save_split(val_dataset.indices, run_dir / "val_split.csv")
    save_split(test_dataset.indices, run_dir / "test_split.csv")

    # num_workers：并行加载进程数；
    # pin_memory=torch.cuda.is_available()：若有 CUDA，则启用页锁定内存，加快主机到 GPU 的数据传输；
    # persistent_workers=cfg.num_workers > 0：若使用多进程，则保持 worker 常驻，减少重复创建开销；
    # drop_last=False：最后一个不满批次的 batch 不丢弃。
    loader_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def main() -> None:
    cfg = build_config(parse_args())
    set_seed(cfg.seed)
    device = get_device()
    # 输出根目录 / 实验名 / 参数串 如 surrogate_runs/pmnet_radiomap3dseer/16_0.0001_0.5_10
    run_dir = prepare_run_dir(cfg)

    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        # 把配置对象 cfg 转成普通字典，再以 JSON 格式写入已经打开的文件对象 handle 中，并且用 2 个空格缩进
        json.dump(asdict(cfg), handle, indent=2)

    model = build_model(cfg.model_type, output_stride=cfg.output_stride, in_channels=2)
    if cfg.checkpoint:
        load_checkpoint(model, cfg.checkpoint, strict=True)
    model = model.to(device)

    if cfg.eval_only:
        if not cfg.checkpoint:
            raise ValueError("--eval-only requires --checkpoint")
        eval_loader = build_eval_loader(cfg)
        eval_metrics, eval_sample_count = evaluate(model, eval_loader, device)
        summary = {
            "dataset": "radiomap3dseer",
            "model_type": cfg.model_type,
            "checkpoint": cfg.checkpoint,
            "eval_sample_count": eval_sample_count,
        }
        summary.update(build_prefixed_metric_summary(eval_metrics, prefix="eval"))
        with (run_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2))
        return

    train_loader, val_loader, test_loader = build_train_val_test_loaders(cfg, run_dir)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step, gamma=cfg.lr_decay)

    best_val_rmse = float("inf")
    best_val_metrics: dict[str, float] | None = None
    best_checkpoint_path = run_dir / f"{cfg.model_type}_radiomap3dseer_best.pt"
    history: list[dict] = []
    global_step = 0

    with (run_dir / "train.log").open("w", encoding="utf-8") as log_handle:
        # 和前面的config.json同理
        log_handle.write(json.dumps(asdict(cfg), indent=2) + "\n")
        for epoch in range(cfg.epochs):
            model.train()
            epoch_loss_total = 0.0
            epoch_samples = 0

            for inputs, targets, _, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=True):
                global_step += 1
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad(set_to_none=True)
                preds = select_prediction(model(inputs))
                # nn.MSELoss() 默认 reduction = 'mean'
                loss = MSE(preds, targets)
                loss.backward()
                # 优化器更新参数时需要的不是 loss 这个标量，而是 loss 对各参数的梯度
                optimizer.step()

                # loss.item() == 当前 batch 的全元素平均 MSE
                epoch_loss_total += loss.item() * inputs.size(0)
                epoch_samples += inputs.size(0)

                if global_step % cfg.log_every == 0:
                    message = f"epoch={epoch + 1}, step={global_step}, train_loss={loss.item():.6f}"
                    print(message)
                    log_handle.write(message + "\n")

            train_loss = epoch_loss_total / max(epoch_samples, 1)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            val_metrics = None
            if epoch % cfg.val_freq == 0:
                val_metrics, _ = evaluate(model, val_loader, device)
                if val_metrics["rmse"] < best_val_rmse:
                    best_val_rmse = val_metrics["rmse"]
                    best_val_metrics = dict(val_metrics)
                    save_checkpoint(model, best_checkpoint_path)

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_mse": None if val_metrics is None else val_metrics["mse"],
                    "val_rmse": None if val_metrics is None else val_metrics["rmse"],
                    "val_mae": None if val_metrics is None else val_metrics["mae"],
                    "val_r2": None if val_metrics is None else val_metrics["r2"],
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
                title=f"RadioMap3DSeer / {cfg.model_type}",
                metric_keys=("train_loss", "val_mse", "val_rmse", "val_mae", "val_r2", "best_val_rmse", "lr"),
            )

    best_model = build_model(cfg.model_type, output_stride=cfg.output_stride, in_channels=2)
    load_checkpoint(best_model, str(best_checkpoint_path), strict=True)
    best_model = best_model.to(device)
    test_metrics, test_sample_count = evaluate(best_model, test_loader, device)

    summary = {
        "dataset": "radiomap3dseer",
        "model_type": cfg.model_type,
        "training_mode": "from_scratch",
        "best_val_rmse": best_val_rmse,
        "test_sample_count": test_sample_count,
        "best_checkpoint": str(best_checkpoint_path),
    }
    if best_val_metrics is not None:
        summary.update(build_prefixed_metric_summary(best_val_metrics, prefix="best_val"))
    summary.update(build_prefixed_metric_summary(test_metrics, prefix="test"))
    with (run_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
