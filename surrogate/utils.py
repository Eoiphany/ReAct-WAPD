"""
用途:
  共享训练/评估运行时工具，例如设备选择、随机种子、指标函数、checkpoint 读写与训练曲线绘图。

直接运行命令:
  无。该文件是公共模块，不单独运行。

导出对象与参数:
  set_seed(seed)
    seed: 随机种子。
  get_device()
    无参数，自动返回 cuda / mps / cpu。
  MSE(pred, target) / RMSE(pred, target) / MAE(pred, target) / R2(pred, target)
    pred: 模型预测张量。
    target: 标签张量。
  load_checkpoint(model, checkpoint_path, strict=True)
    model: 要加载权重的模型。
    checkpoint_path: 权重路径。
    strict: 是否严格匹配参数名。
  save_checkpoint(model, checkpoint_path)
    model: 要保存的模型。
    checkpoint_path: 输出权重路径。
  save_training_plots(history, output_path, title, metric_keys)
    history: 训练历史列表，每个元素是按 epoch 记录的字典。
    output_path: 输出 PNG 路径。
    title: 图标题。
    metric_keys: 要绘制的指标字段名列表。
"""

from __future__ import annotations

import random
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# 指标函数统一特点
# 输入：batch 张量
# 输出：标量
# 默认：按所有元素平均或求和

# nn.MSELoss() 默认 reduction='mean'
# 对所有元素求平均
# 输出：标量
def MSE(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss()(pred, target)


def RMSE(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean().sqrt()


def MAE(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def R2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_mean = target.mean()
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target_mean) ** 2).sum()
    if float(ss_tot.item()) == 0.0:
        return torch.tensor(0.0, device=pred.device)
    return 1.0 - ss_res / ss_tot


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, strict: bool = True) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(key.startswith("module.") for key in state):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
    model.load_state_dict(state, strict=strict)


def save_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


def save_training_plots(
    history: list[dict],
    output_path: Path,
    title: str,
    metric_keys: tuple[str, ...],
) -> None:
    if not history:
        return

    import math

    mpl_config_dir = output_path.parent / ".mpl-cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history if "epoch" in row]
    if not epochs:
        return

    valid_metric_keys = []
    for key in metric_keys:
        values = [row.get(key) for row in history]
        if any(value is not None for value in values):
            valid_metric_keys.append(key)

    if not valid_metric_keys:
        return

    ncols = 2 if len(valid_metric_keys) > 1 else 1
    nrows = math.ceil(len(valid_metric_keys) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, key in zip(axes, valid_metric_keys):
        xs = []
        ys = []
        for row in history:
            value = row.get(key)
            if value is None:
                continue
            xs.append(row["epoch"])
            ys.append(value)
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3)
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(valid_metric_keys) :]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
