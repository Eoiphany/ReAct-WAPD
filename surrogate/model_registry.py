"""
用途:
  统一代理模型注册与构造入口，供 USC / RadioMap3DSeer 训练脚本共用。

直接运行命令:
  无。该文件是公共模块，不单独运行。

导出对象与参数:
  AVAILABLE_MODEL_TYPES / ALL_MODEL_TYPES
    AVAILABLE_MODEL_TYPES: 当前可直接构造并训练的模型名列表。
    ALL_MODEL_TYPES: 包含显式不可用模型的完整列表。
  build_model(model_type, output_stride=16, in_channels=2)
    model_type: 模型名。
    output_stride: 需要时传给对应模型。
    in_channels: 输入通道数。
  select_prediction(output)
    output: 模型原始输出；若是多输出结构，则统一返回最后一个预测张量。
"""

from __future__ import annotations

from typing import Any

import torch

try:
    from .model_pmnet import build_pmnet
    from .model_radionet import build_radionet
    from .model_radiounet import build_radiounet
    from .model_rmnet import build_rmnet
    from .model_transunet import build_transunet
    from .model_unet import build_unet
except ImportError:
    from model_pmnet import build_pmnet
    from model_radionet import build_radionet
    from model_radiounet import build_radiounet
    from model_rmnet import build_rmnet
    from model_transunet import build_transunet
    from model_unet import build_unet

AVAILABLE_MODEL_TYPES = ("pmnet", "rmnet", "unet", "transunet", "radiounet")
ALL_MODEL_TYPES = AVAILABLE_MODEL_TYPES + ("radionet",)


def build_model(model_type: str, output_stride: int = 16, in_channels: int = 2):
    if model_type == "pmnet":
        return build_pmnet(output_stride=output_stride)
    if model_type == "rmnet":
        return build_rmnet(output_stride=output_stride)
    if model_type == "unet":
        return build_unet(in_channels=in_channels)
    if model_type == "transunet":
        return build_transunet(in_channels=in_channels)
    if model_type == "radiounet":
        return build_radiounet(in_channels=in_channels)
    if model_type == "radionet":
        return build_radionet(in_channels=in_channels, output_stride=output_stride)
    raise ValueError(f"Unsupported model_type: {model_type}")


def select_prediction(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output:
        return select_prediction(output[-1])
    raise TypeError(f"Unsupported model output type: {type(output)!r}")
