"""
用途:
  surrogate 子项目包初始化文件，暴露统一模型注册入口，便于从项目根通过 Python -m 导入运行。

直接运行命令:
  无。该文件是包初始化模块，不单独运行。

导出对象与参数:
  AVAILABLE_MODEL_TYPES / ALL_MODEL_TYPES
    可用模型列表 / 包含显式不可用模型的完整列表。
  build_model(model_type, output_stride=16, in_channels=2)
    model_type: 模型名。
    output_stride: 需要时传给对应模型。
    in_channels: 输入通道数。
  select_prediction(output)
    output: 模型原始输出；返回统一用于 loss/metric 的单个预测张量。
"""

from .model_registry import ALL_MODEL_TYPES, AVAILABLE_MODEL_TYPES, build_model, select_prediction

__all__ = [
    "ALL_MODEL_TYPES",
    "AVAILABLE_MODEL_TYPES",
    "build_model",
    "select_prediction",
]
