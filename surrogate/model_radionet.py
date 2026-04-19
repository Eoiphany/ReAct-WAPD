"""
用途:
  RadioNet 模型占位文件。当前仅提供显式失败入口，避免在论文公开细节不足时伪造实现。

直接运行命令:
  无。该文件是模型定义模块，供其他脚本导入。

导出对象与参数:
  build_radionet(*args, **kwargs)
    始终抛出 NotImplementedError，并说明当前缺少可靠复现所需的完整公开实现细节。
"""

from __future__ import annotations


def build_radionet(*args, **kwargs):
    raise NotImplementedError(
        "RadioNet is not implemented: the paper describes the spread-layer concept and key hyper-parameters, "
        "but no official code or sufficiently complete layer-by-layer public specification was found to "
        "reconstruct a reliable implementation without guessing."
    )
