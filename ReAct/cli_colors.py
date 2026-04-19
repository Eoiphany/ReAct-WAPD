"""注释
用途:
  终端彩色输出工具。为 ReAct 的单任务脚本和批量实验脚本提供统一的状态标签样式，
  在支持 ANSI 颜色的终端里高亮 START / STEP / DONE / ERROR 等标签；非 TTY 或显式关闭颜色时回退为纯文本。

示例命令:
  python -m ReAct.run_access_point_decision --planner llamafactory --print-step
  python -m ReAct.run_experiment_suite --planner llamafactory --maps-list ReAct/data/maps_test_paths.txt

参数说明:
  supports_color(stream=None): 检查当前输出流是否适合启用 ANSI 颜色。
  status_line(label, message='', tone='info', use_color=None): 生成带状态标签的一行文本。

逻辑说明:
  该模块只负责格式化状态文本，不参与实验逻辑。颜色能力由 TTY / TERM / NO_COLOR 环境共同决定，
  上层脚本只需要组织 message 内容，再调用 status_line 输出。
"""

from __future__ import annotations

import os
import sys
from typing import TextIO


RESET = "\033[0m"
TONE_CODES = {
    "info": "\033[36m",
    "progress": "\033[34m",
    "success": "\033[32m",
    "warn": "\033[33m",
    "error": "\033[31m",
    "accent": "\033[35m",
}


def supports_color(stream: TextIO | None = None) -> bool:
    active_stream = stream or sys.stdout
    is_tty = hasattr(active_stream, "isatty") and active_stream.isatty()
    term = os.environ.get("TERM", "")
    if os.environ.get("NO_COLOR"):
        return False
    return bool(is_tty and term and term.lower() != "dumb")


def status_line(label: str, message: str = "", tone: str = "info", use_color: bool | None = None) -> str:
    enabled = supports_color() if use_color is None else bool(use_color)
    prefix = f"[{label}]"
    if enabled:
        color = TONE_CODES.get(tone, TONE_CODES["info"])
        prefix = f"{color}{prefix}{RESET}"
    if not message:
        return prefix
    return f"{prefix} {message}"
