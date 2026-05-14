"""注释
命令示例:
python -m py_compile visualization/image_jobs.py

参数含义:
- `VisualizationRuntimePaths`: 统一管理 visualization 目录下的运行时缓存路径。
- `write_sites_payload(...)`: 把当前站点集合写成当前目录可复用的 JSON 文件。
- `build_roi_command(...)`: 生成左图包装脚本命令。
- `build_prediction_command(...)`: 生成右图包装脚本命令。

逻辑说明:
本文件只负责把实时展示相关脚本和中间文件统一收口到 visualization 目录，避免 GUI 直接散拼外部命令和路径。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VisualizationRuntimePaths:
    root: Path
    runtime_root: Path
    traj_root: Path
    images_root: Path
    roi_root: Path
    pred_root: Path
    heatmap_root: Path
    trend_root: Path
    sites_root: Path

    @classmethod
    def from_root(cls, root: Path) -> "VisualizationRuntimePaths":
        runtime_root = root / "runtime"
        return cls(
            root=root,
            runtime_root=runtime_root,
            traj_root=runtime_root / "trajs",
            images_root=runtime_root / "images",
            roi_root=runtime_root / "images" / "roi",
            pred_root=runtime_root / "images" / "pred",
            heatmap_root=runtime_root / "images" / "heatmap",
            trend_root=runtime_root / "images" / "trend",
            sites_root=runtime_root / "sites",
        )

    def ensure(self) -> None:
        for path in (
            self.runtime_root,
            self.traj_root,
            self.images_root,
            self.roi_root,
            self.pred_root,
            self.heatmap_root,
            self.trend_root,
            self.sites_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


def write_sites_payload(output_path: Path, sites: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"sites": sites}, ensure_ascii=False, indent=2), encoding="utf-8")


def build_roi_command(
    *,
    python_executable: Path,
    wrapper_script: Path,
    map_path: Path,
    sites_json_path: Path,
    output_path: Path,
    eval_model: str,
) -> list[str]:
    return [
        str(python_executable),
        str(wrapper_script),
        "--map-path",
        str(map_path),
        "--sites-file",
        str(sites_json_path),
        "--output-path",
        str(output_path),
        "--eval-model",
        str(eval_model),
    ]


def build_prediction_command(
    *,
    python_executable: Path,
    wrapper_script: Path,
    map_path: Path,
    sites_json_path: Path,
    output_dir: Path,
    eval_model: str,
) -> list[str]:
    return [
        str(python_executable),
        str(wrapper_script),
        "--map-path",
        str(map_path),
        "--sites-file",
        str(sites_json_path),
        "--output-dir",
        str(output_dir),
        "--eval-model",
        str(eval_model),
    ]
