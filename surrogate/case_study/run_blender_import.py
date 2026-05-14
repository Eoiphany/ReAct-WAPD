"""
用途:
  调用本机 Blender，把 synthetic scene manifest 导入成 .blend 文件。

直接运行命令:
  python -m paper_experiment.surrogate.case_study.run_blender_import
  python -m paper_experiment.surrogate.case_study.run_blender_import \
    --manifest /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/case_study/output_dataset/blender/synthetic_scene_0_manifest.json \
    --output-blend /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/case_study/output_dataset/blender/synthetic_scene_0.blend

参数说明:
  --blender-bin: Blender 可执行文件路径。
  --manifest: 场景 manifest JSON 路径。
  --output-blend: 输出 .blend 路径。

逻辑说明:
  该脚本只是 Python 封装层，负责组装 Blender 后台命令，真正的导入逻辑在
  blender_import_synthetic_scene.py 中执行。
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .case_study_paths import BLENDER_ROOT


DEFAULT_BLENDER_BIN = Path("/Applications/Blender.app/Contents/MacOS/Blender")
DEFAULT_MANIFEST = BLENDER_ROOT / "synthetic_scene_0_manifest.json"
DEFAULT_OUTPUT_BLEND = BLENDER_ROOT / "synthetic_scene_0.blend"
SCRIPT_PATH = Path(__file__).resolve().parent / "blender_import_synthetic_scene.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import a synthetic case-study scene into Blender.")
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER_BIN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-blend", type=Path, default=DEFAULT_OUTPUT_BLEND)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cmd = [
        str(args.blender_bin),
        "--background",
        "--factory-startup",
        "--python",
        str(SCRIPT_PATH),
        "--",
        "--manifest",
        str(args.manifest),
        "--output-blend",
        str(args.output_blend),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
