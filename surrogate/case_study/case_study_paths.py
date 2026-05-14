"""
用途:
  定义 case_study 子项目的路径常量，供程序化场景生成与导出脚本复用。

直接运行命令:
  无。该文件是公共模块，不单独运行。

参数说明:
  无。

逻辑说明:
  从当前文件位置向上推导项目根、输出目录、测试数据目录，避免硬编码绝对路径。
"""

from __future__ import annotations

from pathlib import Path


CASE_STUDY_DIR = Path(__file__).resolve().parent
SURROGATE_DIR = CASE_STUDY_DIR.parent
PAPER_EXPERIMENT_ROOT = SURROGATE_DIR.parent
CODING_ROOT = PAPER_EXPERIMENT_ROOT.parent

OUTPUT_ROOT = CASE_STUDY_DIR / "output_dataset"
PNG_ROOT = OUTPUT_ROOT / "png"
GAIN_ROOT = OUTPUT_ROOT / "gain"
MESH_ROOT = OUTPUT_ROOT / "meshes"
METADATA_ROOT = OUTPUT_ROOT / "metadata"
BLENDER_ROOT = OUTPUT_ROOT / "blender"
SIONNA_ROOT = OUTPUT_ROOT / "sionna"

SOURCE_TEST_ROOT = CODING_ROOT / "test"
SOURCE_DATASET_ROOT = SOURCE_TEST_ROOT / "dataset"
