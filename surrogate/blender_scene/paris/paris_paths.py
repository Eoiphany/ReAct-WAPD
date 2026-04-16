"""
Shared path definitions for the migrated Paris blender_scene tools.
"""

from __future__ import annotations

from pathlib import Path


PARIS_DIR = Path(__file__).resolve().parent
BLENDER_SCENE_DIR = PARIS_DIR.parent
SURROGATE_DIR = BLENDER_SCENE_DIR.parent
PAPER_EXPERIMENT_ROOT = SURROGATE_DIR.parent
CODING_ROOT = PAPER_EXPERIMENT_ROOT.parent

OUTPUT_DATASET_DIR = PARIS_DIR / "output_dataset"
PNG_DIR = OUTPUT_DATASET_DIR / "png"
GAIN_DIR = OUTPUT_DATASET_DIR / "gain"
ODB_DIR = PARIS_DIR / "odb"
MESHES_DIR = PARIS_DIR / "meshes"
PARIS_SCENE_XML = PARIS_DIR / "paris.xml"

SOURCE_TEST_ROOT = CODING_ROOT / "test"
SOURCE_DATASET_ROOT = SOURCE_TEST_ROOT / "dataset"
SOURCE_TEST_RESULT_ROOT = SOURCE_TEST_ROOT / "test_result"
