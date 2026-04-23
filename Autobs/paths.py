"""注释
命令:
- `Python -m Autobs.train_ppo`

参数含义:
- `Python`: 使用当前环境的 Python 解释器。
- `-m Autobs.train_ppo`: 运行 PPO 训练入口，内部会依赖本文件定义的项目内路径。
- 训练运行时默认读取 `Autobs/config.yaml`，默认写入 `Autobs/checkpoints`，默认读取 `Autobs/models`。
- 训练地图默认来自 `paper_experiment/dataset/png/buildingsWHeight`，不再默认单图训练。
"""

from __future__ import annotations

from pathlib import Path

# 获取当前 Python 文件所在的目录路径
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG_PATH = PACKAGE_ROOT / "config.yaml"
ASSETS_DIR = PACKAGE_ROOT / "assets"
MODELS_DIR = PACKAGE_ROOT / "models"
CHECKPOINT_DIR = PACKAGE_ROOT / "checkpoints"
DEFAULT_CITY_MAP_PATH = ASSETS_DIR / "USC_city_map.png"
DEFAULT_PMNET_SOURCE = MODELS_DIR / "PMNet.py"
DEFAULT_PMNET_WEIGHTS = MODELS_DIR / "PMNet.pt"
DEFAULT_RMNET_SOURCE = MODELS_DIR / "RMNet.py"
DEFAULT_RMNET_WEIGHTS = MODELS_DIR / "RMNet.pt"


def resolve_default_dataset_map_dir(project_root: Path) -> Path:
    project_root = Path(project_root).resolve()
    candidates = (
        project_root / "dataset" / "png" / "buildingsWHeight",
        project_root.parent / "coding" / "test" / "buildingsWHeight",
        project_root.parent / "coding" / "test" / "dataset" / "png" / "buildingsWHeight",
        project_root.parent / "test" / "buildingsWHeight",
        project_root.parent / "test" / "dataset" / "png" / "buildingsWHeight",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_DATASET_MAP_DIR = resolve_default_dataset_map_dir(PROJECT_ROOT)


def ensure_runtime_dirs() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
