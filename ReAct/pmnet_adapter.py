"""注释
命令:
1. `python -m Autobs.train_ppo -v single`
2. `AUTOBS_PMNET_WEIGHTS=/abs/path/to/PMNet.pt \
    python -m Autobs.train_ppo \
    -v multi`

参数含义:
- `AUTOBS_PMNET_SOURCE`: 可选，覆盖默认 `Autobs/models/PMNet.py` 源码路径。
- `AUTOBS_PMNET_WEIGHTS`: 可选，覆盖默认 `Autobs/models/PMNet.pt` 权重路径。
- `python -m Autobs.train_ppo`: 训练时环境奖励会通过本文件加载 PMNet 代理模型进行推理。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torchvision import transforms

from Autobs.paths import (
    DEFAULT_PMNET_SOURCE,
    DEFAULT_PMNET_WEIGHTS,
    DEFAULT_RMNET_SOURCE,
    DEFAULT_RMNET_WEIGHTS,
)


_CACHE = {"model": None, "device": None}


def get_default_artifact_paths() -> dict[str, Path]:
    return {
        "pmnet_source": Path(DEFAULT_PMNET_SOURCE),
        "pmnet_weights": Path(DEFAULT_PMNET_WEIGHTS),
        "rmnet_source": Path(DEFAULT_RMNET_SOURCE),
        "rmnet_weights": Path(DEFAULT_RMNET_WEIGHTS),
    }


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_pmnet_source() -> Path:
    return Path(
        Path.cwd().joinpath(
            Path(
                __import__("os").environ.get("AUTOBS_PMNET_SOURCE", str(DEFAULT_PMNET_SOURCE))
            )
        )
    ).resolve()


def _resolve_pmnet_weights(model_path: str | None = None) -> Path:
    raw_path = model_path or __import__("os").environ.get("AUTOBS_PMNET_WEIGHTS", str(DEFAULT_PMNET_WEIGHTS))
    return Path(raw_path).expanduser().resolve()


def _load_pmnet_class():
    source_path = _resolve_pmnet_source()
    if not source_path.exists():
        raise FileNotFoundError(f"PMNet source file not found: {source_path}")
    spec = importlib.util.spec_from_file_location("AutobsLocalPMNet", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PMNet from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pmnet_cls = getattr(module, "PMNet", None)
    if pmnet_cls is None:
        raise AttributeError(f"PMNet class not found in {source_path}")
    return pmnet_cls


def load_pmnet(model_path: str | None = None) -> Tuple[torch.nn.Module, torch.device]:
    if _CACHE["model"] is not None:
        return _CACHE["model"], _CACHE["device"]

    weights_path = _resolve_pmnet_weights(model_path=model_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"PMNet weights file not found: {weights_path}")

    pmnet_cls = _load_pmnet_class()
    model = pmnet_cls(
        n_blocks=[3, 3, 27, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=16,
    )
    device = _get_device()
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    _CACHE["model"] = model
    _CACHE["device"] = device
    return model, device


def infer_pmnet(inputs: np.ndarray, model_path: str | None = None) -> np.ndarray:
    model, device = load_pmnet(model_path=model_path)
    tensor = transforms.ToTensor()(inputs).float().unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = torch.clip(model(tensor), 0, 1)
    return prediction[0, 0].detach().cpu().numpy()
