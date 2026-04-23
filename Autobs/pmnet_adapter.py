"""注释
命令:
1. `python -m Autobs.train_ppo -v single --network-type pmnet`
2. `python -m Autobs.train_ppo -v single --network-type rmnet --model-path /abs/path/to/RMNet.pt`

参数含义:
- `AUTOBS_PMNET_SOURCE / AUTOBS_PMNET_WEIGHTS`: 可选，覆盖默认 PMNet 代码与权重路径。
- `AUTOBS_RMNET_SOURCE / AUTOBS_RMNET_WEIGHTS`: 可选，覆盖默认 RMNet 代码与权重路径。
- `AUTOBS_SURROGATE_NETWORK_TYPE`: 可选，指定训练环境奖励推理默认使用的 surrogate 类型。
- `python -m Autobs.train_ppo`: 训练时环境奖励会通过本文件按指定 surrogate 类型加载模型推理。
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


NETWORK_TYPE_CHOICES = ("pmnet", "pmnet_v3", "rmnet", "rmnet_v3")
_CACHE: dict[tuple[str, str, str], tuple[torch.nn.Module, torch.device]] = {}


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


def _network_family(network_type: str) -> str:
    if network_type.startswith("pmnet"):
        return "pmnet"
    if network_type.startswith("rmnet"):
        return "rmnet"
    raise ValueError(f"Unsupported network_type: {network_type}")


def _resolve_network_type(network_type: str | None = None) -> str:
    raw = (network_type or __import__("os").environ.get("AUTOBS_SURROGATE_NETWORK_TYPE", "pmnet")).strip().lower()
    if raw not in NETWORK_TYPE_CHOICES:
        raise ValueError(f"Unsupported network_type: {raw}")
    return raw


def _resolve_surrogate_source(network_type: str) -> Path:
    family = _network_family(network_type)
    default_source = DEFAULT_PMNET_SOURCE if family == "pmnet" else DEFAULT_RMNET_SOURCE
    env_key = "AUTOBS_PMNET_SOURCE" if family == "pmnet" else "AUTOBS_RMNET_SOURCE"
    raw_path = __import__("os").environ.get(env_key, str(default_source))
    return Path(Path.cwd().joinpath(Path(raw_path))).resolve()


def _resolve_surrogate_weights(network_type: str, model_path: str | None = None) -> Path:
    family = _network_family(network_type)
    default_weights = DEFAULT_PMNET_WEIGHTS if family == "pmnet" else DEFAULT_RMNET_WEIGHTS
    env_key = "AUTOBS_PMNET_WEIGHTS" if family == "pmnet" else "AUTOBS_RMNET_WEIGHTS"
    raw_path = model_path or __import__("os").environ.get(env_key, str(default_weights))
    return Path(raw_path).expanduser().resolve()


def _load_model_class(network_type: str):
    source_path = _resolve_surrogate_source(network_type)
    if not source_path.exists():
        raise FileNotFoundError(f"Surrogate source file not found: {source_path}")
    spec = importlib.util.spec_from_file_location("AutobsLocalSurrogate", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load surrogate model from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_name = "PMNet" if _network_family(network_type) == "pmnet" else "RMNet"
    model_cls = getattr(module, class_name, None)
    if model_cls is None:
        raise AttributeError(f"{class_name} class not found in {source_path}")
    return model_cls


def _build_model(model_cls, network_type: str):
    output_stride = 8 if network_type.endswith("_v3") else 16
    return model_cls(
        n_blocks=[3, 3, 27, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=output_stride,
    )


def load_surrogate(model_path: str | None = None, network_type: str | None = None) -> Tuple[torch.nn.Module, torch.device]:
    resolved_network_type = _resolve_network_type(network_type)
    source_path = _resolve_surrogate_source(resolved_network_type)
    weights_path = _resolve_surrogate_weights(resolved_network_type, model_path=model_path)
    cache_key = (resolved_network_type, str(source_path), str(weights_path))
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not weights_path.exists():
        raise FileNotFoundError(f"Surrogate weights file not found: {weights_path}")

    model_cls = _load_model_class(resolved_network_type)
    model = _build_model(model_cls, resolved_network_type)
    device = _get_device()
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and any(key.startswith("module.") for key in checkpoint):
        checkpoint = {key.replace("module.", "", 1): value for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    _CACHE[cache_key] = (model, device)
    return model, device


def infer_surrogate(
    inputs: np.ndarray,
    model_path: str | None = None,
    network_type: str | None = None,
) -> np.ndarray:
    model, device = load_surrogate(model_path=model_path, network_type=network_type)
    tensor = transforms.ToTensor()(inputs).float().unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(tensor)
        prediction = torch.nan_to_num(prediction, nan=0.0, posinf=1.0, neginf=0.0)
        prediction = torch.clamp(prediction, 0, 1)
    return prediction[0, 0].detach().cpu().numpy()


def infer_pmnet(inputs: np.ndarray, model_path: str | None = None) -> np.ndarray:
    return infer_surrogate(inputs, model_path=model_path, network_type="pmnet")
