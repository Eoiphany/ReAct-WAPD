"""
用途:
  通用代理模型适配层，支持 PMNet、RMNet 和 proxy 三种评估后端。

示例命令:
  无。该文件是公共模块，供环境和主入口导入。

参数说明:
  load_surrogate(model_type='pmnet', model_path=None): 加载指定代理模型。
  infer_surrogate(inputs, model_type='pmnet', model_path=None): 对 HxWx2 输入做推理。
  infer_pmnet(inputs, model_path=None): PMNet 兼容入口。
  infer_rmnet(inputs, model_path=None): RMNet 兼容入口。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
from torchvision import transforms


ROOT_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((ROOT_DIR / "base_config.yaml").read_text(encoding="utf-8")) or {}
MODEL_CFG = CONFIG.get("surrogate_models", {}) if isinstance(CONFIG, dict) else {}
_CACHE: dict[tuple[str, str], tuple[torch.nn.Module, torch.device]] = {}


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _proxy_eval_model(inputs: np.ndarray) -> np.ndarray:
    if inputs.ndim != 3 or inputs.shape[2] < 2:
        return np.zeros(inputs.shape[:2], dtype=np.float32)
    pixel = np.clip(inputs[:, :, 0], 0.0, 1.0).astype(np.float32)
    tx_layer = np.clip(inputs[:, :, 1], 0.0, 1.0).astype(np.float32)
    if float(np.max(tx_layer)) <= 0.0:
        return np.zeros_like(pixel, dtype=np.float32)
    fy = np.fft.fftfreq(tx_layer.shape[0]) * tx_layer.shape[0]
    fx = np.fft.fftfreq(tx_layer.shape[1]) * tx_layer.shape[1]
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * 18.0 * 18.0)).astype(np.float32)
    kernel /= float(np.max(kernel))
    field = np.fft.ifft2(np.fft.fft2(tx_layer) * np.fft.fft2(kernel)).real.astype(np.float32)
    peak = float(np.max(field))
    if peak > 0:
        field /= peak
    attenuation = (1.0 - 0.35 * pixel).astype(np.float32)
    return np.clip(field * attenuation, 0.0, 1.0)


def load_surrogate(model_type: str = "pmnet", model_path: str | None = None) -> Tuple[torch.nn.Module | str, torch.device | None]:
    if model_type == "proxy":
        return "proxy", None

    if model_type not in MODEL_CFG:
        raise ValueError(f"Unsupported model_type: {model_type}")

    cfg = MODEL_CFG[model_type]
    model_py_path = (ROOT_DIR / cfg["model_py_path"]).resolve()
    checkpoint_path = Path(model_path).resolve() if model_path else (ROOT_DIR / cfg["weights_path"]).resolve()
    cache_key = (model_type, str(checkpoint_path))
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    spec = importlib.util.spec_from_file_location(f"react_{model_type}_model", model_py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model definition from {model_py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder_name = cfg.get("builder_name")
    builder = getattr(module, builder_name, None)
    if builder is None:
        raise AttributeError(f"{builder_name} not found in {model_py_path}")

    model = builder(output_stride=16)
    device = _get_device()
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict) and any(str(key).startswith("module.") for key in ckpt):
        ckpt = {str(key).replace("module.", "", 1): value for key, value in ckpt.items()}
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    _CACHE[cache_key] = (model, device)
    return model, device


def infer_surrogate(inputs: np.ndarray, model_type: str = "pmnet", model_path: str | None = None) -> np.ndarray:
    if model_type == "proxy":
        return _proxy_eval_model(inputs)
    model, device = load_surrogate(model_type=model_type, model_path=model_path)
    tensor = transforms.ToTensor()(inputs).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.clip(model(tensor), 0, 1)
    return pred[0, 0].detach().cpu().numpy()


def infer_pmnet(inputs: np.ndarray, model_path: str | None = None) -> np.ndarray:
    return infer_surrogate(inputs, model_type="pmnet", model_path=model_path)


def infer_rmnet(inputs: np.ndarray, model_path: str | None = None) -> np.ndarray:
    return infer_surrogate(inputs, model_type="rmnet", model_path=model_path)


def load_pmnet(model_path: str | None = None):
    return load_surrogate(model_type="pmnet", model_path=model_path)
