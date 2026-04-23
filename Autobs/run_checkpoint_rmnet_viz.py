"""注释
命令示例:
python -m Autobs.run_checkpoint_rmnet_viz \
  --image dataset/png/buildingsWHeight/348.png \
  --checkpoint Autobs/checkpoints \
  --model-path Autobs/models/RMNet.pt \
  --network-type rmnet \
  --output-dir Autobs/outputs/checkpoint_viz

用途:
- 对单张灰度高度图构造 PPO checkpoint 所需 observation。
- 使用 checkpoint 选一个动作（单站点部署位置）。
- 使用 surrogate 模型（PMNet / RMNet）预测 radiomap，并计算 coverage / spectral efficiency / score。
- 输出可视化 PNG 和 JSON 摘要。

实现说明:
- 策略动作空间本体是 `32 x 32 = 1024` 个离散 action，代码里实际存储和输出的是一维 `action id`；后续通过 `divmod(action, 32)` 与上采样规则映射回原图像素坐标。
- 可视化中的 `Action Mask` 面板不是原始 `32 x 32` 小图，而是把该网格按图像尺寸放大成256后的显示结果，即每个方格都是256/32=8*8的方格；紫色块表示 `mask=0` 的非法动作，亮色块表示 `mask=1` 的合法动作，白色 `x` 仅用于标记 checkpoint 最终选中的 action 对应位置，并不表示额外的第三种 mask 取值。

输出示例:
- `*_checkpoint_rmnet_viz.png`: 输入图、放大的 action mask、surrogate pathgain 热力图与 coverage 面板。
- `*_checkpoint_rmnet_summary.json`: 记录 `action id`、像素坐标 `(tx_row, tx_col)` 与各项指标。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from Autobs.env.utils import (
    DB_MIN,
    calc_action_mask,
    calc_upsampling_loc,
    get_stats,
    load_map_normalized,
)
from Autobs.paths import CHECKPOINT_DIR, DEFAULT_RMNET_WEIGHTS, PACKAGE_ROOT

os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")


DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs" / "checkpoint_viz"
LATIN_FONT_FAMILY = "Times New Roman"
CHINESE_FONT_CANDIDATES = ["SimSun", "Songti SC"]
MODULE_OBS_KEY = "obs"
MODULE_ACTIONS_KEY = "actions"
MODULE_ACTION_DIST_INPUTS_KEY = "action_dist_inputs"


class CheckpointInferenceCompatibilityError(RuntimeError):
    """Raised when an agent cannot be used through the RLModule inference path."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one PPO checkpoint on one image and visualize surrogate output")
    parser.add_argument("--image", required=True, type=str, help="Input grayscale height map")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR), type=str, help="RLlib checkpoint directory")
    parser.add_argument("--model-path", default=str(DEFAULT_RMNET_WEIGHTS), type=str, help="Surrogate weights path")
    parser.add_argument(
        "--network-type",
        default="rmnet",
        choices=["pmnet", "pmnet_v3", "rmnet", "rmnet_v3"],
        help="Surrogate variant",
    )
    parser.add_argument("--device", default="auto", type=str, help="Inference device: auto/cpu/cuda/mps")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), type=str, help="Directory for PNG/JSON outputs")
    parser.add_argument("--explore", action="store_true", help="Use exploratory sampling instead of deterministic action")
    return parser


def get_device(device_name: str):
    import torch

    if device_name != "auto":
        return torch.device(device_name)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_policy_observation(pixel_map: np.ndarray) -> dict[str, np.ndarray]:
    clipped = np.clip(np.asarray(pixel_map, dtype=np.float32), 0.0, 1.0)
    return {
        "observations": clipped.reshape(-1).astype(np.float32),
        "action_mask": calc_action_mask(clipped).astype(np.float32),
    }


def extract_action(action_result: Any) -> int:
    if isinstance(action_result, tuple):
        action_result = action_result[0]
    return int(np.asarray(action_result).item())


def init_model(network_type: str):
    if network_type == "pmnet":
        from Autobs.models.PMNet import PMNet

        return PMNet(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=16,
        )
    if network_type == "pmnet_v3":
        from Autobs.models.PMNet import PMNet

        return PMNet(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=8,
        )
    if network_type == "rmnet":
        from Autobs.models.RMNet import RMNet

        return RMNet(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=16,
        )
    if network_type == "rmnet_v3":
        from Autobs.models.RMNet import RMNet

        return RMNet(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=8,
        )
    raise ValueError(f"Unsupported surrogate type: {network_type}")


def infer_checkpoint_family(state: dict[str, Any] | Any) -> str | None:
    if not isinstance(state, dict):
        return None
    keys = list(state.keys())
    if any(key.startswith("module.") for key in keys):
        keys = [key.replace("module.", "", 1) for key in keys]
    if any(key.startswith("fuse") or key.startswith("context.") or key.startswith("input_fuse.") for key in keys):
        return "rmnet"
    return None


def load_surrogate_model(model_path: str | Path, network_type: str, device_name: str):
    import torch

    device = get_device(device_name)
    model = init_model(network_type)
    state = torch.load(str(Path(model_path).expanduser().resolve()), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(key.startswith("module.") for key in state):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}

    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        inferred = infer_checkpoint_family(state)
        hint = ""
        if inferred and inferred != network_type and not network_type.startswith(inferred):
            hint = f" Checkpoint looks like `{inferred}` weights, but `--network-type` is `{network_type}`."
        raise RuntimeError(f"{exc}{hint}") from exc

    model.to(device)
    model.eval()
    return model, device


def numpy_image_to_tensor(array: np.ndarray):
    import torch

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        tensor = torch.from_numpy(arr).unsqueeze(0)
    elif arr.ndim == 3:
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    return tensor


class RMNetPredictor:
    def __init__(self, model_path: str | Path, network_type: str, device_name: str) -> None:
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.network_type = network_type
        self.model, self.device = load_surrogate_model(self.model_path, network_type, device_name)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        import torch

        tensor = numpy_image_to_tensor(inputs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(tensor)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
            pred = torch.clamp(pred, 0.0, 1.0)
        return pred.detach().cpu().squeeze().numpy().astype(np.float32)


def load_checkpoint_agent(checkpoint_path: str | Path):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    try:
        from ray.rllib.algorithms.algorithm import Algorithm
        from ray.util.annotations import RayDeprecationWarning
    except ModuleNotFoundError:
        return load_numpy_checkpoint_policy(checkpoint_path)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=RayDeprecationWarning)
            return Algorithm.from_checkpoint(str(checkpoint_path))
    except AttributeError as exc:
        if "to_dict" not in str(exc):
            raise
        return load_numpy_checkpoint_policy(checkpoint_path)


def _extract_batched_action(action_value: Any) -> int:
    if hasattr(action_value, "detach"):
        action_value = action_value.detach()
    if hasattr(action_value, "cpu"):
        action_value = action_value.cpu()
    return int(np.asarray(action_value).reshape(-1)[0].item())


def _get_module_device(module: Any) -> Any:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return None
    try:
        first_param = next(parameters())
    except (StopIteration, TypeError):
        return None
    return getattr(first_param, "device", None)


def _as_batched_module_obs(value: Any, torch_module, device: Any) -> Any:
    if isinstance(value, dict):
        return {key: _as_batched_module_obs(item, torch_module, device) for key, item in value.items()}
    tensor = torch_module.as_tensor(np.asarray(value), dtype=torch_module.float32, device=device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    return tensor.unsqueeze(0)


def _compute_checkpoint_action_via_rl_module(agent: Any, observation: dict[str, np.ndarray], explore: bool) -> int:
    get_module = getattr(agent, "get_module", None)
    if not callable(get_module):
        raise CheckpointInferenceCompatibilityError("Agent does not expose get_module()")

    try:
        module = get_module()
    except AttributeError as exc:
        raise CheckpointInferenceCompatibilityError(str(exc)) from exc
    if module is None:
        raise CheckpointInferenceCompatibilityError("Algorithm.get_module() returned None")

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise CheckpointInferenceCompatibilityError("torch is required for RLModule inference") from exc

    forward = getattr(module, "forward_exploration" if explore else "forward_inference", None)
    if not callable(forward):
        raise CheckpointInferenceCompatibilityError("RLModule does not expose the expected forward method")

    device = _get_module_device(module)
    batch = {MODULE_OBS_KEY: _as_batched_module_obs(observation, torch, device)}
    with torch.no_grad():
        outputs = forward(batch)

    if MODULE_ACTIONS_KEY in outputs:
        return _extract_batched_action(outputs[MODULE_ACTIONS_KEY])

    if MODULE_ACTION_DIST_INPUTS_KEY not in outputs:
        raise CheckpointInferenceCompatibilityError("RLModule output does not contain actions or action_dist_inputs")

    dist_cls_getter = getattr(
        module,
        "get_exploration_action_dist_cls" if explore else "get_inference_action_dist_cls",
        None,
    )
    if not callable(dist_cls_getter):
        raise CheckpointInferenceCompatibilityError("RLModule does not expose an action distribution class")

    action_dist = dist_cls_getter().from_logits(outputs[MODULE_ACTION_DIST_INPUTS_KEY])
    if not explore and hasattr(action_dist, "to_deterministic"):
        action_dist = action_dist.to_deterministic()
    return _extract_batched_action(action_dist.sample())


def compute_checkpoint_action(agent, observation: dict[str, np.ndarray], explore: bool) -> int:
    if hasattr(agent, "get_module"):
        try:
            return _compute_checkpoint_action_via_rl_module(agent, observation, explore)
        except CheckpointInferenceCompatibilityError:
            pass
    action_result = agent.compute_single_action(observation=observation, explore=explore)
    return extract_action(action_result)


def _extract_layer_indices(state: dict[str, np.ndarray], prefix: str) -> list[int]:
    indices = []
    prefix = f"{prefix}."
    for key in state:
        if not key.startswith(prefix) or not key.endswith(".weight"):
            continue
        suffix = key[len(prefix) :]
        layer_index = suffix.split(".", 1)[0]
        if layer_index.isdigit():
            indices.append(int(layer_index))
    return sorted(set(indices))


class NumpyCheckpointPolicy:
    def __init__(self, state: dict[str, np.ndarray]) -> None:
        import torch

        self.torch = torch
        self.encoder_layers = [
            (
                torch.from_numpy(np.asarray(state[f"encoder.actor_encoder.net.mlp.{idx}.weight"], dtype=np.float32)),
                torch.from_numpy(np.asarray(state[f"encoder.actor_encoder.net.mlp.{idx}.bias"], dtype=np.float32)),
            )
            for idx in _extract_layer_indices(state, "encoder.actor_encoder.net.mlp")
        ]
        self.pi_weight = torch.from_numpy(np.asarray(state["pi.net.mlp.0.weight"], dtype=np.float32))
        self.pi_bias = torch.from_numpy(np.asarray(state["pi.net.mlp.0.bias"], dtype=np.float32))
        if not self.encoder_layers:
            raise ValueError("No actor encoder layers found in checkpoint state")

    def _forward_encoder(self, obs: np.ndarray):
        hidden = self.torch.from_numpy(np.asarray(obs, dtype=np.float32))
        for weight, bias in self.encoder_layers:
            hidden = self.torch.relu(weight @ hidden + bias)
        return hidden

    def compute_single_action(self, observation: dict[str, np.ndarray], explore: bool = False) -> int:
        logits = self.pi_weight @ self._forward_encoder(observation["observations"]) + self.pi_bias
        action_mask = self.torch.from_numpy(np.asarray(observation["action_mask"], dtype=np.float32))
        masked_logits = self.torch.where(action_mask > 0.0, logits, self.torch.full_like(logits, -1e30))

        if explore:
            valid = action_mask > 0.0
            valid_indices = self.torch.nonzero(valid, as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                raise ValueError("Action mask contains no valid actions")
            sampled = self.torch.distributions.Categorical(logits=masked_logits[valid]).sample()
            return int(valid_indices[sampled].item())
        return int(self.torch.argmax(masked_logits).item())

    def stop(self) -> None:
        return None


def load_numpy_checkpoint_policy(checkpoint_path: str | Path) -> NumpyCheckpointPolicy:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    module_state_path = checkpoint_path / "learner_group" / "learner" / "rl_module" / "default_policy" / "module_state.pkl"
    if not module_state_path.exists():
        raise FileNotFoundError(f"Fallback module_state not found: {module_state_path}")
    with module_state_path.open("rb") as handle:
        state = load_pickle_state(handle)
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint state type: {type(state)!r}")
    return NumpyCheckpointPolicy(state)


def load_pickle_state(handle):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*numpy\.core\.numeric is deprecated.*",
            category=DeprecationWarning,
        )
        return pickle.load(handle)


def action_mask_grid(action_mask: np.ndarray) -> np.ndarray:
    side = int(round(np.sqrt(action_mask.size)))
    if side * side != action_mask.size:
        raise ValueError(f"Action mask cannot be reshaped into square grid: {action_mask.size}")
    return action_mask.reshape(side, side)


def upsample_action_grid(grid: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    scale_y = max(target_shape[0] // grid.shape[0], 1)
    scale_x = max(target_shape[1] // grid.shape[1], 1)
    upsampled = np.repeat(np.repeat(grid, scale_y, axis=0), scale_x, axis=1)
    return upsampled[: target_shape[0], : target_shape[1]]


def render_visualization(
    pixel_map: np.ndarray,
    pathgain_db: np.ndarray,
    action_mask: np.ndarray,
    tx_loc: tuple[int, int],
    metrics: dict[str, float],
    output_path: str | Path,
) -> None:
    mpl_config_dir = PACKAGE_ROOT / ".mplconfig"
    cache_dir = PACKAGE_ROOT / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    chinese_font = next((name for name in CHINESE_FONT_CANDIDATES if name in available_fonts), CHINESE_FONT_CANDIDATES[0])
    matplotlib.rcParams["font.family"] = [LATIN_FONT_FAMILY, chinese_font]
    matplotlib.rcParams["axes.unicode_minus"] = False

    row, col = tx_loc
    action_grid = upsample_action_grid(action_mask_grid(action_mask), pixel_map.shape)
    coverage_map = pathgain_db >= -117.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    ax = axes[0, 0]
    ax.imshow(pixel_map, cmap="gray", vmin=0.0, vmax=1.0)
    ax.scatter([col], [row], c="red", s=36, marker="x")
    ax.set_title("Input Height Map")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(action_grid, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.scatter([col], [row], c="white", s=36, marker="x")
    ax.set_title("Action Mask")
    ax.axis("off")

    ax = axes[1, 0]
    heatmap = ax.imshow(pathgain_db, cmap="turbo", vmin=DB_MIN, vmax=np.max(pathgain_db))
    ax.scatter([col], [row], c="white", s=36, marker="x")
    ax.set_title("RMNet Pathgain (dB)")
    ax.axis("off")
    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    ax.imshow(coverage_map, cmap="magma", vmin=0.0, vmax=1.0)
    ax.scatter([col], [row], c="cyan", s=36, marker="x")
    ax.set_title(
        "Coverage / Score\n"
        f"coverage={metrics['coverage']:.4f}  "
        f"se={metrics['spectral_efficiency']:.4f}  "
        f"score={metrics['score']:.4f}"
    )
    ax.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def evaluate_checkpoint(
    image_path: str | Path,
    checkpoint_path: str | Path,
    model_path: str | Path,
    network_type: str,
    device_name: str,
    output_dir: str | Path,
    explore: bool = False,
) -> dict[str, Any]:
    image_path = Path(image_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pixel_map = load_map_normalized(image_path)
    observation = build_policy_observation(pixel_map)
    agent = load_checkpoint_agent(checkpoint_path)
    predictor = RMNetPredictor(model_path=model_path, network_type=network_type, device_name=device_name)

    try:
        action = compute_checkpoint_action(agent, observation, explore=explore)
    finally:
        stop = getattr(agent, "stop", None)
        if callable(stop):
            stop()

    tx_loc = calc_upsampling_loc(action, pixel_map)
    pathgain_db, metrics = get_stats(pixel_map, [tx_loc], pmnet=predictor)

    stem = image_path.stem
    png_path = output_dir / f"{stem}_checkpoint_rmnet_viz.png"
    json_path = output_dir / f"{stem}_checkpoint_rmnet_summary.json"
    render_visualization(
        pixel_map=pixel_map,
        pathgain_db=pathgain_db,
        action_mask=observation["action_mask"],
        tx_loc=tx_loc,
        metrics=metrics,
        output_path=png_path,
    )

    summary = {
        "image": str(image_path),
        "checkpoint": str(checkpoint_path),
        "model_path": str(Path(model_path).expanduser().resolve()),
        "network_type": network_type,
        "device": str(get_device(device_name)),
        "action": int(action),
        "tx_row": int(tx_loc[0]),
        "tx_col": int(tx_loc[1]),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "visualization": str(png_path),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = evaluate_checkpoint(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        model_path=args.model_path,
        network_type=args.network_type,
        device_name=args.device,
        output_dir=args.output_dir,
        explore=args.explore,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
