"""注释
命令:
1. `python -m ReAct.run_access_point_decision --init-mode two_stage --two-stage-module-state ../Autobs/outputs/rerank/best_module_state.pt`
2. `python -m ReAct.run_experiment_suite --init-mode two_stage --two-stage-module-state ../Autobs/outputs/rerank/best_module_state.pt`

参数含义:
- `infer_module_state_observation_version(module_state_path)`: 根据二阶段策略模块权重的输入通道数推断 `single/multi` 观测版本。
- `init_locs_from_two_stage_policy(city_map_path, module_state_path, version, top_k, device_name, coverage_threshold_db)`:
  使用二阶段策略模块按顺序选择初始站点位置。

脚本逻辑说明:
本文件从 Autobs 的二阶段策略学习链路中抽取最小可复用推理实现，仅保留
“加载 `best_module_state.pt` -> 构造 single/multi observation -> 做 action masking ->
按顺序选点”这一段。这样 ReAct 可以直接消费二阶段策略模块权重，而不再依赖
PPO checkpoint 或 Ray/RLlib 的运行时恢复接口。
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

if __package__ in {None, ""}:
    _CURRENT_FILE = Path(__file__).resolve()
    _PACKAGE_PARENT = _CURRENT_FILE.parent.parent
    if str(_PACKAGE_PARENT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_PARENT))

from Autobs.utils import (
    DEFAULT_SCORE_MARGIN_SCALE_DB as AUTOBS_DEFAULT_SCORE_MARGIN_SCALE_DB,
    compute_score_components as compute_autobs_score_components,
)
from env_utils import (
    TX_POWER_DBM,
    _roi_mask,
    calc_action_mask,
    calc_upsampling_loc,
    coverage_threshold_db as default_coverage_threshold_db,
    get_stats,
    load_map_normalized,
    map_size,
    normalized_pathgain_to_db,
)
from heuristic.optimizer_bridge import solve_target_layout
from surrogate_adapter import MODEL_CFG


_POLICY_CACHE: Dict[Tuple[str, str, str], "_TwoStageInitPolicy"] = {}
_ISOLATED_SURROGATE_CACHE: Dict[Tuple[str, str, str], "_IsolatedSurrogatePredictor"] = {}
SCORE_RERANK_TOP_N = 32
SCORE_RERANK_COVERAGE_WINDOW = 0.01
REACT_TWO_STAGE_RERANK_COVERAGE_TARGET = 0.92
REACT_TWO_STAGE_RERANK_SPECTRAL_EFFICIENCY_TARGET = 1.85


def load_module_state(module_state_path: str | Path) -> dict:
    path = Path(module_state_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Two-stage module state not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load two-stage module state from {path}. "
            "The file may be truncated, corrupted, or not a valid PyTorch checkpoint."
        ) from exc
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported module-state payload type: {type(payload)!r}")
    if any(str(key).startswith("module.") for key in payload):
        payload = {str(key).replace("module.", "", 1): value for key, value in payload.items()}
    return payload


def infer_module_state_observation_version(module_state_path: str | Path) -> str:
    state = load_module_state(module_state_path)
    first_conv = state.get("encoder.backbone.0.weight")
    if first_conv is not None:
        weight = np.asarray(first_conv)
        if weight.ndim == 4:
            in_channels = int(weight.shape[1])
            if in_channels == 1:
                return "single"
            if in_channels == 2:
                return "multi"
    for key, value in state.items():
        if str(key).endswith("encoder.actor_encoder.net.mlp.0.weight") or str(key).endswith("actor_encoder.net.mlp.0.weight"):
            weight = np.asarray(value)
            if weight.ndim == 2:
                input_dim = int(weight.shape[1])
                if input_dim == map_size * map_size:
                    return "single"
                if input_dim == 2 * map_size * map_size:
                    return "multi"
    return "single"


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def _enable_deterministic_torch_runtime() -> None:
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    if hasattr(torch.backends, "mkldnn"):
        try:
            torch.backends.mkldnn.enabled = False
        except Exception:
            pass


class _IsolatedSurrogatePredictor:
    def __init__(self, model_type: str, model_path: str | Path, device_name: str) -> None:
        cfg = MODEL_CFG.get(model_type) or {}
        model_py_rel = cfg.get("model_py_path")
        builder_name = cfg.get("builder_name")
        if not model_py_rel or not builder_name:
            raise KeyError(f"ReAct surrogate config missing model definition for {model_type}")

        model_py_path = (Path(__file__).resolve().parent / model_py_rel).resolve()
        checkpoint_path = Path(model_path).expanduser().resolve()
        spec = importlib.util.spec_from_file_location(f"isolated_{model_type}_model", model_py_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load model definition from {model_py_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        builder = getattr(module, builder_name, None)
        if builder is None:
            raise AttributeError(f"{builder_name} not found in {model_py_path}")

        self.device = _resolve_device(device_name)
        self.model = builder(output_stride=16)
        state = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(str(key).startswith("module.") for key in state):
            state = {str(key).replace("module.", "", 1): value for key, value in state.items()}
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        tensor = transforms.ToTensor()(inputs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = torch.clamp(self.model(tensor), 0.0, 1.0)
        return pred[0, 0].detach().cpu().numpy().astype(np.float32)


def _build_isolated_rmnet_fn(device_name: str):
    rmnet_cfg = MODEL_CFG.get("rmnet") or {}
    weights_rel = rmnet_cfg.get("weights_path")
    if not weights_rel:
        raise KeyError("ReAct surrogate config missing rmnet.weights_path")
    weights_path = (Path(__file__).resolve().parent / weights_rel).resolve()
    cache_key = ("rmnet", str(weights_path), str(_resolve_device(device_name)))
    predictor = _ISOLATED_SURROGATE_CACHE.get(cache_key)
    if predictor is None:
        predictor = _IsolatedSurrogatePredictor("rmnet", weights_path, device_name)
        _ISOLATED_SURROGATE_CACHE[cache_key] = predictor
    return predictor


def _resize_map(pixel_map: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    if pixel_map.shape == size:
        return pixel_map
    crop_u8 = (np.clip(pixel_map, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(crop_u8).resize(size, resample=Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


class _CNNActorEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, feature_dim: int = 256) -> None:
        super().__init__()
        self.in_channels, self.map_side = self._infer_channels_and_map_side(int(input_dim))
        channels = (32, 64, 128, 128)
        blocks: List[torch.nn.Module] = []
        in_channels = self.in_channels
        for out_channels in channels:
            blocks.extend(
                [
                    torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        self.backbone = torch.nn.Sequential(*blocks)
        self.actor_out_channels = in_channels

    @staticmethod
    def _infer_channels_and_map_side(input_dim: int) -> Tuple[int, int]:
        for channels in (1, 2, 3, 4):
            if input_dim % channels != 0:
                continue
            map_side = int(round((input_dim / channels) ** 0.5))
            if map_side * map_side * channels == input_dim:
                return channels, map_side
        raise ValueError(f"Cannot infer CNN input shape from flattened dim: {input_dim}")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        batch = observations.view(observations.shape[0], self.in_channels, self.map_side, self.map_side)
        return self.backbone(batch.float())


class _HeatmapPolicyHead(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        action_side = int(round(out_dim ** 0.5))
        if action_side * action_side != out_dim:
            raise ValueError(f"Action space size must be a perfect square, got {out_dim}")
        conv_hidden = max(int(hidden_dim // 2), 32)
        dropout = 0.15
        self.action_side = action_side
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout2d(p=dropout),
            torch.nn.Conv2d(hidden_dim, conv_hidden, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout2d(p=dropout),
            torch.nn.Upsample(size=(self.action_side, self.action_side), mode="bilinear", align_corners=False),
            torch.nn.Conv2d(conv_hidden, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).flatten(start_dim=1)


class _TwoStageInitPolicy(torch.nn.Module):
    def __init__(self, version: str) -> None:
        super().__init__()
        input_channels = 1 if version == "single" else 2
        input_dim = input_channels * map_size * map_size
        self.encoder = _CNNActorEncoder(input_dim=input_dim, feature_dim=256)
        self.pi = _HeatmapPolicyHead(
            in_channels=self.encoder.actor_out_channels,
            hidden_dim=256,
            out_dim=int(calc_action_mask(np.ones((map_size, map_size), dtype=np.float32)).size),
        )

    def forward(self, observations: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        logits = self.pi(self.encoder(observations))
        action_mask = torch.nan_to_num(action_mask.float(), nan=0.0, posinf=0.0, neginf=0.0)
        legal_mask = action_mask > 0.0
        has_legal_action = torch.any(legal_mask, dim=-1, keepdim=True)
        legal_mask = torch.where(has_legal_action, legal_mask, torch.ones_like(legal_mask, dtype=torch.bool))
        illegal_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
        return torch.where(legal_mask, logits, illegal_logits)


def _load_policy_model(module_state_path: str | Path, version: str, device_name: str) -> _TwoStageInitPolicy:
    resolved_path = str(Path(module_state_path).expanduser().resolve())
    resolved_version = infer_module_state_observation_version(resolved_path) if version == "auto" else version
    cache_key = (resolved_path, resolved_version, device_name)
    if cache_key in _POLICY_CACHE:
        return _POLICY_CACHE[cache_key]

    if resolved_version not in {"single", "multi"}:
        raise ValueError(f"Unsupported two-stage policy version: {resolved_version}")

    model = _TwoStageInitPolicy(version=resolved_version)
    current_state = model.state_dict()
    merged_state = dict(current_state)
    module_state = load_module_state(resolved_path)
    for key, value in module_state.items():
        if key not in current_state:
            continue
        if hasattr(current_state[key], "shape") and hasattr(value, "shape"):
            if tuple(current_state[key].shape) != tuple(value.shape):
                continue
        merged_state[key] = value
    model.load_state_dict(merged_state, strict=True)
    model.to(_resolve_device(device_name))
    model.eval()
    _POLICY_CACHE[cache_key] = model
    return model


def _mask_selected_actions(action_mask: np.ndarray, selected_actions: List[int]) -> np.ndarray:
    masked = np.array(action_mask, copy=True, dtype=np.float32)
    for action in selected_actions:
        if 0 <= int(action) < masked.size:
            masked[int(action)] = 0.0
    return masked


def _build_policy_observation(
    pixel_map: np.ndarray,
    tx_locs: List[Tuple[int, int]],
    selected_actions: List[int],
    version: str,
    coverage_threshold_db: float,
    rmnet_fn,
) -> Dict[str, np.ndarray]:
    action_mask = _mask_selected_actions(calc_action_mask(pixel_map).astype(np.float32), selected_actions)
    if version == "single":
        obs = np.clip(pixel_map.reshape(-1), 0.0, 1.0).astype(np.float32)
        return {"observations": obs, "action_mask": action_mask}
    if version != "multi":
        raise ValueError(f"Unsupported two-stage policy version: {version}")

    if not tx_locs:
        obs = np.tile(pixel_map.reshape(-1), 2).astype(np.float32)
    else:
        strongest_pathgain_norm, _, _, _ = get_stats(pixel_map, tx_locs, rmnet_fn)
        strongest_pathgain_db = normalized_pathgain_to_db(strongest_pathgain_norm)
        strongest_rx_power_dbm = TX_POWER_DBM + strongest_pathgain_db.astype(np.float64)
        covered = (strongest_rx_power_dbm >= coverage_threshold_db).astype(np.float32)
        obs = np.concatenate((pixel_map, covered), axis=None).astype(np.float32)
    return {"observations": np.clip(obs, 0.0, 1.0).astype(np.float32), "action_mask": action_mask}


def _compute_policy_action(model: _TwoStageInitPolicy, observation: Dict[str, np.ndarray]) -> int:
    return int(torch.argmax(_compute_policy_logits(model, observation)).item())


def _compute_policy_logits(model: _TwoStageInitPolicy, observation: Dict[str, np.ndarray]) -> torch.Tensor:
    device = next(model.parameters()).device
    observations = torch.from_numpy(np.asarray(observation["observations"], dtype=np.float32)).unsqueeze(0).to(device)
    action_mask = torch.from_numpy(np.asarray(observation["action_mask"], dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(observations, action_mask)[0]


def _score_candidate_layout(
    pixel_map: np.ndarray,
    tx_locs: List[Tuple[int, int]],
    coverage_threshold_db: float,
    rmnet_fn,
) -> dict[str, float]:
    strongest_pathgain_norm, coverage_reward, spectral_efficiency_reward, _ = get_stats(pixel_map, tx_locs, rmnet_fn)
    strongest_pathgain_db = normalized_pathgain_to_db(strongest_pathgain_norm)
    roi_mask = _roi_mask(pixel_map)
    roi_rx_power_dbm = TX_POWER_DBM + np.asarray(strongest_pathgain_db, dtype=np.float64)[roi_mask]
    if roi_rx_power_dbm.size == 0:
        rss_margin = -1.0
    else:
        rss_margin = float(
            np.mean(
                np.tanh(
                    (roi_rx_power_dbm - float(coverage_threshold_db)) / max(AUTOBS_DEFAULT_SCORE_MARGIN_SCALE_DB, 1e-6)
                )
            )
        )
    score_components = compute_autobs_score_components(
        coverage=float(coverage_reward),
        spectral_efficiency=float(spectral_efficiency_reward),
        rss_margin=rss_margin,
        coverage_target=REACT_TWO_STAGE_RERANK_COVERAGE_TARGET,
        spectral_efficiency_target=REACT_TWO_STAGE_RERANK_SPECTRAL_EFFICIENCY_TARGET,
    )
    return {
        "coverage": float(coverage_reward),
        "spectral_efficiency": float(spectral_efficiency_reward),
        "score": float(score_components["score"]),
    }


def _compute_policy_action_with_score_rerank(
    model: _TwoStageInitPolicy,
    observation: Dict[str, np.ndarray],
    pixel_map: np.ndarray,
    tx_locs: List[Tuple[int, int]],
    coverage_threshold_db: float,
    rmnet_fn,
    rerank_top_n: int = SCORE_RERANK_TOP_N,
) -> int:
    logits = _compute_policy_logits(model, observation)
    action_mask = np.asarray(observation["action_mask"], dtype=np.float32)
    legal_actions = np.flatnonzero(action_mask > 0.0)
    if legal_actions.size == 0:
        raise RuntimeError("Two-stage policy observation contains no legal actions")
    if legal_actions.size == 1 or rerank_top_n <= 1:
        return int(legal_actions[int(np.argmax(logits[legal_actions].detach().cpu().numpy()))])

    legal_logits = logits[legal_actions].detach().cpu().numpy()
    candidate_count = min(int(rerank_top_n), int(legal_actions.size))
    top_order = np.argsort(-legal_logits)[:candidate_count]
    candidate_actions = legal_actions[top_order]

    best_action = int(candidate_actions[0])
    candidate_summaries = []
    for action in candidate_actions.astype(int).tolist():
        candidate_tx_locs = tx_locs + [calc_upsampling_loc(action, pixel_map)]
        candidate_metrics = _score_candidate_layout(
            pixel_map=pixel_map,
            tx_locs=candidate_tx_locs,
            coverage_threshold_db=coverage_threshold_db,
            rmnet_fn=rmnet_fn,
        )
        candidate_summaries.append((action, candidate_metrics, float(logits[action].item())))

    best_key = (-float("inf"), -float("inf"), -float("inf"), -float("inf"))
    for action, candidate_metrics, logit_value in candidate_summaries:
        candidate_key = (
            float(candidate_metrics["spectral_efficiency"]),
            float(candidate_metrics["coverage"]),
            float(candidate_metrics["score"]),
            logit_value,
        )
        if candidate_key > best_key:
            best_key = candidate_key
            best_action = int(action)
    return best_action


def init_locs_from_two_stage_policy(
    city_map_path: str,
    module_state_path: str,
    version: str = "auto",
    top_k: int = 1,
    device_name: str = "auto",
    coverage_threshold_db: float = default_coverage_threshold_db,
) -> List[Tuple[int, int]]:
    _enable_deterministic_torch_runtime()
    resolved_version = infer_module_state_observation_version(module_state_path) if version == "auto" else version
    model = _load_policy_model(module_state_path, resolved_version, device_name)
    rmnet_fn = _build_isolated_rmnet_fn(device_name)
    pixel_map = _resize_map(load_map_normalized(city_map_path), (map_size, map_size))

    selected_actions: List[int] = []
    tx_locs: List[Tuple[int, int]] = []
    max_steps = max(0, int(top_k))
    for _ in range(max_steps):
        observation = _build_policy_observation(
            pixel_map=pixel_map,
            tx_locs=tx_locs,
            selected_actions=selected_actions,
            version=resolved_version,
            coverage_threshold_db=coverage_threshold_db,
            rmnet_fn=rmnet_fn,
        )
        if float(np.sum(observation["action_mask"])) <= 0.0:
            break
        action = _compute_policy_action_with_score_rerank(
            model,
            observation,
            pixel_map=pixel_map,
            tx_locs=tx_locs,
            coverage_threshold_db=coverage_threshold_db,
            rmnet_fn=rmnet_fn,
        )
        if observation["action_mask"][action] <= 0.0:
            raise RuntimeError(f"Two-stage policy selected an illegal or repeated action: {action}")
        selected_actions.append(action)
        tx_locs.append(calc_upsampling_loc(action, pixel_map))
    return tx_locs


def init_locs_from_heuristic_layout(
    city_map_path: str,
    goal: dict,
    constraints: dict,
    planner_name: str,
    eval_model: str,
    eval_model_path: str,
    eval_device: str,
    output_dir: str | Path,
    top_k: int = 1,
    max_evals: int = 100,
    candidate_stride: int = 12,
    candidate_limit: int = 256,
) -> List[Tuple[int, int]]:
    requested_sites = max(1, int(top_k))
    init_constraints = dict(constraints or {})
    init_constraints["site_limit"] = requested_sites
    init_constraints["site_exact"] = requested_sites
    positions, _ = solve_target_layout(
        planner_name=planner_name,
        height_map_path=city_map_path,
        goal=goal,
        constraints=init_constraints,
        eval_model=eval_model,
        eval_model_path=eval_model_path,
        output_dir=Path(output_dir),
        fallback_k=requested_sites,
        max_evals=max_evals,
        candidate_stride=candidate_stride,
        candidate_limit=candidate_limit,
        device=eval_device,
    )
    return [(int(row), int(col)) for row, col in positions[:requested_sites]]
