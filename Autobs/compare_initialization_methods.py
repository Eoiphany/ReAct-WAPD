"""注释
命令示例:
conda activate autobs
cd autodl-tmp/code/

python Autobs/compare_initialization_methods.py \
  --maps-file /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/data/maps_test_paths_localrun.txt \
  --methods random_init pretrain_init bandit_init run_sa run_greedy run_ga run_pso \
  --k-max 1 \
  --device mps \
  --model-path /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/checkpoints/rmnet_radiomap3dseer.pt \
  --pretrain-module-state /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/pretrained_policy/best_module_state.pt \
  --bandit-module-state /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/bandit_policy/best_module_state.pt \
  --network-type rmnet \
  --output-dir Autobs/outputs/init_comparev4

   /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/runs/pmnet_radiomap3dseer/16_0.0001_0.5_10/pmnet_radiomap3dseer_best.pt
   /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/models/PMNet.pt
   
   /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/checkpoints


参数说明:
- --maps-file: 地图路径清单文件；默认读取 `ReAct/data/maps_test_paths_localrun.txt`。
- --methods: 要对比的方法集合。推荐使用 `random_init`、`ppo_init`、`run_sa`、`run_greedy`、`run_ga`、`run_pso`、
  `run_candidate_enumeration`、`run_exhaustive_search`；旧名字 `run_bruteforce`、`run_full_enumeration` 仍兼容并会自动归一化。
- --map-limit: 可选，仅跑前 N 张图，便于 smoke test。
- --k-max: 每张图部署的站点数。
- --model-path: 统一评估 surrogate 权重路径，同时传给启发式脚本。
- --network-type: surrogate 模型类型，支持 `pmnet`、`pmnet_v3`、`rmnet`、`rmnet_v3`。v3是输出维度不一样，我们使用普通版本即可。
- --ppo-checkpoint: PPO checkpoint 路径；当 `methods` 包含 `ppo_init` 时必需。
- --ppo-version: `auto/single/multi`。`auto` 会在 `k-max=1` 时走单站点观测，否则走多站点观测。
- --device: 推理设备，`auto/cpu/cuda/mps`。
- --output-dir: 输出目录；会保存逐图记录、按方法汇总均值和失败日志。
- --coverage-target / --spectral-efficiency-target: 与现有启发式脚本一致的目标值。
- --max-evals: 传给启发式 `run_*` 脚本的评估预算。
- --d-min / --repair-max-tries / --w1 / --w2 / --coverage-threshold-db / --noise-coefficient-db:
  与现有启发式脚本一致的约束和评分参数。
- 其余 `--sa-*`、`--greedy-*`、`--ga-*`、`--pso-*`、`--bruteforce-*` 参数会原样传给对应 `run_*` 脚本。

脚本逻辑说明:
- 先读取 `maps_test_paths_localrun.txt` 中的全部地图路径，并自动修正旧前缀 `/Users/epiphanyer/coding/...` 到当前工作区
  `/Users/epiphanyer/Desktop/coding/...`。
- 对 `run_*` 方法，脚本直接调用 `paper_experiment/Heuristic` 里的原始脚本，读取各自输出的 `best_metrics.json`，
  避免在 `Autobs` 内部再复制一版启发式实现。
- 对 `random_init`，脚本在动作网格的合法位置中均匀随机采样 `k-max` 个不重复动作，并用 `Autobs.env.utils.get_stats`
  计算覆盖率、平均频谱效率、平均信道容量、冗余率和 score。
- 对 `ppo_init`，脚本复用现有 PPO checkpoint 加载逻辑，按单站点或多站点观测顺序依次选点，再用同一套指标函数统一评估。
- 最终输出 `all_records.csv/json`、`summary_by_method.csv/json` 和 `failures.json`，其中 summary 给出每种方法在整份测试图上的平均覆盖率、
  平均信道容量、平均频谱效率、平均冗余率和平均 score。
- 现在脚本是“随机和 PPO 只做汇总，不做单方法目录落盘；启发式才有单独文件夹”。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    _CURRENT_FILE = Path(__file__).resolve()
    _PACKAGE_PARENT = _CURRENT_FILE.parent.parent
    if str(_PACKAGE_PARENT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_PARENT))

from Autobs.env.utils import (
    calc_action_mask,
    calc_upsampling_loc,
    CHANNEL_BANDWIDTH_HZ,
    DEFAULT_COVERAGE_TARGET,
    DEFAULT_COVERAGE_THRESHOLD_DB,
    DEFAULT_NOISE_COEFFICIENT_DB,
    DEFAULT_SPECTRAL_EFFICIENCY_TARGET,
    DEFAULT_W1,
    DEFAULT_W2,
    DEFAULT_SCORE_MARGIN_SCALE_DB,
    TX_POWER_DBM,
    THERMAL_NOISE_DENSITY_DBM_PER_HZ,
    BITS_PER_MEGABIT,
    compute_score_components,
    get_site_pathgain_maps,
    load_map_normalized,
)
from Autobs.paths import CHECKPOINT_DIR, DEFAULT_RMNET_WEIGHTS, PACKAGE_ROOT, PROJECT_ROOT
from Autobs.pmnet_adapter import infer_pmnet
from Autobs.pretrain.pretrain_policy import build_policy_module, forward_masked_logits
from Autobs.run_checkpoint_rmnet_viz import compute_checkpoint_action, load_checkpoint_agent, load_pickle_state
from Autobs.train_ppo import apply_module_state, load_module_state

WORKSPACE_ROOT = PROJECT_ROOT.parent
HEURISTIC_DIR = PROJECT_ROOT / "Heuristic"
DEFAULT_MAPS_FILE = WORKSPACE_ROOT / "ReAct" / "data" / "maps_test_paths_localrun.txt"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs" / "init_compare"
METHOD_CHOICES = (
    "random_init",
    "ppo_init",
    "pretrain_init",
    "bandit_init",
    "run_sa",
    "run_greedy",
    "run_ga",
    "run_pso",
    "run_candidate_enumeration",
    "run_exhaustive_search",
    "run_bruteforce",
    "run_full_enumeration",
)
DEFAULT_METHODS = ("random_init", "ppo_init", "run_sa", "run_greedy", "run_ga", "run_pso")
NETWORK_CHOICES = ("pmnet", "pmnet_v3", "rmnet", "rmnet_v3")
ROI_COUNT_THRESHOLDS = (1, 2)
OUTPUT_HEIGHT_MIN_M = 9.6
OUTPUT_HEIGHT_MAX_M = 22.8
OUTPUT_MIN_BUILDING_GRAY = 32
METHOD_ALIASES = {
    "run_bruteforce": "run_candidate_enumeration",
    "run_full_enumeration": "run_exhaustive_search",
}
MODULE_RERANK_TOP_N = 8
MODULE_RERANK_COVERAGE_WINDOW = 0.01
HEURISTIC_SCRIPT_BY_METHOD = {
    "run_sa": HEURISTIC_DIR / "run_sa.py",
    "run_greedy": HEURISTIC_DIR / "run_greedy.py",
    "run_ga": HEURISTIC_DIR / "run_ga.py",
    "run_pso": HEURISTIC_DIR / "run_pso.py",
    "run_candidate_enumeration": HEURISTIC_DIR / "run_candidate_enumeration.py",
    "run_exhaustive_search": HEURISTIC_DIR / "run_exhaustive_search.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare heuristic / random / PPO initialization methods on a map list.")
    parser.add_argument("--maps-file", default=str(DEFAULT_MAPS_FILE), type=str)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=METHOD_CHOICES)
    parser.add_argument("--map-limit", type=int)
    parser.add_argument("--k-max", default=2, type=int)
    parser.add_argument("--model-path", default=str(DEFAULT_RMNET_WEIGHTS), type=str)
    parser.add_argument("--network-type", default="rmnet", choices=NETWORK_CHOICES)
    parser.add_argument("--ppo-checkpoint", default=str(CHECKPOINT_DIR), type=str)
    parser.add_argument("--ppo-version", default="auto", choices=["auto", "single", "multi"])
    parser.add_argument("--pretrain-module-state", default="", type=str)
    parser.add_argument("--bandit-module-state", default="", type=str)
    parser.add_argument("--policy-version", default="auto", choices=["auto", "single", "multi"])
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), type=str)
    parser.add_argument("--coverage-target", default=DEFAULT_COVERAGE_TARGET, type=float)
    parser.add_argument("--spectral-efficiency-target", default=DEFAULT_SPECTRAL_EFFICIENCY_TARGET, type=float)
    parser.add_argument("--max-evals", default=200, type=int)
    parser.add_argument("--d-min", default=12.0, type=float)
    parser.add_argument("--repair-max-tries", default=100, type=int)
    parser.add_argument("--w1", default=DEFAULT_W1, type=float)
    parser.add_argument("--w2", default=DEFAULT_W2, type=float)
    parser.add_argument("--coverage-threshold-db", default=DEFAULT_COVERAGE_THRESHOLD_DB, type=float)
    parser.add_argument("--noise-coefficient-db", default=DEFAULT_NOISE_COEFFICIENT_DB, type=float)
    parser.add_argument("--sa-initial-temp", default=1.0, type=float)
    parser.add_argument("--sa-cooling-rate", default=0.995, type=float)
    parser.add_argument("--sa-gaussian-sigma", default=6.0, type=float)
    parser.add_argument("--greedy-candidate-stride", default=8, type=int)
    parser.add_argument("--greedy-candidate-limit", default=5000, type=int)
    parser.add_argument("--ga-population-size", default=24, type=int)
    parser.add_argument("--ga-elite-size", default=4, type=int)
    parser.add_argument("--ga-tournament-size", default=3, type=int)
    parser.add_argument("--ga-mutation-rate", default=0.3, type=float)
    parser.add_argument("--ga-gaussian-sigma", default=6.0, type=float)
    parser.add_argument("--pso-swarm-size", default=20, type=int)
    parser.add_argument("--pso-inertia", default=0.7, type=float)
    parser.add_argument("--pso-c1", default=1.4, type=float)
    parser.add_argument("--pso-c2", default=1.4, type=float)
    parser.add_argument("--pso-velocity-clamp", default=8.0, type=float)
    parser.add_argument("--bruteforce-candidate-stride", default=16, type=int)
    parser.add_argument("--bruteforce-candidate-limit", default=80, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def normalize_method_name(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def resolve_map_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    raw_text = str(candidate)
    replacements = []
    old_workspace_prefix = "/Users/epiphanyer/coding/"
    if raw_text.startswith(old_workspace_prefix):
        replacements.append(WORKSPACE_ROOT / raw_text.removeprefix(old_workspace_prefix))
    if not candidate.is_absolute():
        replacements.append(WORKSPACE_ROOT / raw_text)
        replacements.append(PROJECT_ROOT / raw_text)

    for alt in replacements:
        if alt.exists():
            return alt.resolve()

    raise FileNotFoundError(f"Map path not found: {raw_path}")


def load_map_paths(paths_file: str, limit: int | None = None) -> list[Path]:
    file_path = Path(paths_file).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    resolved = [resolve_map_path(line) for line in lines]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--map-limit must be positive")
        resolved = resolved[:limit]
    if not resolved:
        raise ValueError("No valid map paths were loaded")
    return resolved


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def _numpy_image_to_tensor(array: np.ndarray) -> torch.Tensor:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        tensor = torch.from_numpy(arr).unsqueeze(0)
    elif arr.ndim == 3:
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    return tensor


def gray_to_height_m(gray_value: int | float) -> float:
    gray = int(round(float(gray_value)))
    if gray <= 0:
        return 0.0
    clipped = min(max(gray, OUTPUT_MIN_BUILDING_GRAY), 255)
    scaled = (clipped - OUTPUT_MIN_BUILDING_GRAY) / (255 - OUTPUT_MIN_BUILDING_GRAY)
    return float(OUTPUT_HEIGHT_MIN_M + scaled * (OUTPUT_HEIGHT_MAX_M - OUTPUT_HEIGHT_MIN_M))


def add_position_heights(
    pixel_map: np.ndarray,
    positions_xy: list[list[int]] | list[tuple[int, int]] | None,
) -> list[list[float]] | None:
    if not positions_xy:
        return None

    positions_xyz: list[list[float]] = []
    height, width = pixel_map.shape
    for x, y in positions_xy:
        row = int(np.clip(np.rint(float(y)), 0, height - 1))
        col = int(np.clip(np.rint(float(x)), 0, width - 1))
        gray_value = int(round(float(np.clip(pixel_map[row, col], 0.0, 1.0) * 255.0)))
        positions_xyz.append([col, row, round(gray_to_height_m(gray_value), 4)])
    return positions_xyz


def _init_surrogate_model(network_type: str) -> torch.nn.Module:
    if network_type == "pmnet":
        from Autobs.models.PMNet import PMNet

        return PMNet(n_blocks=[3, 3, 27, 3], atrous_rates=[6, 12, 18], multi_grids=[1, 2, 4], output_stride=16)
    if network_type == "pmnet_v3":
        from Autobs.models.PMNet import PMNet

        return PMNet(n_blocks=[3, 3, 27, 3], atrous_rates=[6, 12, 18], multi_grids=[1, 2, 4], output_stride=8)
    if network_type == "rmnet":
        from Autobs.models.RMNet import RMNet

        return RMNet(n_blocks=[3, 3, 27, 3], atrous_rates=[6, 12, 18], multi_grids=[1, 2, 4], output_stride=16)
    if network_type == "rmnet_v3":
        from Autobs.models.RMNet import RMNet

        return RMNet(n_blocks=[3, 3, 27, 3], atrous_rates=[6, 12, 18], multi_grids=[1, 2, 4], output_stride=8)
    raise ValueError(f"Unsupported network_type: {network_type}")


class LocalSurrogatePredictor:
    def __init__(self, model_path: str, network_type: str, device_name: str) -> None:
        self.device = _resolve_device(device_name)
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.network_type = network_type
        self.model = _init_surrogate_model(network_type)
        state = torch.load(self.model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(key.startswith("module.") for key in state):
            state = {key.replace("module.", "", 1): value for key, value in state.items()}
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        tensor = _numpy_image_to_tensor(inputs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(tensor)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
            pred = torch.clamp(pred, 0.0, 1.0)
        return pred.detach().cpu().squeeze().numpy().astype(np.float32)


def build_random_layout(pixel_map: np.ndarray, k_max: int, rng: np.random.Generator) -> tuple[list[int], list[tuple[int, int]]]:
    action_mask = calc_action_mask(pixel_map).astype(np.float32)
    legal_actions = np.flatnonzero(action_mask > 0.0)
    if legal_actions.size < k_max:
        raise ValueError(f"Not enough legal actions for k_max={k_max}: only {legal_actions.size} available")
    sampled_actions = rng.choice(legal_actions, size=k_max, replace=False).astype(int).tolist()
    tx_locs = [calc_upsampling_loc(action, pixel_map) for action in sampled_actions]
    return sampled_actions, tx_locs


def _mask_selected_actions(action_mask: np.ndarray, selected_actions: list[int]) -> np.ndarray:
    masked = np.array(action_mask, copy=True, dtype=np.float32)
    for action in selected_actions:
        if 0 <= action < masked.size:
            masked[action] = 0.0
    return masked


def build_ppo_observation(
    pixel_map: np.ndarray,
    tx_locs: list[tuple[int, int]],
    selected_actions: list[int],
    version: str,
    coverage_threshold_db: float,
) -> dict[str, np.ndarray]:
    action_mask = _mask_selected_actions(calc_action_mask(pixel_map).astype(np.float32), selected_actions)
    if version == "single":
        obs = np.clip(pixel_map.reshape(-1), 0.0, 1.0).astype(np.float32)
        return {"observations": obs, "action_mask": action_mask}
    if version != "multi":
        raise ValueError(f"Unsupported PPO version: {version}")

    if not tx_locs:
        obs = np.tile(pixel_map.reshape(-1), 2).astype(np.float32)
    else:
        site_pathgain_db = get_site_pathgain_maps(pixel_map, tx_locs, pmnet=infer_pmnet)
        strongest_pathgain_db = np.max(site_pathgain_db, axis=0).astype(np.float32)
        strongest_rx_power_dbm = TX_POWER_DBM + strongest_pathgain_db
        covered = (strongest_rx_power_dbm >= coverage_threshold_db).astype(np.float32)
        obs = np.concatenate((pixel_map, covered), axis=None).astype(np.float32)
    return {"observations": np.clip(obs, 0.0, 1.0).astype(np.float32), "action_mask": action_mask}


def select_ppo_layout(
    *,
    pixel_map: np.ndarray,
    checkpoint_path: str,
    k_max: int,
    ppo_version: str,
    coverage_threshold_db: float,
) -> tuple[list[int], list[tuple[int, int]]]:
    version = ppo_version
    if version == "auto":
        version = infer_ppo_observation_version(checkpoint_path)

    agent = load_checkpoint_agent(checkpoint_path)
    selected_actions: list[int] = []
    tx_locs: list[tuple[int, int]] = []
    try:
        for _ in range(k_max):
            observation = build_ppo_observation(
                pixel_map=pixel_map,
                tx_locs=tx_locs,
                selected_actions=selected_actions,
                version=version,
                coverage_threshold_db=coverage_threshold_db,
            )
            if float(np.sum(observation["action_mask"])) <= 0.0:
                raise RuntimeError("PPO observation action mask contains no legal actions")
            action = int(compute_checkpoint_action(agent, observation, explore=False))
            if observation["action_mask"][action] <= 0.0:
                raise RuntimeError(f"PPO selected an illegal or repeated action: {action}")
            selected_actions.append(action)
            tx_locs.append(calc_upsampling_loc(action, pixel_map))
    finally:
        stop = getattr(agent, "stop", None)
        if callable(stop):
            stop()
    return selected_actions, tx_locs


def infer_ppo_observation_version(checkpoint_path: str | Path) -> str:
    checkpoint_dir = Path(checkpoint_path).expanduser().resolve()
    module_state_path = checkpoint_dir / "learner_group" / "learner" / "rl_module" / "default_policy" / "module_state.pkl"
    if not module_state_path.exists():
        return "multi"

    with module_state_path.open("rb") as handle:
        state = load_pickle_state(handle)
    if not isinstance(state, dict):
        return "multi"

    for key, value in state.items():
        if key.endswith("encoder.actor_encoder.net.mlp.0.weight") or key.endswith("actor_encoder.net.mlp.0.weight"):
            weight = np.asarray(value)
            if weight.ndim == 2:
                input_dim = int(weight.shape[1])
                if input_dim == 256 * 256:
                    return "single"
                if input_dim == 2 * 256 * 256:
                    return "multi"
    return "multi"


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
        if key.endswith("encoder.actor_encoder.net.mlp.0.weight") or key.endswith("actor_encoder.net.mlp.0.weight"):
            weight = np.asarray(value)
            if weight.ndim == 2:
                input_dim = int(weight.shape[1])
                if input_dim == 256 * 256:
                    return "single"
                if input_dim == 2 * 256 * 256:
                    return "multi"
    return "single"


def _make_module_agent(module):
    class _ModuleAgent:
        def __init__(self, inner_module):
            self._inner_module = inner_module

        def get_module(self):
            return self._inner_module

    return _ModuleAgent(module)


def load_module_policy(module_state_path: str | Path, version: str, device_name: str):
    device = _resolve_device(device_name)
    module = build_policy_module(version, device=device)
    module_state = load_module_state(module_state_path)
    apply_module_state(_make_module_agent(module), module_state)
    module.to(device)
    module.eval()
    return module


def compute_module_action(module, observation: dict[str, np.ndarray], explore: bool = False) -> int:
    logits = compute_module_logits(module, observation)
    if explore:
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())
    return int(torch.argmax(logits).item())


def compute_module_logits(module, observation: dict[str, np.ndarray]) -> torch.Tensor:
    device = next(module.parameters()).device
    observations = torch.from_numpy(np.asarray(observation["observations"], dtype=np.float32)).unsqueeze(0).to(device)
    action_mask = torch.from_numpy(np.asarray(observation["action_mask"], dtype=np.float32)).unsqueeze(0).to(device)
    return forward_masked_logits(module, observations, action_mask)[0]


def rerank_module_action(
    *,
    module,
    observation: dict[str, np.ndarray],
    pixel_map: np.ndarray,
    tx_locs: list[tuple[int, int]],
    predictor: "LocalSurrogatePredictor",
    coverage_target: float,
    spectral_efficiency_target: float,
    w1: float,
    w2: float,
    coverage_threshold_db: float,
    noise_coefficient_db: float,
    rerank_top_n: int = MODULE_RERANK_TOP_N,
    coverage_window: float = MODULE_RERANK_COVERAGE_WINDOW,
) -> int:
    logits = compute_module_logits(module, observation)
    action_mask = np.asarray(observation["action_mask"], dtype=np.float32)
    legal_actions = np.flatnonzero(action_mask > 0.0)
    if legal_actions.size == 0:
        raise RuntimeError("Module-state policy observation action mask contains no legal actions")
    if legal_actions.size == 1 or rerank_top_n <= 1:
        return int(legal_actions[0])

    legal_logits = logits[legal_actions].detach().cpu().numpy()
    candidate_count = min(int(rerank_top_n), int(legal_actions.size))
    top_order = np.argsort(-legal_logits)[:candidate_count]
    candidate_actions = legal_actions[top_order]

    ranked_candidates: list[tuple[tuple[float, ...], int]] = []
    best_coverage = -float("inf")
    candidate_metrics: list[tuple[int, dict[str, float], float]] = []
    for action in candidate_actions.astype(int).tolist():
        candidate_tx_locs = tx_locs + [calc_upsampling_loc(action, pixel_map)]
        metrics = evaluate_layout(
            pixel_map,
            candidate_tx_locs,
            predictor,
            coverage_target=coverage_target,
            spectral_efficiency_target=spectral_efficiency_target,
            w1=w1,
            w2=w2,
            coverage_threshold_db=coverage_threshold_db,
            noise_coefficient_db=noise_coefficient_db,
        )
        best_coverage = max(best_coverage, float(metrics["coverage"]))
        candidate_metrics.append((action, metrics, float(logits[action].item())))

    for action, metrics, logit_value in candidate_metrics:
        near_best_coverage = 1.0 if float(metrics["coverage"]) >= (best_coverage - coverage_window) else 0.0
        key = (
            near_best_coverage,
            float(metrics["coverage"]),
            float(metrics["spectral_efficiency"]),
            float(metrics["score"]),
            logit_value,
        )
        ranked_candidates.append((key, int(action)))
    ranked_candidates.sort(reverse=True)
    return int(ranked_candidates[0][1])


def select_module_state_layout(
    *,
    pixel_map: np.ndarray,
    module_state_path: str,
    k_max: int,
    policy_version: str,
    coverage_threshold_db: float,
    device_name: str,
    predictor: "LocalSurrogatePredictor",
    coverage_target: float,
    spectral_efficiency_target: float,
    w1: float,
    w2: float,
    noise_coefficient_db: float,
) -> tuple[list[int], list[tuple[int, int]]]:
    version = policy_version
    if version == "auto":
        version = infer_module_state_observation_version(module_state_path)

    module = load_module_policy(module_state_path, version=version, device_name=device_name)
    selected_actions: list[int] = []
    tx_locs: list[tuple[int, int]] = []
    for _ in range(k_max):
        observation = build_ppo_observation(
            pixel_map=pixel_map,
            tx_locs=tx_locs,
            selected_actions=selected_actions,
            version=version,
            coverage_threshold_db=coverage_threshold_db,
        )
        if float(np.sum(observation["action_mask"])) <= 0.0:
            raise RuntimeError("Module-state policy observation action mask contains no legal actions")
        action = rerank_module_action(
            module=module,
            observation=observation,
            pixel_map=pixel_map,
            tx_locs=tx_locs,
            predictor=predictor,
            coverage_target=coverage_target,
            spectral_efficiency_target=spectral_efficiency_target,
            w1=w1,
            w2=w2,
            coverage_threshold_db=coverage_threshold_db,
            noise_coefficient_db=noise_coefficient_db,
        )
        if observation["action_mask"][action] <= 0.0:
            raise RuntimeError(f"Module-state policy selected an illegal or repeated action: {action}")
        selected_actions.append(action)
        tx_locs.append(calc_upsampling_loc(action, pixel_map))
    return selected_actions, tx_locs


def evaluate_layout(
    pixel_map: np.ndarray,
    tx_locs: list[tuple[int, int]],
    predictor: LocalSurrogatePredictor,
    *,
    coverage_target: float,
    spectral_efficiency_target: float,
    w1: float,
    w2: float,
    coverage_threshold_db: float,
    noise_coefficient_db: float,
) -> dict[str, float]:
    roi_mask = build_roi_mask(pixel_map)
    if not tx_locs:
        strongest_pathgain_db = np.full_like(pixel_map, -162.0, dtype=np.float32)
        summed_rx_power_mw = np.zeros_like(pixel_map, dtype=np.float32)
        covered_site_counts = np.zeros_like(pixel_map, dtype=np.int16)
    else:
        site_pathgain_db = get_site_pathgain_maps(pixel_map, tx_locs, pmnet=predictor)
        strongest_pathgain_db = np.max(site_pathgain_db, axis=0).astype(np.float32)
        rx_power_dbm = TX_POWER_DBM + site_pathgain_db.astype(np.float64)
        summed_rx_power_mw = np.sum(np.power(10.0, rx_power_dbm / 10.0), axis=0).astype(np.float32)
        covered_site_counts = np.count_nonzero(rx_power_dbm >= coverage_threshold_db, axis=0).astype(np.int16)

    return evaluate_radio_metrics_with_redundancy(
        pathgain_db=strongest_pathgain_db,
        summed_rx_power_mw=summed_rx_power_mw,
        roi_mask=roi_mask,
        coverage_target=coverage_target,
        spectral_efficiency_target=spectral_efficiency_target,
        w1=w1,
        w2=w2,
        coverage_threshold_db=coverage_threshold_db,
        noise_coefficient_db=noise_coefficient_db,
        covered_site_counts=covered_site_counts,
    )


def build_roi_mask(pixel_map: np.ndarray) -> np.ndarray:
    mask = pixel_map <= 0.01
    if not mask.any():
        mask = np.ones_like(pixel_map, dtype=bool)
    return mask


def compute_total_noise_power_dbm(noise_coefficient_db: float) -> float:
    return THERMAL_NOISE_DENSITY_DBM_PER_HZ + 10.0 * np.log10(CHANNEL_BANDWIDTH_HZ) + noise_coefficient_db


def compute_total_noise_power_mw(noise_coefficient_db: float) -> float:
    return float(np.power(10.0, compute_total_noise_power_dbm(noise_coefficient_db) / 10.0))


def evaluate_radio_metrics_with_redundancy(
    *,
    pathgain_db: np.ndarray,
    summed_rx_power_mw: np.ndarray,
    roi_mask: np.ndarray,
    coverage_target: float,
    spectral_efficiency_target: float,
    w1: float,
    w2: float,
    coverage_threshold_db: float,
    noise_coefficient_db: float,
    covered_site_counts: np.ndarray,
) -> dict[str, float]:
    strongest_pathgain_db = pathgain_db[roi_mask].astype(np.float64)
    total_rx_power_mw = summed_rx_power_mw[roi_mask].astype(np.float64)
    strongest_rx_power_dbm = TX_POWER_DBM + strongest_pathgain_db
    strongest_rx_power_mw = np.power(10.0, strongest_rx_power_dbm / 10.0)
    coverage = float(np.mean(strongest_rx_power_dbm >= coverage_threshold_db))

    noise_power_mw = compute_total_noise_power_mw(noise_coefficient_db)
    interference_power_mw = np.maximum(total_rx_power_mw - strongest_rx_power_mw, 0.0)
    sinr_linear = strongest_rx_power_mw / np.maximum(interference_power_mw + noise_power_mw, 1e-30)
    spectral_efficiency = float(np.mean(np.log2(1.0 + sinr_linear)))
    channel_capacity_mbps = float(CHANNEL_BANDWIDTH_HZ * spectral_efficiency / BITS_PER_MEGABIT)

    roi_covered_site_counts = covered_site_counts[roi_mask].astype(np.int32)
    covered_pixels = roi_covered_site_counts >= ROI_COUNT_THRESHOLDS[0]
    covered_pixel_count = int(np.count_nonzero(covered_pixels))
    redundancy_rate = 0.0
    if covered_pixel_count > 0:
        redundant_pixel_count = int(np.count_nonzero(roi_covered_site_counts >= ROI_COUNT_THRESHOLDS[1]))
        redundancy_rate = redundant_pixel_count / covered_pixel_count

    rss_margin = float(
        np.mean(
            np.tanh(
                (strongest_rx_power_dbm - coverage_threshold_db) / max(DEFAULT_SCORE_MARGIN_SCALE_DB, 1e-6)
            )
        )
    )
    score_components = compute_score_components(
        coverage=coverage,
        spectral_efficiency=spectral_efficiency,
        rss_margin=rss_margin,
        coverage_target=coverage_target,
        spectral_efficiency_target=spectral_efficiency_target,
        w1=w1,
        w2=w2,
    )
    return {
        "coverage": coverage,
        "spectral_efficiency": spectral_efficiency,
        "channel_capacity_mbps": channel_capacity_mbps,
        "redundancy_rate": redundancy_rate,
        **score_components,
    }


def run_random_init(
    map_path: Path,
    predictor: LocalSurrogatePredictor,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, Any]:
    pixel_map = load_map_normalized(map_path)
    selected_actions, tx_locs = build_random_layout(pixel_map, args.k_max, rng)
    metrics = evaluate_layout(
        pixel_map,
        tx_locs,
        predictor,
        coverage_target=args.coverage_target,
        spectral_efficiency_target=args.spectral_efficiency_target,
        w1=args.w1,
        w2=args.w2,
        coverage_threshold_db=args.coverage_threshold_db,
        noise_coefficient_db=args.noise_coefficient_db,
    )
    return {
        "method": "random_init",
        "selected_actions": selected_actions,
        "positions_xy": [[int(col), int(row)] for row, col in tx_locs],
        **metrics,
    }


def run_ppo_init(
    map_path: Path,
    predictor: LocalSurrogatePredictor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    pixel_map = load_map_normalized(map_path)
    selected_actions, tx_locs = select_ppo_layout(
        pixel_map=pixel_map,
        checkpoint_path=args.ppo_checkpoint,
        k_max=args.k_max,
        ppo_version=args.ppo_version,
        coverage_threshold_db=args.coverage_threshold_db,
    )
    metrics = evaluate_layout(
        pixel_map,
        tx_locs,
        predictor,
        coverage_target=args.coverage_target,
        spectral_efficiency_target=args.spectral_efficiency_target,
        w1=args.w1,
        w2=args.w2,
        coverage_threshold_db=args.coverage_threshold_db,
        noise_coefficient_db=args.noise_coefficient_db,
    )
    return {
        "method": "ppo_init",
        "selected_actions": selected_actions,
        "positions_xy": [[int(col), int(row)] for row, col in tx_locs],
        **metrics,
    }


def run_module_state_init(
    *,
    method_name: str,
    map_path: Path,
    predictor: LocalSurrogatePredictor,
    args: argparse.Namespace,
    module_state_path: str,
) -> dict[str, Any]:
    pixel_map = load_map_normalized(map_path)
    selected_actions, tx_locs = select_module_state_layout(
        pixel_map=pixel_map,
        module_state_path=module_state_path,
        k_max=args.k_max,
        policy_version=args.policy_version,
        coverage_threshold_db=args.coverage_threshold_db,
        device_name=args.device,
        predictor=predictor,
        coverage_target=args.coverage_target,
        spectral_efficiency_target=args.spectral_efficiency_target,
        w1=args.w1,
        w2=args.w2,
        noise_coefficient_db=args.noise_coefficient_db,
    )
    metrics = evaluate_layout(
        pixel_map,
        tx_locs,
        predictor,
        coverage_target=args.coverage_target,
        spectral_efficiency_target=args.spectral_efficiency_target,
        w1=args.w1,
        w2=args.w2,
        coverage_threshold_db=args.coverage_threshold_db,
        noise_coefficient_db=args.noise_coefficient_db,
    )
    return {
        "method": method_name,
        "selected_actions": selected_actions,
        "positions_xy": [[int(col), int(row)] for row, col in tx_locs],
        **metrics,
    }


def heuristic_output_dir(root_output_dir: Path, method: str, map_path: Path) -> Path:
    return root_output_dir / "heuristic_runs" / normalize_method_name(method) / map_path.stem


def build_heuristic_command(method: str, map_path: Path, output_dir: Path, args: argparse.Namespace) -> list[str]:
    normalized_method = normalize_method_name(method)
    resolved_device = str(_resolve_device(args.device))
    command = [
        sys.executable,
        str(HEURISTIC_SCRIPT_BY_METHOD[normalized_method]),
        "--height-map",
        str(map_path),
        "--k-max",
        str(args.k_max),
        "--coverage-target",
        str(args.coverage_target),
        "--spectral-efficiency-target",
        str(args.spectral_efficiency_target),
        "--model-path",
        str(Path(args.model_path).expanduser().resolve()),
        "--network-type",
        args.network_type,
        "--output-dir",
        str(output_dir),
        "--device",
        resolved_device,
        "--w1",
        str(args.w1),
        "--w2",
        str(args.w2),
        "--coverage-threshold-db",
        str(args.coverage_threshold_db),
        "--noise-coefficient-db",
        str(args.noise_coefficient_db),
    ]
    if normalized_method in {"run_sa", "run_greedy", "run_ga", "run_pso", "run_candidate_enumeration"}:
        command += ["--max-evals", str(args.max_evals)]
    if normalized_method in {"run_sa", "run_greedy", "run_ga", "run_pso"}:
        command += ["--d-min", str(args.d_min), "--repair-max-tries", str(args.repair_max_tries)]
    if normalized_method == "run_sa":
        command += [
            "--initial-temp",
            str(args.sa_initial_temp),
            "--cooling-rate",
            str(args.sa_cooling_rate),
            "--gaussian-sigma",
            str(args.sa_gaussian_sigma),
        ]
    elif normalized_method == "run_greedy":
        command += [
            "--candidate-stride",
            str(args.greedy_candidate_stride),
            "--candidate-limit",
            str(args.greedy_candidate_limit),
        ]
    elif normalized_method == "run_ga":
        command += [
            "--population-size",
            str(args.ga_population_size),
            "--elite-size",
            str(args.ga_elite_size),
            "--tournament-size",
            str(args.ga_tournament_size),
            "--mutation-rate",
            str(args.ga_mutation_rate),
            "--gaussian-sigma",
            str(args.ga_gaussian_sigma),
        ]
    elif normalized_method == "run_pso":
        command += [
            "--swarm-size",
            str(args.pso_swarm_size),
            "--inertia",
            str(args.pso_inertia),
            "--c1",
            str(args.pso_c1),
            "--c2",
            str(args.pso_c2),
            "--velocity-clamp",
            str(args.pso_velocity_clamp),
        ]
    elif normalized_method == "run_candidate_enumeration":
        command += [
            "--candidate-stride",
            str(args.bruteforce_candidate_stride),
            "--candidate-limit",
            str(args.bruteforce_candidate_limit),
        ]
    return command


def extract_primary_metrics(best_metrics: dict[str, Any]) -> dict[str, Any]:
    if "coverage" in best_metrics and "spectral_efficiency" in best_metrics:
        normalized = dict(best_metrics)
        if "score" not in normalized and "best_score" in normalized:
            normalized["score"] = normalized["best_score"]
        if "channel_capacity_mbps" not in normalized and "channel_capacity" in normalized:
            normalized["channel_capacity_mbps"] = normalized["channel_capacity"]
        return normalized

    primary_kind = best_metrics.get("primary_layout_kind")
    if primary_kind == "first_feasible" and isinstance(best_metrics.get("first_feasible"), dict):
        return best_metrics["first_feasible"]
    if isinstance(best_metrics.get("global_best_score"), dict):
        return best_metrics["global_best_score"]
    raise ValueError("Cannot extract primary metrics from best_metrics.json")


def run_heuristic_method(method: str, map_path: Path, root_output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    normalized_method = normalize_method_name(method)
    output_dir = heuristic_output_dir(root_output_dir, normalized_method, map_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_heuristic_command(normalized_method, map_path, output_dir, args)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{normalized_method} failed on {map_path.name} with return code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )

    best_metrics_path = output_dir / "best_metrics.json"
    if not best_metrics_path.exists():
        raise FileNotFoundError(f"Expected heuristic output not found: {best_metrics_path}")
    best_metrics = json.loads(best_metrics_path.read_text(encoding="utf-8"))
    primary = extract_primary_metrics(best_metrics)
    return {
        "method": normalized_method,
        "positions_xy": primary.get("positions_xy") or primary.get("best_positions_xy") or best_metrics.get("best_positions_xy"),
        "positions_xy_rounded": (
            primary.get("positions_xy_rounded")
            or primary.get("best_positions_xy_rounded")
            or best_metrics.get("best_positions_xy_rounded")
        ),
        "coverage": float(primary["coverage"]),
        "spectral_efficiency": float(primary["spectral_efficiency"]),
        "channel_capacity_mbps": float(primary["channel_capacity_mbps"]),
        "redundancy_rate": float(primary["redundancy_rate"]),
        "score": float(primary["score"]),
        "base_score": float(primary["base_score"]),
        "penalty": float(primary["penalty"]),
        "eval_count": int(best_metrics.get("eval_count", 0)),
        "solver_runtime_sec": float(best_metrics["total_runtime_sec"]) if "total_runtime_sec" in best_metrics else None,
        "heuristic_output_dir": str(output_dir),
    }


def save_records_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    if not records:
        return
    fieldnames: list[str] = []
    for record in records:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    success_records = [record for record in records if int(record.get("success", 0)) == 1]
    metrics = (
        "coverage",
        "spectral_efficiency",
        "channel_capacity_mbps",
        "redundancy_rate",
        "score",
        "runtime_sec",
    )
    summary: dict[str, dict[str, float | int]] = {}
    methods = sorted({record["method"] for record in records})
    for method in methods:
        method_records = [record for record in success_records if record["method"] == method]
        method_summary: dict[str, float | int] = {
            "count": len(method_records),
            "failure_count": sum(1 for record in records if record["method"] == method and int(record.get("success", 0)) == 0),
        }
        if method_records:
            for metric in metrics:
                values = np.asarray([float(record[metric]) for record in method_records], dtype=np.float64)
                method_summary[f"mean_{metric}"] = float(np.mean(values))
                method_summary[f"std_{metric}"] = float(np.std(values))
        summary[method] = method_summary
    return summary


def save_summary_csv(summary: dict[str, dict[str, float | int]], output_path: Path) -> None:
    rows = []
    for method, metrics in summary.items():
        row = {"method": method, **metrics}
        rows.append(row)
    save_records_csv(rows, output_path)


def run_one_method(
    method: str,
    map_path: Path,
    predictor: LocalSurrogatePredictor,
    args: argparse.Namespace,
    rng: np.random.Generator,
    root_output_dir: Path,
) -> dict[str, Any]:
    normalized_method = normalize_method_name(method)
    start = time.perf_counter()
    if normalized_method == "random_init":
        payload = run_random_init(map_path, predictor, args, rng)
    elif normalized_method == "ppo_init":
        payload = run_ppo_init(map_path, predictor, args)
    elif normalized_method == "pretrain_init":
        if not args.pretrain_module_state:
            raise ValueError("--pretrain-module-state is required when methods include pretrain_init")
        payload = run_module_state_init(
            method_name="pretrain_init",
            map_path=map_path,
            predictor=predictor,
            args=args,
            module_state_path=args.pretrain_module_state,
        )
    elif normalized_method == "bandit_init":
        if not args.bandit_module_state:
            raise ValueError("--bandit-module-state is required when methods include bandit_init")
        payload = run_module_state_init(
            method_name="bandit_init",
            map_path=map_path,
            predictor=predictor,
            args=args,
            module_state_path=args.bandit_module_state,
        )
    else:
        payload = run_heuristic_method(normalized_method, map_path, root_output_dir, args)
    runtime_sec = time.perf_counter() - start
    record = {
        "method": normalized_method,
        "image": str(map_path),
        "sample_name": map_path.stem,
        "success": 1,
        "runtime_sec": float(runtime_sec),
        **payload,
    }
    positions_xy = record.get("positions_xy_rounded") or record.get("positions_xy")
    if positions_xy:
        pixel_map = load_map_normalized(map_path)
        record["positions_xyz"] = add_position_heights(pixel_map, positions_xy)
    return record


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.methods = [normalize_method_name(method) for method in args.methods]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    map_paths = load_map_paths(args.maps_file, limit=args.map_limit)
    predictor = LocalSurrogatePredictor(args.model_path, args.network_type, args.device)
    rng = np.random.default_rng(args.seed)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for map_index, map_path in enumerate(map_paths, start=1):
        print(f"[{map_index}/{len(map_paths)}] {map_path.name}")
        for method in args.methods:
            try:
                record = run_one_method(method, map_path, predictor, args, rng, output_dir)
                records.append(record)
                print(
                    f"  {method}: coverage={record['coverage']:.4f} "
                    f"se={record['spectral_efficiency']:.4f} "
                    f"runtime_sec={record['runtime_sec']:.4f} "
                    f"positions_xyz={record.get('positions_xyz')}"
                )
            except Exception as exc:
                failure = {
                    "method": method,
                    "image": str(map_path),
                    "sample_name": map_path.stem,
                    "success": 0,
                    "error": str(exc),
                }
                failures.append(failure)
                records.append(failure.copy())
                print(f"  {method}: FAILED {exc}")
                if args.stop_on_error:
                    raise

    summary = summarize_records(records)
    all_records_json = output_dir / "all_records.json"
    all_records_csv = output_dir / "all_records.csv"
    summary_json = output_dir / "summary_by_method.json"
    summary_csv = output_dir / "summary_by_method.csv"
    failures_json = output_dir / "failures.json"

    all_records_json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    save_records_csv(records, all_records_csv)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_summary_csv(summary, summary_csv)
    failures_json.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
