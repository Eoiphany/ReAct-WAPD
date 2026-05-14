"""注释
命令示例:
python -m visualization.react_decision_gui

参数含义:
- `REQUEST_OPTIONS`: GUI 允许的需求文件选项，只包含 fixed 与 unfixed 两类。
- `PLANNER_OPTIONS`: GUI 允许的 planner，仅包含 qwen 与 llamafactory。
- `FOLLOW_UP_METHOD_OPTIONS`: GUI 后续决策方法；默认使用 explain_weighted。
- `build_decision_command(...)`: 根据 GUI 当前选项拼装 `ReAct.run_access_point_decision` 命令。

逻辑说明:
本文件集中维护 GUI 可选运行项与命令拼装规则，避免把 request/planner/method 的白名单散落在界面代码里。
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from ReAct.experiments.summary_utils import DEFAULT_TWO_STAGE_MODULE_STATE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUEST_ROOT = PROJECT_ROOT / "ReAct" / "exp" / "exp1_fixed_vs_unfixed" / "_generated_requests"
def _autobs_python_candidates() -> list[Path]:
    preferred_homes = [
        Path.home() / "miniconda3" / "envs" / "autobs" / "bin",
        Path.home() / "Miniconda" / "envs" / "autobs" / "bin",
    ]
    return [
        *(candidate for home in preferred_homes for candidate in (home / "python3.10", home / "python", home / "python3.12")),
        PROJECT_ROOT / ".." / "miniconda3" / "envs" / "autobs" / "bin" / "python3.10",
        PROJECT_ROOT / ".." / "miniconda3" / "envs" / "autobs" / "bin" / "python",
        PROJECT_ROOT / ".." / "miniconda3" / "envs" / "autobs" / "bin" / "python3.12",
        PROJECT_ROOT / ".." / "Miniconda" / "envs" / "autobs" / "bin" / "python3.10",
        PROJECT_ROOT / ".." / "Miniconda" / "envs" / "autobs" / "bin" / "python",
        PROJECT_ROOT / ".." / "Miniconda" / "envs" / "autobs" / "bin" / "python3.12",
    ]


def _qwen_python_candidates() -> list[Path]:
    preferred_homes = [
        Path.home() / "miniconda3" / "envs" / "qwen" / "bin",
        Path.home() / "Miniconda" / "envs" / "qwen" / "bin",
    ]
    return [
        *(candidate for home in preferred_homes for candidate in (home / "python3.10", home / "python", home / "python3.12")),
        PROJECT_ROOT / ".." / "miniconda3" / "envs" / "qwen" / "bin" / "python3.10",
        PROJECT_ROOT / ".." / "miniconda3" / "envs" / "qwen" / "bin" / "python",
        PROJECT_ROOT / ".." / "miniconda3" / "envs" / "qwen" / "bin" / "python3.12",
        PROJECT_ROOT / ".." / "Miniconda" / "envs" / "qwen" / "bin" / "python3.10",
        PROJECT_ROOT / ".." / "Miniconda" / "envs" / "qwen" / "bin" / "python",
        PROJECT_ROOT / ".." / "Miniconda" / "envs" / "qwen" / "bin" / "python3.12",
    ]


def _first_existing_python(candidates: list[Path], fallback: Path) -> Path:
    for candidate in candidates:
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return path
    return fallback


def _resolve_default_python() -> Path:
    env_value = os.environ.get("VISUALIZATION_PYTHON", "").strip()
    if env_value:
        return Path(env_value).expanduser()
    current_python = Path(sys.executable).expanduser().resolve()
    if current_python.exists():
        return current_python
    return _first_existing_python(_autobs_python_candidates(), current_python)


DEFAULT_PYTHON = _resolve_default_python()

REQUEST_OPTIONS = [
    {"label": "fixed_sites_4", "path": REQUEST_ROOT / "fixed_sites_4.txt"},
    {"label": "coverage_capacity_budget_site6", "path": REQUEST_ROOT / "coverage_capacity_budget_site6.txt"},
]
PLANNER_OPTIONS = [
    "heuristic_greedy",
    "heuristic_sa",
    "heuristic_ga",
    "heuristic_pso",
    "heuristic_exhaustive",
    "qwen",
    "llamafactory",
]
EVAL_MODEL_OPTIONS = ["pmnet", "rmnet"]
EXECUTION_TARGET_OPTIONS = ["local", "server"]
FOLLOW_UP_METHOD_OPTIONS = [
    {"key": "heuristic_greedy", "planner": "heuristic_greedy", "llm_mode": "decide", "label": "Greedy"},
    {"key": "heuristic_sa", "planner": "heuristic_sa", "llm_mode": "decide", "label": "SA"},
    {"key": "heuristic_ga", "planner": "heuristic_ga", "llm_mode": "decide", "label": "GA"},
    {"key": "heuristic_pso", "planner": "heuristic_pso", "llm_mode": "decide", "label": "PSO"},
    {"key": "heuristic_exhaustive", "planner": "heuristic_exhaustive", "llm_mode": "decide", "label": "全量穷举"},
    {"key": "llm_decide", "planner": "", "llm_mode": "decide", "label": "LLM直接输出动作"},
    {"key": "llm_explain_weighted", "planner": "", "llm_mode": "explain_weighted", "label": "LLM可解释性权重"},
]
DEFAULT_FOLLOW_UP_METHOD = "llm_explain_weighted"
DEFAULT_INIT_MODE = "two_stage"
DEFAULT_EVAL_MODEL = "rmnet"
DEFAULT_EVAL_DEVICE = "auto"
DEFAULT_MAX_STEPS = 8
DEFAULT_CANDIDATE_SAMPLE = 16
DEFAULT_LLM_TOP_K = 8
DEFAULT_HEURISTIC_SEARCH_BUDGET = 600
DEFAULT_HEURISTIC_ONLINE_CANDIDATE_SAMPLE = 128
DEFAULT_QWEN_DEVICE = "auto"
DEFAULT_QWEN_DTYPE = "auto"
DEFAULT_QWEN_MAX_NEW_TOKENS = 320
DEFAULT_LLAMAFACTORY_TEMPLATE = "qwen"
DEFAULT_LLAMAFACTORY_BACKEND = "huggingface"
DEFAULT_LLAMAFACTORY_DTYPE = "auto"
DEFAULT_TWO_STAGE_VERSION = "auto"
DEFAULT_TWO_STAGE_INIT_K = 1


def _first_existing_path(*candidates: Path | str | None) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return path
    return None


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


DEFAULT_QWEN_MODEL_PATH = _first_existing_path(
    _env_path("VIS_QWEN_MODEL_PATH"),
    PROJECT_ROOT / "Qwen" / "Qwen2.5-7B",
    PROJECT_ROOT / "Qwen2.5-7B",
)
DEFAULT_LLAMAFACTORY_ROOT = _first_existing_path(
    _env_path("VIS_LLAMAFACTORY_ROOT"),
    PROJECT_ROOT / "Qwen" / "LLaMA-Factory",
    PROJECT_ROOT / "LLaMA-Factory",
)
DEFAULT_LLAMAFACTORY_ADAPTER = _first_existing_path(
    _env_path("VIS_LLAMAFACTORY_ADAPTER"),
    PROJECT_ROOT / "Qwen" / "LLaMA-Factory" / "saves" / "Qwen2.5-7B" / "lora" / "train_2026-02-14-14-09-21",
    PROJECT_ROOT / "LLaMA-Factory" / "saves" / "Qwen2.5-7B" / "lora" / "train_2026-02-14-14-09-21",
)
DEFAULT_LLAMAFACTORY_MODEL = _first_existing_path(
    _env_path("VIS_LLAMAFACTORY_MODEL"),
    DEFAULT_QWEN_MODEL_PATH,
)


def validate_runtime_inputs(*, map_path: Path, request_key: str, planner: str, eval_model: str) -> list[str]:
    errors: list[str] = []
    if not map_path.exists():
        errors.append(f"Map path not found: {map_path}")
    try:
        request_path = request_path_for_key(request_key)
    except KeyError as exc:
        errors.append(str(exc))
    else:
        if not request_path.exists():
            errors.append(f"Request file not found: {request_path}")
    if planner not in PLANNER_OPTIONS:
        errors.append(f"Unsupported planner: {planner}")
    if eval_model not in EVAL_MODEL_OPTIONS:
        errors.append(f"Unsupported eval model: {eval_model}")
    if planner == "qwen" and DEFAULT_QWEN_MODEL_PATH is None:
        errors.append("Qwen model path not found. Expected a valid directory such as paper_experiment/Qwen/Qwen2.5-7B.")
    if planner == "llamafactory":
        if DEFAULT_LLAMAFACTORY_ROOT is None:
            errors.append("LLaMA-Factory root not found. Expected a valid directory such as paper_experiment/Qwen/LLaMA-Factory.")
        if DEFAULT_LLAMAFACTORY_MODEL is None:
            errors.append("LLaMA-Factory model path not found. Expected a valid directory such as paper_experiment/Qwen/Qwen2.5-7B.")
        if DEFAULT_LLAMAFACTORY_ADAPTER is None:
            errors.append(
                "LLaMA-Factory adapter not found. Expected a valid adapter directory such as "
                "paper_experiment/Qwen/LLaMA-Factory/saves/Qwen2.5-7B/lora/train_2026-02-14-14-09-21."
            )
    if not DEFAULT_TWO_STAGE_MODULE_STATE.exists():
        errors.append(f"Two-stage module state not found: {DEFAULT_TWO_STAGE_MODULE_STATE}")
    return errors


def resolve_local_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_eval_device() -> str:
    if DEFAULT_EVAL_DEVICE == "auto":
        return resolve_local_device()
    return DEFAULT_EVAL_DEVICE


def resolve_python_for_target(execution_target: str, current_python: Path | None = None) -> Path:
    base_python = Path(current_python or DEFAULT_PYTHON).expanduser().resolve()
    if execution_target == "server":
        return _first_existing_python(_qwen_python_candidates(), base_python)
    return _first_existing_python(_autobs_python_candidates(), base_python)


def request_path_for_key(request_key: str) -> Path:
    for item in REQUEST_OPTIONS:
        if item["label"] == request_key:
            return Path(item["path"])
    raise KeyError(f"Unknown request key: {request_key}")


def follow_up_option_for_key(method_key: str) -> dict[str, str]:
    for item in FOLLOW_UP_METHOD_OPTIONS:
        if item["key"] == method_key:
            return item
    raise KeyError(f"Unknown follow-up method key: {method_key}")


def build_decision_command(
    *,
    python_executable: Path,
    project_root: Path,
    map_path: Path,
    request_key: str,
    planner: str,
    eval_model: str,
    follow_up_method: str,
    max_steps: int,
    candidate_sample: int,
    llm_top_k_candidates: int,
    heuristic_search_budget: int,
    heuristic_online_candidate_sample: int,
    traj_dir: Path,
    traj_id: str,
    visualization_sync_dir: Path | None = None,
    visualization_sync_timeout_sec: float = 0.0,
) -> list[str]:
    request_path = request_path_for_key(request_key)
    eval_device = resolve_eval_device()
    return [
        str(python_executable),
        "-u",
        "-m",
        "ReAct.run_access_point_decision",
        "--city-map-path",
        str(map_path),
        "--user-request-path",
        str(request_path),
        "--planner",
        planner,
        "--llm-decision-mode",
        follow_up_method,
        "--max-steps",
        str(int(max_steps)),
        "--candidate-sample",
        str(int(candidate_sample)),
        "--llm-top-k-candidates",
        str(int(llm_top_k_candidates)),
        "--heuristic-search-budget",
        str(int(heuristic_search_budget)),
        "--heuristic-online-candidate-sample",
        str(int(heuristic_online_candidate_sample)),
        "--traj-dir",
        str(traj_dir),
        "--traj-id",
        traj_id,
        "--init-mode",
        DEFAULT_INIT_MODE,
        "--two-stage-init-k",
        "1",
        "--eval-model",
        eval_model,
        "--eval-device",
        eval_device,
        "--qwen-model-path",
        str(DEFAULT_QWEN_MODEL_PATH or ""),
        "--qwen-device",
        eval_device,
        "--qwen-dtype",
        DEFAULT_QWEN_DTYPE,
        "--qwen-max-new-tokens",
        str(DEFAULT_QWEN_MAX_NEW_TOKENS),
        "--llamafactory-root",
        str(DEFAULT_LLAMAFACTORY_ROOT or ""),
        "--llamafactory-model",
        str(DEFAULT_LLAMAFACTORY_MODEL or ""),
        "--llamafactory-adapter",
        str(DEFAULT_LLAMAFACTORY_ADAPTER or ""),
        "--llamafactory-template",
        DEFAULT_LLAMAFACTORY_TEMPLATE,
        "--llamafactory-backend",
        DEFAULT_LLAMAFACTORY_BACKEND,
        "--llamafactory-dtype",
        DEFAULT_LLAMAFACTORY_DTYPE,
        "--two-stage-module-state",
        str(DEFAULT_TWO_STAGE_MODULE_STATE),
        "--two-stage-version",
        DEFAULT_TWO_STAGE_VERSION,
        "--print-step",
        "--visualization-sync-dir",
        str(visualization_sync_dir or ""),
        "--visualization-sync-timeout-sec",
        str(float(visualization_sync_timeout_sec)),
    ]
