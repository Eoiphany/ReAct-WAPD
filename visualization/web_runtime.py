"""注释
命令示例:
python -m py_compile visualization/web_runtime.py

参数含义:
- `build_options_payload(...)`: 生成前端启动时需要的固定选项与地图列表。
- `build_state_payload(...)`: 把运行状态和 dashboard 数据拼成统一 API 响应。
- `WebRuntimeManager`: Web 后端运行管理器，负责启动 ReAct、轮询 trajectory、刷新图片。

逻辑说明:
本文件把 Web 版可视化需要的运行控制集中到一个轻量管理器里，后端 HTTP 层只负责把请求转交给它，不直接碰
`ReAct` 的命令装配、trajectory 解析和图片生成细节。
"""

from __future__ import annotations

import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from visualization import image_jobs, runtime_config, trajectory_parser
from visualization.live_step_renderer import (
    preload_predictor,
    render_capacity_heatmap,
    render_deployment_prediction,
    render_metric_history_trend,
    render_roi_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIS_ROOT = Path(__file__).resolve().parent
SURROGATE_MODULE_ROOT = PROJECT_ROOT / "surrogate"
if str(SURROGATE_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(SURROGATE_MODULE_ROOT))

from data_surrogate import resolve_radiomap_sample_pairs, resolve_usc_sample_ids
from utils import configure_plot_style

DATASET_MAP_ROOT = PROJECT_ROOT / "dataset" / "png" / "buildingsWHeight"
MAP_CANDIDATE_TRAJ_ROOT = (
    PROJECT_ROOT
    / "ReAct"
    / "exp"
    / "exp1_fixed_vs_unfixed"
    / "runs"
    / "llm"
    / "unfixed_llamafactory_explain_weighted"
    / "trajs"
)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
SURROGATE_PROGRESS_RE = re.compile(r"^(?:Eval|Epoch\s+\d+/\d+):")
SURROGATE_IGNORED_LOG_PATTERNS = [
    re.compile(r"^libgomp: Invalid value for environment variable OMP_NUM_THREADS$"),
    re.compile(r"^findfont:"),
    re.compile(r"^Font family ['\"].+['\"] not found"),
    re.compile(r"^UserWarning: Glyph .+ missing from font"),
    re.compile(r"^Glyph .+ missing from font"),
]


def _numeric_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (0, f"{int(path.stem):09d}")
    except ValueError:
        return (1, path.stem)


def _build_dataset_map_entries() -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for path in _select_top_dataset_maps():
        entries.append({"label": path.name, "path": str(path), "source": "dataset"})
    return entries


def _select_top_dataset_maps(limit: int = 10) -> list[Path]:
    ranked: list[tuple[tuple[float, float, float, float, float, float], Path]] = []
    for traj_path in sorted(MAP_CANDIDATE_TRAJ_ROOT.glob("*.json")):
        try:
            payload = json.loads(traj_path.read_text(encoding="utf-8"))
            latest = payload[-1]
            observation = json.loads(latest["observations"][-1])
            state = observation.get("state", {}) or {}
            metrics = state.get("last_metrics", {}) or {}
            sites = state.get("sites", []) or []
            coverage = float(metrics.get("coverage", 0.0) or 0.0)
            capacity = float(metrics.get("capacity", 0.0) or 0.0)
            redundancy = float(metrics.get("redundancy_rate", 0.0) or 0.0)
            steps = float(len(latest.get("step_logs", []) or []))
            site_count = float(len(sites))
            hit_target = 1.0 if (coverage >= 0.92 and capacity >= 1.85 and site_count <= 6.0) else 0.0
            map_id = traj_path.stem.split("__", 1)[0]
            map_path = DATASET_MAP_ROOT / f"{map_id}.png"
            if not map_path.exists():
                continue
            score = (hit_target, capacity, coverage, -site_count, -steps, -redundancy)
            ranked.append((score, map_path))
        except Exception:
            continue
    if ranked:
        ranked.sort(reverse=True, key=lambda item: item[0])
        selected: list[Path] = []
        seen: set[str] = set()
        for _, map_path in ranked:
            if map_path.name in seen:
                continue
            seen.add(map_path.name)
            selected.append(map_path)
            if len(selected) >= limit:
                break
        if selected:
            return selected
    return sorted(DATASET_MAP_ROOT.glob("*.png"), key=_numeric_sort_key)[:limit]

def list_map_entries() -> list[dict[str, str]]:
    return _build_dataset_map_entries()


def build_options_payload(map_entries: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "maps": map_entries,
        "request_options": [item["label"] for item in runtime_config.REQUEST_OPTIONS],
        "planner_options": list(runtime_config.PLANNER_OPTIONS),
        "eval_model_options": list(runtime_config.EVAL_MODEL_OPTIONS),
        "execution_target_options": list(runtime_config.EXECUTION_TARGET_OPTIONS),
        "follow_up_methods": list(runtime_config.FOLLOW_UP_METHOD_OPTIONS),
        "default_follow_up_method": runtime_config.DEFAULT_FOLLOW_UP_METHOD,
        "default_init_method": "TSPL",
        "defaults": {
            "max_steps": runtime_config.DEFAULT_MAX_STEPS,
            "candidate_sample": runtime_config.DEFAULT_CANDIDATE_SAMPLE,
            "llm_top_k_candidates": runtime_config.DEFAULT_LLM_TOP_K,
            "heuristic_search_budget": runtime_config.DEFAULT_HEURISTIC_SEARCH_BUDGET,
            "heuristic_online_candidate_sample": runtime_config.DEFAULT_HEURISTIC_ONLINE_CANDIDATE_SAMPLE,
            "planner": "llamafactory",
            "request_key": runtime_config.REQUEST_OPTIONS[0]["label"],
            "eval_model": runtime_config.DEFAULT_EVAL_MODEL,
            "execution_target": "local",
        },
        "surrogate": build_surrogate_options_payload(),
    }


def build_state_payload(
    *,
    status: str,
    traj_path: Path | None,
    dashboard_state: dict[str, Any] | None,
    runtime_paths: image_jobs.VisualizationRuntimePaths,
    stdout_lines: list[str] | None = None,
    stderr_lines: list[str] | None = None,
    image_error: str = "",
    image_version: str = "",
    process_running: bool = False,
) -> dict[str, Any]:
    def latest_status_line(lines: list[str] | None) -> str:
        for line in reversed(lines or []):
            if line.startswith("COMMAND:"):
                continue
            if line.startswith("[") or line.startswith("START") or line.startswith("STOP") or line.startswith("IMAGE"):
                return line
        return ""

    state = dashboard_state or {}
    diagnosis = state.get("diagnosis", "")
    if not diagnosis:
        diagnosis = latest_status_line(stdout_lines)
    if not diagnosis and image_error:
        diagnosis = image_error
    if not diagnosis and status == "failed":
        stderr_preview = "\n".join((stderr_lines or [])[-8:])
        diagnosis = stderr_preview or image_error or "Decision run failed."
    roi_path = runtime_paths.roi_root / "current_roi.png"
    pred_path = runtime_paths.pred_root / "current_pred" / "latest_pred.png"
    capacity_path = runtime_paths.heatmap_root / "current_capacity.png"
    trend_path = runtime_paths.trend_root / "current_metric_trend.png"
    flow_state = state.get(
        "flow",
        trajectory_parser.determine_flow_state(has_trajectory=False, process_running=False, step_count=0, finished=False),
    )
    if (
        process_running
        and int(state.get("current_ap_count", 0) or 0) >= 1
        and int(state.get("current_step", 0) or 0) == 0
    ):
        flow_state = dict(flow_state)
        flow_state["select_request"] = "complete"
        flow_state["request_structuring"] = "complete"
        flow_state["initial_deployment"] = "complete"
        flow_state["decision_loop"] = "current"
        flow_state["decision_complete_arrow"] = "pending"
        flow_state["complete"] = "pending"
    if status == "done":
        flow_state = dict(flow_state)
        flow_state["select_request"] = "complete"
        flow_state["request_structuring"] = "complete"
        flow_state["initial_deployment"] = "complete"
        flow_state["decision_loop"] = "complete"
        flow_state["decision_complete_arrow"] = "complete"
        flow_state["complete"] = "current"
    return {
        "status": status,
        "traj_path": str(traj_path) if traj_path else "",
        "current_step": int(state.get("current_step", 0) or 0),
        "current_ap_count": int(state.get("current_ap_count", 0) or 0),
        "request_text": state.get("request_text", ""),
        "diagnosis": diagnosis,
        "goal": state.get("goal", {}) or {},
        "goal_human_readable": state.get("goal_human_readable", ""),
        "constraints": state.get("constraints", {}) or {},
        "ok": bool(state.get("ok", False)),
        "metrics": state.get(
            "metrics",
            {"coverage": 0.0, "spectral_efficiency": 0.0, "redundancy_rate": 0.0},
        ),
        "sites": state.get("sites", []) or [],
        "table_rows": state.get("table_rows", []) or [],
        "metric_history": state.get("metric_history", []) or [],
        "method_labels": state.get(
            "method_labels",
            {"init": "TSPL", "follow_up": "LLM可解释性权重"},
        ),
        "flow": flow_state,
        "logs": {
            "stdout": stdout_lines or [],
            "stderr": stderr_lines or [],
            "image_error": image_error,
        },
        "images": {
            "roi_url": "runtime/images/roi/current_roi.png" if roi_path.exists() else "",
            "pred_url": "runtime/images/pred/current_pred/latest_pred.png" if pred_path.exists() else "",
            "capacity_url": "runtime/images/heatmap/current_capacity.png" if capacity_path.exists() else "",
            "trend_url": "runtime/images/trend/current_metric_trend.png" if trend_path.exists() else "",
            "version": image_version,
        },
    }


SURROGATE_MODEL_LABELS = {
    "pmnet": "PMNet",
    "rmnet": "RMNet",
    "radiounet": "RadioUNet",
    "transunet": "TransUNet",
    "unet": "UNet",
}
SURROGATE_DATASET_LABELS = {"radiomap3dseer": "RadioMap3DSeer", "usc": "USC"}
SURROGATE_RUNTIME_ROOT = VIS_ROOT / "runtime" / "surrogate"
DEFAULT_SERVER_PROJECT_ROOT = Path("/root/autodl-tmp")
DEFAULT_SURROGATE_OUTPUT_ROOTS = {
    "local": "surrogate/local_runs",
    "server": "surrogate/server_runs",
}


def resolve_surrogate_project_root(execution_target: str) -> Path:
    if execution_target == "server":
        env_value = os.environ.get("VIS_SERVER_PROJECT_ROOT", "").strip()
        return Path(env_value).expanduser() if env_value else DEFAULT_SERVER_PROJECT_ROOT
    return PROJECT_ROOT


def default_surrogate_output_root(execution_target: str) -> str:
    return DEFAULT_SURROGATE_OUTPUT_ROOTS.get(execution_target, DEFAULT_SURROGATE_OUTPUT_ROOTS["local"])


def resolve_surrogate_path(execution_target: str, value: str, *, allow_empty: bool = False) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "" if allow_empty else str(resolve_surrogate_project_root(execution_target))
    project_root = resolve_surrogate_project_root(execution_target)
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    if execution_target == "server":
        parts = list(path.parts)
        if parts and parts[0] == project_root.name:
            path = Path(*parts[1:]) if len(parts) > 1 else Path(".")
    return str(project_root / path)


def resolve_surrogate_output_root(execution_target: str, value: str) -> str:
    raw = str(value or "").strip()
    default_relative = default_surrogate_output_root(execution_target)
    if not raw:
        return resolve_surrogate_path(execution_target, default_relative)
    known_defaults: set[str] = set(DEFAULT_SURROGATE_OUTPUT_ROOTS.values())
    for target, relative in DEFAULT_SURROGATE_OUTPUT_ROOTS.items():
        known_defaults.add(str(resolve_surrogate_project_root(target) / relative))
    if raw in known_defaults:
        return resolve_surrogate_path(execution_target, default_relative)
    return resolve_surrogate_path(execution_target, raw)


def resolve_surrogate_eval_run_dir(execution_target: str, value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    resolved = Path(resolve_surrogate_path(execution_target, raw, allow_empty=True)).resolve()
    direct_csv = resolved / "test_split.csv"
    if direct_csv.exists():
        return str(resolved)
    child_hits = sorted(
        {path.parent.resolve() for path in resolved.glob("*/test_split.csv")},
        key=lambda item: str(item),
    )
    if len(child_hits) == 1:
        return str(child_hits[0])
    if len(child_hits) > 1:
        raise ValueError(f"Multiple run directories found under: {resolved}")
    raise FileNotFoundError(f"test_split.csv not found under: {resolved}")


def _build_surrogate_process_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    omp_value = str(env.get("OMP_NUM_THREADS", "")).strip()
    if omp_value and omp_value.isdigit() and int(omp_value) > 0:
        return env
    env.pop("OMP_NUM_THREADS", None)
    return env


def format_shell_command(command: list[str]) -> str:
    if not command:
        return ""
    lines = [command[0]]
    index = 1
    while index < len(command):
        part = command[index]
        if part.startswith("--"):
            if index + 1 < len(command) and not command[index + 1].startswith("--"):
                lines.append(f"  {part} {command[index + 1]} \\")
                index += 2
                continue
            lines.append(f"  {part} \\")
            index += 1
            continue
        lines.append(f"  {part} \\")
        index += 1
    if len(lines) > 1:
        lines[0] = lines[0] + " \\"
        lines[-1] = lines[-1].removesuffix(" \\")
    return "\n".join(lines)


def build_surrogate_options_payload() -> dict[str, Any]:
    return {
        "project_roots": {
            "local": str(PROJECT_ROOT),
            "server": str(resolve_surrogate_project_root("server")),
        },
        "python_executables": {
            "local": str(runtime_config.resolve_python_for_target("local", runtime_config.DEFAULT_PYTHON)),
            "server": str(runtime_config.resolve_python_for_target("server", runtime_config.DEFAULT_PYTHON)),
        },
        "default_execution_target": "local",
        "datasets": {
            "radiomap3dseer": {
                "data_root": "dataset",
                "csv_file": "",
                "output_roots": DEFAULT_SURROGATE_OUTPUT_ROOTS,
                "output_stride": 16,
                "batch_size": 16,
                "epochs": 30,
                "train_ratio": 0.7,
                "val_ratio": 0.1,
                "test_ratio": 0.2,
                "lr": "0.0001",
                "lr_decay": 0.5,
                "step_size": 10,
                "weight_decay": 0.0,
                "val_freq": 1,
                "log_freq": 100,
                "num_workers": 4,
                "seed": 42,
                "use_height": "true",
                "test_samples": ["scene=348, tx=7", "scene=488, tx=4", "scene=678, tx=11"],
            },
            "usc": {
                "data_root": "usc-data",
                "csv_file": "",
                "output_roots": DEFAULT_SURROGATE_OUTPUT_ROOTS,
                "output_stride": 16,
                "batch_size": 16,
                "epochs": 30,
                "train_ratio": 0.7,
                "val_ratio": 0.1,
                "test_ratio": 0.2,
                "lr": "0.0005",
                "lr_decay": 0.5,
                "step_size": 10,
                "weight_decay": 0.0,
                "val_freq": 1,
                "log_freq": 100,
                "num_workers": 4,
                "seed": 42,
                "use_height": "false",
                "test_samples": ["sample_id=00017", "sample_id=00042", "sample_id=00108"],
            },
        },
    }


def _surrogate_run_dir_from_payload(payload: dict[str, Any], *, increment_if_exists: bool = False) -> Path:
    dataset_key = str(payload["datasetKey"])
    model_type = str(payload["modelType"])
    batch_size = str(payload["batchSize"])
    lr = str(payload["lr"])
    lr_decay = str(payload["lrDecay"])
    step_size = str(payload["stepSize"])
    run_dir = Path(payload["outputRoot"]) / f"{model_type}_{dataset_key}" / f"{batch_size}_{lr}_{lr_decay}_{step_size}"
    if not increment_if_exists or not run_dir.exists():
        return run_dir
    suffix = 1
    while True:
        candidate = run_dir.parent / f"{run_dir.name}_{suffix:03d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _surrogate_train_script_path(dataset_key: str) -> Path:
    script_name = "train_radiomap3dseer_surrogate.py" if dataset_key == "radiomap3dseer" else "train_usc_surrogate.py"
    return PROJECT_ROOT / "surrogate" / script_name


def _surrogate_train_command(payload: dict[str, Any]) -> list[str]:
    dataset_key = str(payload["datasetKey"])
    python_executable = runtime_config.resolve_python_for_target(
        str(payload.get("executionTarget") or "local"),
        runtime_config.DEFAULT_PYTHON,
    )
    command = [
        str(python_executable),
        str(_surrogate_train_script_path(dataset_key)),
        "--model-type",
        str(payload["modelType"]),
        "--data-root",
        str(payload["dataRoot"]),
        "--output-root",
        str(payload["outputRoot"]),
        "--output-stride",
        str(payload["outputStride"]),
        "--batch-size",
        str(payload["batchSize"]),
        "--train-ratio",
        str(payload["trainRatio"]),
        "--test-ratio",
        str(payload["testRatio"]),
        "--lr",
        str(payload["lr"]),
        "--lr-decay",
        str(payload["lrDecay"]),
        "--step",
        str(payload["stepSize"]),
        "--weight-decay",
        str(payload["weightDecay"]),
        "--val-freq",
        str(payload["valFreq"]),
        "--num-workers",
        str(payload["numWorkers"]),
        "--log-every",
        str(payload["logFreq"]),
        "--seed",
        str(payload["seed"]),
    ]
    if payload.get("csvFile"):
        command.extend(["--csv-file", str(payload["csvFile"])])
    if dataset_key == "radiomap3dseer":
        command.extend(["--epochs", str(payload["epochs"])])
        command.append("--use-height" if str(payload["useHeight"]).lower() == "true" else "--no-use-height")
    else:
        command.extend(["--num-epochs", str(payload["epochs"])])
    return command


def _resolve_surrogate_sample_names(dataset_key: str, data_root: str, csv_file: str | None) -> list[str]:
    if dataset_key == "usc":
        return resolve_usc_sample_ids(data_root, csv_file)
    pairs = resolve_radiomap_sample_pairs(data_root, csv_file)
    return [f"{scene_id}_{tx_id}" for scene_id, tx_id in pairs]


def _format_surrogate_sample_items(dataset_key: str, sample_names: list[str], limit: int = 12) -> list[str]:
    subset = sample_names[:limit]
    if dataset_key == "usc":
        return [f"sample_id={item}" for item in subset]
    items: list[str] = []
    for item in subset:
        scene_id, tx_id = item.split("_", 1)
        items.append(f"scene={scene_id}, tx={tx_id}")
    return items


def _pick_random_surrogate_samples(sample_names: list[str], limit: int = 10) -> list[str]:
    if len(sample_names) <= limit:
        return sample_names
    return random.sample(sample_names, limit)


def resolve_surrogate_samples_payload(payload: dict[str, Any]) -> dict[str, Any]:
    execution_target = str(payload.get("executionTarget") or "local")
    dataset_key = str(payload.get("datasetKey") or "radiomap3dseer")
    data_root = resolve_surrogate_path(execution_target, str(payload.get("dataRoot") or ""))
    csv_file = resolve_surrogate_path(
        execution_target,
        str(payload.get("csvFile") or ""),
        allow_empty=True,
    ) or None
    checkpoint_run_dir = resolve_surrogate_eval_run_dir(
        execution_target,
        str(payload.get("checkpointRunDir") or ""),
    ) or None
    if not csv_file and checkpoint_run_dir:
        candidate = Path(checkpoint_run_dir).resolve() / "test_split.csv"
        if candidate.exists():
            csv_file = str(candidate)
    resolved_sample_names = _resolve_surrogate_sample_names(dataset_key, data_root, csv_file)
    provided_sample_names = [str(item).strip() for item in (payload.get("sampleNames") or []) if str(item).strip()]
    if provided_sample_names:
        available = set(resolved_sample_names)
        sample_names = [name for name in provided_sample_names if name in available]
        if not sample_names:
            sample_names = _pick_random_surrogate_samples(resolved_sample_names, limit=10) if csv_file else resolved_sample_names
    else:
        sample_names = _pick_random_surrogate_samples(resolved_sample_names, limit=10) if csv_file else resolved_sample_names
    selected_sample = str(payload.get("selectedSample") or "").strip()
    if selected_sample and selected_sample not in sample_names:
        selected_sample = ""
    if not selected_sample and sample_names:
        selected_sample = sample_names[0]
    if csv_file and checkpoint_run_dir and Path(csv_file).parent == Path(checkpoint_run_dir).resolve():
        sample_source = "来自当前输出目录的 test_split.csv"
    elif csv_file:
        sample_source = "来自显式指定的样本文件"
    else:
        sample_source = "来自当前数据集默认测试样本集合。"
    return {
        "datasetKey": dataset_key,
        "dataRoot": data_root,
        "csvFile": csv_file or "",
        "checkpointRunDir": checkpoint_run_dir or "",
        "sampleNames": sample_names,
        "sampleItems": _format_surrogate_sample_items(dataset_key, sample_names, limit=min(max(len(sample_names), 12), 200)),
        "sampleSource": sample_source,
        "selectedSample": selected_sample,
    }


def _render_surrogate_metric_panels(history_path: Path, output_dir: Path, title_prefix: str) -> list[dict[str, str]]:
    if not history_path.exists():
        return []
    history = json.loads(history_path.read_text(encoding="utf-8"))
    rows = [row for row in history if row.get("epoch") is not None]
    if not rows:
        return []

    mpl_cache_dir = output_dir / ".mpl-cache"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    configure_plot_style()
    import matplotlib.pyplot as plt

    plot_specs = [
        ("best_val_rmse", "Best Validation RMSE", "best_val_rmse.png"),
        ("train_loss", "Training Loss", "train_loss.png"),
        ("val_mae", "Validation MAE", "val_mae.png"),
        ("val_r2", "Validation R2", "val_r2.png"),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    plots: list[dict[str, str]] = []
    for metric_key, metric_title, filename in plot_specs:
        xs = [row["epoch"] for row in rows if row.get(metric_key) is not None]
        ys = [row[metric_key] for row in rows if row.get(metric_key) is not None]
        if not xs:
            continue
        figure_path = output_dir / filename
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.2)
        ax.set_title(f"{title_prefix} | {metric_title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_title)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        plots.append({"title": metric_title, "path": str(figure_path)})
    return plots


class SurrogateRuntimeManager:
    def __init__(self) -> None:
        self.runtime_root = SURROGATE_RUNTIME_ROOT
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.status = "idle"
        self.mode = "train"
        self.payload: dict[str, Any] | None = None
        self.runtime_dir: Path | None = None
        self.run_dir: Path | None = None
        self.command_preview = ""
        self.stdout_lines: list[str] = []
        self.stderr_lines: list[str] = []
        self.metric_plots: list[dict[str, str]] = []
        self.compare_figure_path = ""
        self.sample_items: list[str] = []
        self.sample_names: list[str] = []
        self.selected_sample = ""
        self.sample_source = ""
        self.checkpoints: list[dict[str, str]] = []
        self.metrics_rows: list[dict[str, Any]] = []
        self.history_length = 0
        self.latest_epoch = 0
        self.metric_version = ""
        self._last_rendered_metric_version = ""
        self._watch_thread: threading.Thread | None = None
        self._console_progress_active = {"stdout": False, "stderr": False}

    def _clear_locked(self) -> None:
        self.process = None
        self.status = "idle"
        self.mode = "train"
        self.payload = None
        self.runtime_dir = None
        self.run_dir = None
        self.command_preview = ""
        self.stdout_lines = []
        self.stderr_lines = []
        self.metric_plots = []
        self.compare_figure_path = ""
        self.sample_items = []
        self.sample_names = []
        self.selected_sample = ""
        self.sample_source = ""
        self.checkpoints = []
        self.metrics_rows = []
        self.history_length = 0
        self.latest_epoch = 0
        self.metric_version = ""
        self._last_rendered_metric_version = ""
        self._watch_thread = None
        self._console_progress_active = {"stdout": False, "stderr": False}

    @staticmethod
    def _is_progress_line(line: str) -> bool:
        return bool(SURROGATE_PROGRESS_RE.match(line))

    def _append_stream_line_locked(self, lines: list[str], line: str, *, replace_progress: bool = False) -> None:
        if (replace_progress or self._is_progress_line(line)) and lines and self._is_progress_line(lines[-1]):
            lines[-1] = line
        else:
            lines.append(line)
        del lines[:-240]

    @staticmethod
    def _should_ignore_log_line(line: str) -> bool:
        return any(pattern.search(line) for pattern in SURROGATE_IGNORED_LOG_PATTERNS)

    def _print_stream_line(self, line: str, stream_name: str) -> None:
        prefix = "[surrogate]" if stream_name == "stdout" else "[surrogate][stderr]"
        progress_active = self._console_progress_active.get(stream_name, False)
        if self._is_progress_line(line):
            if sys.stdout.isatty():
                print(f"\r{prefix} {line}", end="", flush=True)
            else:
                print(f"{prefix} {line}", flush=True)
            self._console_progress_active[stream_name] = True
            return
        if progress_active and sys.stdout.isatty():
            print("", flush=True)
        print(f"{prefix} {line}", flush=True)
        self._console_progress_active[stream_name] = False

    def _terminate_locked(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except Exception:
            self.process.terminate()
        try:
            self.process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except Exception:
                self.process.kill()

    def _handle_stream_fragment(self, lines: list[str], stream_name: str, raw_text: str, *, is_progress: bool) -> None:
        line = ANSI_ESCAPE_RE.sub("", raw_text).replace("\r", "").strip()
        if not line:
            return
        if self._should_ignore_log_line(line):
            return
        with self.lock:
            self._append_stream_line_locked(lines, line, replace_progress=is_progress)
            self._print_stream_line(line, stream_name)

    def _consume_pipe(self, pipe: Any, lines: list[str], stream_name: str) -> None:
        if pipe is None:
            return
        buffer = ""
        last_sep_was_cr = False
        try:
            while True:
                chunk = pipe.read(4096)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk
                for ch in text:
                    if ch == "\r":
                        self._handle_stream_fragment(lines, stream_name, buffer, is_progress=True)
                        buffer = ""
                        last_sep_was_cr = True
                        continue
                    if ch == "\n":
                        if buffer:
                            self._handle_stream_fragment(lines, stream_name, buffer, is_progress=False)
                        elif last_sep_was_cr:
                            last_sep_was_cr = False
                            continue
                        buffer = ""
                        last_sep_was_cr = False
                        continue
                    buffer += ch
                    last_sep_was_cr = False
            if buffer:
                self._handle_stream_fragment(lines, stream_name, buffer, is_progress=False)
        finally:
            if self._console_progress_active.get(stream_name, False) and sys.stdout.isatty():
                print("", flush=True)
            self._console_progress_active[stream_name] = False
            pipe.close()

    def _start_process_locked(self, command: list[str], cwd: Path) -> None:
        self.process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
            start_new_session=True,
            env=_build_surrogate_process_env(),
        )
        threading.Thread(target=self._consume_pipe, args=(self.process.stdout, self.stdout_lines, "stdout"), daemon=True).start()
        threading.Thread(target=self._consume_pipe, args=(self.process.stderr, self.stderr_lines, "stderr"), daemon=True).start()
        self._watch_thread = threading.Thread(target=self._watch_process_loop, daemon=True)
        self._watch_thread.start()

    def _normalize_payload_paths(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        execution_target = str(normalized.get("executionTarget") or "local")
        normalized["executionTarget"] = execution_target
        normalized["projectRoot"] = str(resolve_surrogate_project_root(execution_target))
        normalized["dataRoot"] = resolve_surrogate_path(execution_target, str(normalized.get("dataRoot") or "dataset"))
        normalized["outputRoot"] = resolve_surrogate_output_root(
            execution_target,
            str(normalized.get("outputRoot") or ""),
        )
        normalized["csvFile"] = resolve_surrogate_path(
            execution_target,
            str(normalized.get("csvFile") or ""),
            allow_empty=True,
        )
        normalized["checkpointRunDir"] = resolve_surrogate_eval_run_dir(
            execution_target,
            str(normalized.get("checkpointRunDir") or ""),
        )
        checkpoints: list[dict[str, str]] = []
        for item in normalized.get("checkpoints") or []:
            checkpoints.append(
                {
                    "modelType": str(item["modelType"]),
                    "path": resolve_surrogate_path(execution_target, str(item["path"]), allow_empty=True),
                }
            )
        normalized["checkpoints"] = checkpoints
        return normalized

    def _watch_process_loop(self) -> None:
        while True:
            with self.lock:
                process = self.process
            if process is None:
                return
            code = process.poll()
            if code is None:
                time.sleep(0.5)
                continue
            with self.lock:
                if self.status != "stopped":
                    self.status = "done" if code == 0 else "failed"
                self._refresh_artifacts_locked()
                self._watch_thread = None
            return

    def start_train(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("A surrogate task is already active.")
            self._clear_locked()
            self.mode = "train"
            self.payload = self._normalize_payload_paths(payload)
            self.checkpoints = []
            self.run_dir = _surrogate_run_dir_from_payload(self.payload, increment_if_exists=True)
            self.runtime_dir = self.runtime_root / f"train_{self.payload['datasetKey']}_{self.payload['modelType']}_{int(time.time())}"
            self.runtime_dir.mkdir(parents=True, exist_ok=True)
            command = _surrogate_train_command(self.payload)
            self.command_preview = format_shell_command(command)
            self.stdout_lines = ["COMMAND:", *self.command_preview.splitlines()]
            self.status = "training"
            self._start_process_locked(command, PROJECT_ROOT)
            return {"status": self.status}

    def start_eval(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("A surrogate task is already active.")
            checkpoints = payload.get("checkpoints") or []
            if not checkpoints:
                raise RuntimeError("Checkpoint visualization requires at least one checkpoint.")
            self._clear_locked()
            self.mode = "eval"
            self.payload = self._normalize_payload_paths(payload)
            self.checkpoints = [{"modelType": str(item["modelType"]), "path": str(item["path"])} for item in self.payload.get("checkpoints") or []]
            self.runtime_dir = self.runtime_root / f"eval_{self.payload['datasetKey']}_{int(time.time())}"
            self.runtime_dir.mkdir(parents=True, exist_ok=True)
            sample_payload = resolve_surrogate_samples_payload(self.payload)
            self.sample_names = list(sample_payload["sampleNames"])
            self.sample_items = list(sample_payload["sampleItems"])
            self.sample_source = str(sample_payload["sampleSource"])
            self.selected_sample = str(sample_payload["selectedSample"])
            runner_payload = {
                **self.payload,
                "runtimeDir": str(self.runtime_dir),
                "checkpoints": self.checkpoints,
                "selectedSample": self.selected_sample,
            }
            payload_path = self.runtime_dir / "payload.json"
            payload_path.write_text(json.dumps(runner_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            command = [
                str(runtime_config.resolve_python_for_target(str(self.payload.get("executionTarget") or "local"), runtime_config.DEFAULT_PYTHON)),
                str(VIS_ROOT / "surrogate_eval_runner.py"),
                "--payload-json",
                str(payload_path),
            ]
            self.command_preview = format_shell_command(command)
            self.stdout_lines = ["COMMAND:", *self.command_preview.splitlines()]
            self.status = "evaluating"
            self._start_process_locked(command, PROJECT_ROOT)
            return {"status": self.status}

    def stop(self) -> dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self._terminate_locked()
                self.status = "stopped"
            return {"status": self.status}

    def reset(self) -> dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self._terminate_locked()
            self._clear_locked()
            return {"status": self.status}

    def _refresh_artifacts_locked(self) -> None:
        if self.mode == "train" and self.run_dir is not None:
            history_path = self.run_dir / "history.json"
            if history_path.exists() and self.runtime_dir is not None:
                try:
                    history = json.loads(history_path.read_text(encoding="utf-8"))
                except Exception:
                    history = []
                if isinstance(history, list):
                    rows = [row for row in history if isinstance(row, dict) and row.get("epoch") is not None]
                    new_history_length = len(rows)
                    new_latest_epoch = int(rows[-1]["epoch"]) if rows else 0
                else:
                    new_history_length = 0
                    new_latest_epoch = 0
                
                # 只在数据真正变化时才重新生成图片
                data_changed = (
                    new_history_length != self.history_length or 
                    new_latest_epoch != self.latest_epoch
                )
                
                self.history_length = new_history_length
                self.latest_epoch = new_latest_epoch
                
                if data_changed:
                    # 数据变化了，重新生成图片和 metric_version
                    stat = history_path.stat()
                    self.metric_version = f"{int(stat.st_mtime_ns)}:{self.history_length}:{self.latest_epoch}"
                    title_prefix = f"{SURROGATE_DATASET_LABELS.get(str(self.payload.get('datasetKey')), '')} / {SURROGATE_MODEL_LABELS.get(str(self.payload.get('modelType')), '')}"
                    self.metric_plots = _render_surrogate_metric_panels(history_path, self.runtime_dir / "metric_panels", title_prefix)
                    print(f"[surrogate] Metric data changed: epoch={self.latest_epoch}, history_length={self.history_length}, regenerated images, new metric_version={self.metric_version}")
                else:
                    print(f"[surrogate] Metric data unchanged: epoch={self.latest_epoch}, history_length={self.history_length}, keeping existing metric_version={self.metric_version}")
                # 如果数据没变，保持原来的 metric_version 和 metric_plots
                
            test_split_path = self.run_dir / "test_split.csv"
            if test_split_path.exists() and not self.sample_items and self.payload is not None:
                dataset_key = str(self.payload["datasetKey"])
                csv_file = str(test_split_path)
                try:
                    sample_names = _resolve_surrogate_sample_names(dataset_key, str(self.payload["dataRoot"]), csv_file)
                    self.sample_items = _format_surrogate_sample_items(dataset_key, sample_names)
                    self.sample_source = f"来自本次训练生成的 test_split.csv: {test_split_path}"
                except Exception:
                    pass
        if self.mode == "eval" and self.runtime_dir is not None:
            manifest_path = self.runtime_dir / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.sample_items = list(manifest.get("sampleItems") or self.sample_items)
                self.sample_names = list(manifest.get("sampleNames") or self.sample_names)
                self.sample_source = str(manifest.get("sampleSource") or self.sample_source)
                self.selected_sample = str(manifest.get("sampleName") or self.selected_sample)
                self.metrics_rows = list(manifest.get("metrics") or self.metrics_rows)
                compare_path = manifest.get("compareFigurePath")
                if compare_path:
                    self.compare_figure_path = str(compare_path)

    def get_state(self) -> dict[str, Any]:
        with self.lock:
            failure_detail = ""
            if self.process is not None and self.process.poll() is None:
                if self.mode == "train":
                    self.status = "training"
                elif self.mode == "eval":
                    self.status = "evaluating"
            self._refresh_artifacts_locked()
            payload = self.payload or {}
            run_dir_text = str(self.run_dir) if self.run_dir is not None else ""
            if self.mode == "eval" and self.runtime_dir is not None:
                run_dir_text = str(self.runtime_dir / "eval_runs")
            banner = (
                f"TRAIN | {SURROGATE_DATASET_LABELS.get(str(payload.get('datasetKey')), '')} / {SURROGATE_MODEL_LABELS.get(str(payload.get('modelType')), '')} 训练执行中。"
                if self.mode == "train"
                else f"EVAL | 正在生成 {len(self.checkpoints)} 个 checkpoint 的单样本可视化。"
            )
            if self.status == "done":
                banner = "TRAIN | 训练已完成，可查看 best checkpoint、split 与 metrics_summary。" if self.mode == "train" else "EVAL | 单样本可视化已完成，可查看多模型对比图与误差指标。"
            if self.status == "failed":
                for line in reversed(self.stderr_lines):
                    if line.strip():
                        failure_detail = line.strip()
                        break
                if not failure_detail:
                    for line in reversed(self.stdout_lines):
                        if line.strip() and not line.startswith("COMMAND:"):
                            failure_detail = line.strip()
                            break
                base_banner = (
                    "TRAIN | 训练执行失败，请查看页面日志栏或服务端控制台。"
                    if self.mode == "train"
                    else "EVAL | checkpoint 可视化失败，请查看页面日志栏或服务端控制台。"
                )
                banner = f"{base_banner} 最后一条错误: {failure_detail}" if failure_detail else base_banner
            if self.status == "stopped":
                banner = "任务已停止。"
            return {
                "status": self.status,
                "mode": self.mode,
                "runDir": run_dir_text,
                "commandPreview": self.command_preview,
                "logs": {"stdout": self.stdout_lines[-160:], "stderr": self.stderr_lines[-80:]},
                "metricPlots": [
                    {
                        "title": item["title"],
                        "url": str(Path(item["path"]).resolve().relative_to(VIS_ROOT)).replace(os.sep, "/"),
                    }
                    for item in self.metric_plots
                    if Path(item["path"]).exists()
                ],
                "compareFigureUrl": (
                    str(Path(self.compare_figure_path).resolve().relative_to(VIS_ROOT)).replace(os.sep, "/")
                    if self.compare_figure_path and Path(self.compare_figure_path).exists()
                    else ""
                ),
                "sampleItems": list(self.sample_items),
                "sampleNames": list(self.sample_names),
                "selectedSample": self.selected_sample,
                "sampleSource": self.sample_source,
                "checkpoints": list(self.checkpoints),
                "metricsRows": list(self.metrics_rows),
                "banner": banner,
                "activeConfig": payload,
                "historyLength": self.history_length,
                "latestEpoch": self.latest_epoch,
                "metricVersion": self.metric_version,
            }


class WebRuntimeManager:
    def __init__(self) -> None:
        self.runtime_paths = image_jobs.VisualizationRuntimePaths.from_root(VIS_ROOT)
        self.runtime_paths.ensure()
        self.python_executable = runtime_config.DEFAULT_PYTHON
        self.render_device = os.environ.get("VIS_RENDER_DEVICE", "cpu").strip() or "cpu"
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.status = "idle"
        self.current_traj_path: Path | None = None
        self.current_map_path: Path | None = None
        self.current_eval_model = runtime_config.DEFAULT_EVAL_MODEL
        self.current_image_signature = ""
        self.last_stdout_lines: list[str] = []
        self.last_stderr_lines: list[str] = []
        self.last_image_error = ""
        self.last_dashboard_state: dict[str, Any] | None = None
        self.last_rendered_dashboard_state: dict[str, Any] | None = None
        self.last_preview_payload: dict[str, Any] | None = None
        self.current_run_token = ""
        self.display_step_index = -1
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._state_watch_thread: threading.Thread | None = None
        self._image_refresh_thread: threading.Thread | None = None
        self._pending_image_states: deque[dict[str, Any]] = deque()
        self.current_sync_dir: Path | None = None

    def _clear_run_artifacts_locked(self) -> None:
        self.current_traj_path = None
        self.current_map_path = None
        self.current_eval_model = runtime_config.DEFAULT_EVAL_MODEL
        self.current_image_signature = ""
        self.last_stdout_lines = []
        self.last_stderr_lines = []
        self.last_image_error = ""
        self.last_dashboard_state = None
        self.last_rendered_dashboard_state = None
        self.current_run_token = ""
        self.display_step_index = -1
        self._pending_image_states.clear()
        self.current_sync_dir = None
        stale_paths = [
            self.runtime_paths.roi_root / "current_roi.png",
            self.runtime_paths.pred_root / "current_pred" / "latest_pred.png",
            self.runtime_paths.heatmap_root / "current_capacity.png",
            self.runtime_paths.trend_root / "current_metric_trend.png",
            self.runtime_paths.sites_root / "current_sites.json",
        ]
        for stale_path in stale_paths:
            try:
                if stale_path.exists():
                    stale_path.unlink()
            except OSError:
                pass

    def _clear_runtime_state_locked(self) -> None:
        self.process = None
        self.status = "idle"
        self.last_preview_payload = None
        self._stdout_thread = None
        self._stderr_thread = None
        self._state_watch_thread = None
        self._image_refresh_thread = None
        self._clear_run_artifacts_locked()

    def _terminate_process_locked(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.last_stdout_lines.append("STOP: terminate requested")
        print("STOP: terminate requested", flush=True)
        self.process.terminate()
        try:
            self.process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            self.last_stdout_lines.append("STOP: kill requested")
            print("STOP: kill requested", flush=True)
            self.process.kill()
            try:
                self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass

    def get_options(self) -> dict[str, Any]:
        return build_options_payload(list_map_entries())

    def _build_preview_payload(self, request_key: str) -> dict[str, Any]:
        from ReAct.run_access_point_decision import infer_request_overrides

        request_path = runtime_config.request_path_for_key(request_key)
        request_text = request_path.read_text(encoding="utf-8").strip()
        goal, constraints, objective = infer_request_overrides(request_text)
        summary = "已完成需求文件读取，可进行需求结构化。"
        if goal or constraints:
            summary = "需求结构化完成，可进行初始站点部署。"
        return {
            "request_key": request_key,
            "request_text": request_text,
            "goal": goal,
            "goal_human_readable": trajectory_parser._describe_goal(goal, constraints),
            "constraints": constraints,
            "objective": objective,
            "diagnosis": summary,
        }

    def preview_request(self, request_key: str) -> dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self._terminate_process_locked()
            self.status = "idle"
            self._clear_run_artifacts_locked()
            payload = self._build_preview_payload(request_key)
            self.last_preview_payload = payload
            return payload

    def start_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("A decision run is already active.")
            map_path = Path(payload["map_path"]).resolve()
            request_key = str(payload["request_key"])
            planner = str(payload["planner"])
            eval_model = str(payload.get("eval_model") or runtime_config.DEFAULT_EVAL_MODEL)
            execution_target = str(payload.get("execution_target") or "local")
            follow_up_key = str(payload.get("follow_up_method") or runtime_config.DEFAULT_FOLLOW_UP_METHOD)
            follow_up_option = runtime_config.follow_up_option_for_key(follow_up_key)
            follow_up_planner = str(follow_up_option.get("planner") or "")
            planner = str(payload.get("planner") or planner)
            if follow_up_planner:
                planner = follow_up_planner
            follow_up_method = str(follow_up_option["llm_mode"])
            validation_errors = runtime_config.validate_runtime_inputs(
                map_path=map_path,
                request_key=request_key,
                planner=planner,
                eval_model=eval_model,
            )
            if validation_errors:
                raise RuntimeError(" | ".join(validation_errors))
            traj_id = f"{map_path.stem}__{request_key}__{planner}__{follow_up_method}"
            self.current_traj_path = self.runtime_paths.traj_root / f"{traj_id}.json"
            if self.current_traj_path.exists():
                self.current_traj_path.unlink()
            self.current_map_path = map_path
            self.current_eval_model = eval_model
            self.current_image_signature = ""
            self.current_run_token = traj_id
            self.display_step_index = -1
            self.current_sync_dir = self.runtime_paths.runtime_root / "sync" / traj_id
            if self.current_sync_dir.exists():
                for stale_file in self.current_sync_dir.glob("*"):
                    try:
                        stale_file.unlink()
                    except OSError:
                        pass
            self.current_sync_dir.mkdir(parents=True, exist_ok=True)
            self.python_executable = runtime_config.resolve_python_for_target(
                execution_target,
                current_python=runtime_config.DEFAULT_PYTHON,
            )
            self.last_stdout_lines = []
            self.last_stderr_lines = []
            self.last_image_error = ""
            self.last_dashboard_state = None
            self.last_rendered_dashboard_state = None
            self.last_preview_payload = self._build_preview_payload(request_key)
            for stale_image in (
                self.runtime_paths.roi_root / "current_roi.png",
                self.runtime_paths.pred_root / "current_pred" / "latest_pred.png",
                self.runtime_paths.heatmap_root / "current_capacity.png",
                self.runtime_paths.trend_root / "current_metric_trend.png",
            ):
                try:
                    if stale_image.exists():
                        stale_image.unlink()
                except OSError:
                    pass
            command = runtime_config.build_decision_command(
                python_executable=self.python_executable,
                project_root=PROJECT_ROOT,
                map_path=map_path,
                request_key=request_key,
                planner=planner,
                eval_model=eval_model,
                follow_up_method=follow_up_method,
                max_steps=int(payload.get("max_steps", runtime_config.DEFAULT_MAX_STEPS)),
                candidate_sample=int(payload.get("candidate_sample", runtime_config.DEFAULT_CANDIDATE_SAMPLE)),
                llm_top_k_candidates=int(payload.get("llm_top_k_candidates", runtime_config.DEFAULT_LLM_TOP_K)),
                heuristic_search_budget=int(
                    payload.get("heuristic_search_budget", runtime_config.DEFAULT_HEURISTIC_SEARCH_BUDGET)
                ),
                heuristic_online_candidate_sample=int(
                    payload.get(
                        "heuristic_online_candidate_sample",
                        runtime_config.DEFAULT_HEURISTIC_ONLINE_CANDIDATE_SAMPLE,
                    )
                ),
                traj_dir=self.runtime_paths.traj_root,
                traj_id=traj_id,
                visualization_sync_dir=self.current_sync_dir,
                visualization_sync_timeout_sec=0.0,
            )
            env = os.environ.copy()
            env["VISUALIZATION_PYTHON"] = str(self.python_executable)
            env["PYTHONUNBUFFERED"] = "1"
            command_text = " ".join(command)
            self.last_stdout_lines.append("COMMAND: " + command_text)
            self.last_stdout_lines.append(f"START: planner={planner} map={map_path.stem} request={request_key}")
            self.last_stdout_lines.append(f"RENDER: device={self.render_device}")
            self.last_stdout_lines.append(f"SYNC: dir={self.current_sync_dir}")
            print(f"[visualization] COMMAND: {command_text}", flush=True)
            print(f"START: planner={planner} map={map_path.stem} request={request_key}", flush=True)
            print(f"RENDER: device={self.render_device}", flush=True)
            print(f"SYNC: dir={self.current_sync_dir}", flush=True)
            try:
                preload_predictor(self.current_eval_model, self.render_device)
            except Exception as exc:
                self.last_stderr_lines.append(f"render warmup failed: {exc}")
                del self.last_stderr_lines[:-200]
            self.process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
            self.status = "running"
            self._stdout_thread = threading.Thread(target=self._consume_pipe, args=(self.process.stdout, self.last_stdout_lines), daemon=True)
            self._stderr_thread = threading.Thread(target=self._consume_pipe, args=(self.process.stderr, self.last_stderr_lines), daemon=True)
            self._stdout_thread.start()
            self._stderr_thread.start()
            self._state_watch_thread = threading.Thread(target=self._run_state_watch_loop, daemon=True)
            self._state_watch_thread.start()
            return {"status": self.status, "traj_path": str(self.current_traj_path)}

    def stop_run(self) -> dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self._terminate_process_locked()
                self.status = "stopped"
            return {"status": self.status}

    def reset_run(self) -> dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self._terminate_process_locked()
            self._clear_runtime_state_locked()
            return {"status": self.status}

    def get_state(self) -> dict[str, Any]:
        with self.lock:
            self._refresh_process_status()
            dashboard_state = self._current_dashboard_state_locked()
            process_running = self.process is not None and self.process.poll() is None
            return build_state_payload(
                status=self.status,
                traj_path=self.current_traj_path,
                dashboard_state=dashboard_state,
                runtime_paths=self.runtime_paths,
                stdout_lines=self.last_stdout_lines[-120:],
                stderr_lines=self.last_stderr_lines[-120:],
                image_error=self.last_image_error,
                image_version=(
                    f"{self.current_run_token}-"
                    f"{dashboard_state.get('current_step', 0)}-"
                    f"{len(dashboard_state.get('sites', []) or [])}-"
                    f"{self.current_eval_model}"
                    if dashboard_state is not None
                    else self.current_run_token
                ),
                process_running=process_running,
            )

    def _current_dashboard_state_locked(self) -> dict[str, Any] | None:
        # 获取当前的 dashboard_state
        state = None
        if self.last_rendered_dashboard_state is not None:
            state = dict(self.last_rendered_dashboard_state)
        elif self.last_dashboard_state is not None:
            state = dict(self.last_dashboard_state)
        elif self.last_preview_payload is not None:
            state = {
                "current_step": 0,
                "current_ap_count": 0,
                "request_text": self.last_preview_payload.get("request_text", ""),
                "goal": self.last_preview_payload.get("goal", {}) or {},
                "goal_human_readable": self.last_preview_payload.get("goal_human_readable", ""),
                "constraints": self.last_preview_payload.get("constraints", {}) or {},
                "diagnosis": self.last_preview_payload.get("diagnosis", ""),
                "ok": False,
                "metrics": {"coverage": 0.0, "spectral_efficiency": 0.0, "redundancy_rate": 0.0},
                "sites": [],
                "table_rows": [],
                "metric_history": [],
                "method_labels": {"init": "TSPL", "follow_up": "LLM可解释性权重"},
                "flow": {
                    "select_request": "complete",
                    "request_structuring": "complete",
                    "initial_deployment": "current" if self.process is not None and self.process.poll() is None else "pending",
                    "decision_loop": "pending",
                    "decision_complete_arrow": "pending",
                    "complete": "pending",
                },
            }
        else:
            return None
        
        # 修复：如果已经渲染了 INIT (step=0) 的图片，强制更新 flow 状态
        if self.last_rendered_dashboard_state is not None and "flow" in state:
            current_step = int(state.get("current_step", 0) or 0)
            rendered_step = int(self.last_rendered_dashboard_state.get("current_step", -1) or -1)
            # 只要渲染过 step=0 的图片，就应该切换到 decision_loop 状态
            if rendered_step >= 0:
                state["flow"] = dict(state["flow"])
                state["flow"]["initial_deployment"] = "complete"
                state["flow"]["decision_loop"] = "current"
        
        return state

    def _advance_dashboard_state_once(self) -> bool:
        with self.lock:
            self._refresh_process_status()
            traj_path = self.current_traj_path
            process_running = self.process is not None and self.process.poll() is None
            display_step_index = self.display_step_index
            # 判断初始化站点是否已经渲染完成
            has_initial_render = (
                self.last_rendered_dashboard_state is not None 
                and int(self.last_rendered_dashboard_state.get("current_step", -1) or -1) >= 0
            )
        if traj_path is None:
            return False

        dashboard_state: dict[str, Any] | None = None
        next_step_index: int | None = None
        try:
            latest_state = trajectory_parser.build_dashboard_state(
                traj_path,
                process_running=process_running,
                has_initial_render=has_initial_render,
            )
        except Exception:
            latest_state = None
        if latest_state is not None:
            latest_available_step = int(latest_state.get("latest_available_step", 0) or 0)
            if display_step_index < 0:
                next_step_index = 0
            elif display_step_index < latest_available_step:
                next_step_index = display_step_index + 1
            if next_step_index is not None:
                try:
                    dashboard_state = trajectory_parser.build_dashboard_state(
                        traj_path,
                        process_running=process_running,
                        observation_index=next_step_index,
                        has_initial_render=has_initial_render,
                    )
                except Exception:
                    dashboard_state = None
            elif self.last_dashboard_state is None:
                dashboard_state = latest_state
                next_step_index = latest_available_step

        if dashboard_state is None:
            return False
        with self.lock:
            self.last_dashboard_state = dashboard_state
            if next_step_index is not None:
                self.display_step_index = next_step_index
        self._schedule_image_refresh(dashboard_state)
        return True

    def _run_state_watch_loop(self) -> None:
        import time

        while True:
            self._advance_dashboard_state_once()

            with self.lock:
                caught_up = self._display_is_caught_up_locked()
                should_exit = (
                    (self.process is None or self.process.poll() is not None)
                    and caught_up
                    and not self._pending_image_states
                    and (self._image_refresh_thread is None or not self._image_refresh_thread.is_alive())
                    and self.status in {"done", "failed", "stopped", "idle"}
                )
                if should_exit:
                    self._state_watch_thread = None
                    return
            time.sleep(0.2)

    def _display_is_caught_up_locked(self) -> bool:
        traj_path = self.current_traj_path
        if traj_path is None or not traj_path.exists():
            return True
        try:
            latest_state = trajectory_parser.build_dashboard_state(
                traj_path,
                process_running=self.process is not None and self.process.poll() is None,
            )
        except Exception:
            return False
        latest_available_step = int(latest_state.get("latest_available_step", 0) or 0)
        return self.display_step_index >= latest_available_step

    def _schedule_image_refresh(self, dashboard_state: dict[str, Any]) -> None:
        with self.lock:
            pending_state = dict(dashboard_state)
            pending_signature = self._image_signature_for_state(pending_state)
            if self._pending_image_states:
                last_signature = self._image_signature_for_state(self._pending_image_states[-1])
                if pending_signature == last_signature:
                    return
                self._pending_image_states.clear()
            self._pending_image_states.append(pending_state)
            if self._image_refresh_thread is not None and self._image_refresh_thread.is_alive():
                return
            self._image_refresh_thread = threading.Thread(target=self._run_image_refresh_loop, daemon=True)
            self._image_refresh_thread.start()

    def _run_image_refresh_loop(self) -> None:
        while True:
            with self.lock:
                dashboard_state = self._pending_image_states.popleft() if self._pending_image_states else None
            if dashboard_state is None:
                with self.lock:
                    self._image_refresh_thread = None
                return
            self._refresh_images_if_needed(dashboard_state)

    def _refresh_process_status(self) -> None:
        if self.process is None:
            return
        code = self.process.poll()
        if code is None:
            self.status = "running"
            return
        if self.status == "stopped":
            return
        self.status = "done" if code == 0 else "failed"

    def _refresh_images_if_needed(self, dashboard_state: dict[str, Any]) -> bool:
        if self.current_map_path is None:
            return False
        signature = self._image_signature_for_state(dashboard_state)
        if signature == self.current_image_signature:
            return True
        self.last_image_error = ""
        sites_path = self.runtime_paths.sites_root / "current_sites.json"
        roi_output = self.runtime_paths.roi_root / "current_roi.png"
        pred_output_dir = self.runtime_paths.pred_root / "current_pred"
        capacity_output = self.runtime_paths.heatmap_root / "current_capacity.png"
        trend_output = self.runtime_paths.trend_root / "current_metric_trend.png"
        image_jobs.write_sites_payload(sites_path, dashboard_state.get("sites", []))
        try:
            render_roi_coverage(
                map_path=self.current_map_path,
                sites=dashboard_state.get("sites", []),
                output_path=roi_output,
                eval_model=self.current_eval_model,
                render_device=self.render_device,
            )
            pred_output = pred_output_dir / "latest_pred.png"
            render_deployment_prediction(
                map_path=self.current_map_path,
                sites=dashboard_state.get("sites", []),
                output_path=pred_output,
                eval_model=self.current_eval_model,
                render_device=self.render_device,
            )
            render_capacity_heatmap(
                map_path=self.current_map_path,
                sites=dashboard_state.get("sites", []),
                output_path=capacity_output,
                eval_model=self.current_eval_model,
                render_device=self.render_device,
            )
            render_metric_history_trend(
                dashboard_state.get("metric_history", []) or [],
                trend_output,
            )
            self.last_stdout_lines.append(f"saved_roi={roi_output.resolve()}")
            self.last_stdout_lines.append(f"saved_pred={pred_output.resolve()}")
            self.last_stdout_lines.append(f"saved_capacity={capacity_output.resolve()}")
            self.last_stdout_lines.append(f"saved_trend={trend_output.resolve()}")
            self.last_rendered_dashboard_state = dict(dashboard_state)
            self._write_visualization_ack(int(dashboard_state.get("current_step", 0) or 0))
            print(f"saved_roi={roi_output.resolve()}", flush=True)
            print(f"saved_pred={pred_output.resolve()}", flush=True)
            print(f"saved_capacity={capacity_output.resolve()}", flush=True)
            print(f"saved_trend={trend_output.resolve()}", flush=True)
            del self.last_stdout_lines[:-200]
            self.current_image_signature = signature
            return True
        except Exception as exc:
            self.last_image_error = str(exc)
            self.current_image_signature = ""
            self.last_stdout_lines.append("IMAGE: refresh failed")
            print("IMAGE: refresh failed", flush=True)
            self.last_stderr_lines.append(f"image refresh failed: {exc}")
            del self.last_stderr_lines[:-200]
            return False

    def _image_signature_for_state(self, dashboard_state: dict[str, Any]) -> str:
        return repr(
            (
                str(self.current_map_path),
                self.current_eval_model,
                int(dashboard_state.get("current_step", 0) or 0),
                dashboard_state.get("sites", []),
            )
        )

    def _write_visualization_ack(self, observation_index: int) -> None:
        sync_dir = self.current_sync_dir
        if sync_dir is None:
            return
        sync_dir.mkdir(parents=True, exist_ok=True)
        ack_path = sync_dir / f"step_{int(observation_index):04d}.done"
        ack_path.write_text("ok\n", encoding="utf-8")

    @staticmethod
    def _consume_pipe(pipe: Any, lines: list[str]) -> None:
        if pipe is None:
            return
        try:
            for raw_line in pipe:
                line = ANSI_ESCAPE_RE.sub("", raw_line).strip()
                if line:
                    lines.append(line)
                    del lines[:-200]
                    print(line, flush=True)
        finally:
            pipe.close()
