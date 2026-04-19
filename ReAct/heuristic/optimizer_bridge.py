"""
用途:
  统一桥接 `paper_experiment/Heuristic` 下的多种启发式算法。先调用对应 `run_*.py` 生成目标布局，再把目标布局转换成闭环环境里的单步动作。

示例命令:
  无。该文件由 ReAct 主入口内部调用。

参数说明:
  solve_target_layout(...): 运行外部启发式脚本并返回目标布局。
  next_action_from_target_layout(env, target_layout): 把目标布局转成下一步 Propose/Refine/Finish 动作。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
HEURISTIC_ROOT = ROOT_DIR.parent / "Heuristic"
CONFIG = yaml.safe_load((ROOT_DIR / "base_config.yaml").read_text(encoding="utf-8")) or {}
MODEL_CFG = CONFIG.get("surrogate_models", {}) if isinstance(CONFIG, dict) else {}


SCRIPT_MAP = {
    "heuristic": "run_greedy.py",
    "heuristic_greedy": "run_greedy.py",
    "heuristic_sa": "run_sa.py",
    "heuristic_ga": "run_ga.py",
    "heuristic_pso": "run_pso.py",
    "heuristic_bruteforce": "run_bruteforce.py",
    "heuristic_full_enum": "run_full_enumeration.py",
}


def _resolve_model_artifacts(eval_model: str) -> tuple[str, str]:
    if eval_model == "proxy":
        raise ValueError("External Heuristic planners only support pmnet or rmnet, not proxy.")
    if eval_model not in MODEL_CFG:
        raise ValueError(f"Unsupported eval_model: {eval_model}")
    cfg = MODEL_CFG[eval_model]
    model_path = str((ROOT_DIR / cfg["weights_path"]).resolve())
    return model_path, str(eval_model)


def _target_site_count(goal: Dict[str, Any], constraints: Dict[str, Any], fallback: int) -> int:
    site_exact = constraints.get("site_exact")
    site_limit = constraints.get("site_limit")
    if site_exact is not None:
        return max(1, int(site_exact))
    if site_limit is not None:
        return max(1, int(site_limit))
    return max(1, int(fallback))


def solve_target_layout(
    planner_name: str,
    height_map_path: str,
    goal: Dict[str, Any],
    constraints: Dict[str, Any],
    eval_model: str,
    output_dir: Path,
    fallback_k: int,
    max_evals: int = 200,
    candidate_stride: int = 8,
    candidate_limit: int = 500,
    device: str = "mps",
) -> tuple[list[tuple[int, int]], dict]:
    script_name = SCRIPT_MAP.get(planner_name)
    if script_name is None:
        raise ValueError(f"Unsupported heuristic planner: {planner_name}")
    script_path = HEURISTIC_ROOT / script_name
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    coverage_target = goal.get("targets", {}).get("coverage_pct")
    capacity_target = goal.get("targets", {}).get("capacity")
    coverage_target = 0.95 if coverage_target is None else float(coverage_target)
    capacity_target = 0.0 if capacity_target is None else float(capacity_target)
    k_max = _target_site_count(goal, constraints, fallback=fallback_k)
    model_path, network_type = _resolve_model_artifacts(eval_model)

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script_path),
        "--height-map",
        str(Path(height_map_path).resolve()),
        "--k-max",
        str(k_max),
        "--coverage-target",
        str(coverage_target),
        "--spectral-efficiency-target",
        str(capacity_target),
        "--model-path",
        model_path,
        "--network-type",
        network_type,
        "--output-dir",
        str(output_dir),
        "--device",
        device,
    ]
    if planner_name != "heuristic_full_enum":
        cmd.extend(["--max-evals", str(max_evals)])
    if planner_name in {"heuristic_greedy", "heuristic", "heuristic_bruteforce"}:
        cmd.extend(["--candidate-stride", str(candidate_stride), "--candidate-limit", str(candidate_limit)])

    subprocess.run(cmd, check=True)

    layout_path = output_dir / "best_layout.npy"
    metrics_path = output_dir / "best_metrics.json"
    positions_xy = np.load(layout_path)
    positions_rc: list[tuple[int, int]] = []
    for x, y in positions_xy:
        row = int(round(float(y)))
        col = int(round(float(x)))
        positions_rc.append((row, col))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return positions_rc, metrics


def next_action_from_target_layout(env, target_layout: list[tuple[int, int]]) -> Dict[str, Any]:
    current_locs = list(env.tx_locs)
    current_set = {(int(r), int(c)) for r, c in current_locs}
    target_set = {(int(r), int(c)) for r, c in target_layout}

    missing = [site for site in target_layout if (int(site[0]), int(site[1])) not in current_set]
    extra = [site for site in current_locs if (int(site[0]), int(site[1])) not in target_set]

    if extra and missing:
        extra_site = extra[0]
        extra_idx = current_locs.index(extra_site)
        row, col = missing[0]
        return {"name": "Refine", "args": {"rule_or_delta": {"op": "move", "id": int(extra_idx), "row": int(row), "col": int(col)}}}
    if missing:
        row, col = missing[0]
        z_m = float(env.pixel_map[int(row), int(col)] * (19.8 - 6.6) + 6.6 + 3.0)
        return {"name": "Propose", "args": {"sites": [{"row": int(row), "col": int(col), "z_m": z_m}], "mode": "add"}}
    if extra:
        extra_site = extra[0]
        extra_idx = current_locs.index(extra_site)
        return {"name": "Refine", "args": {"rule_or_delta": {"op": "remove", "id": int(extra_idx)}}}

    metrics = env._evaluate()
    return {
        "name": "Finish",
        "args": {
            "final_site_set": [
                {"row": int(r), "col": int(c), "z_m": float(env.pixel_map[int(r), int(c)] * (19.8 - 6.6) + 6.6 + 3.0)}
                for r, c in current_locs
            ],
            "metrics": {"coverage": float(metrics.coverage), "capacity": float(metrics.capacity)},
        },
    }
