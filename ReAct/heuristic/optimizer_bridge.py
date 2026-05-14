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
import re
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
    "heuristic_candidate_enum": "run_candidate_enumeration.py",
    "heuristic_exhaustive": "run_exhaustive_search.py",
    "heuristic_bruteforce": "run_candidate_enumeration.py",
    "heuristic_full_enum": "run_exhaustive_search.py",
}

PLANNER_ALIASES = {
    "heuristic_bruteforce": "heuristic_candidate_enum",
    "heuristic_full_enum": "heuristic_exhaustive",
}


def _resolve_model_artifacts(eval_model: str, eval_model_path: str = "") -> tuple[str, str]:
    if eval_model == "proxy":
        raise ValueError("External Heuristic planners only support pmnet or rmnet, not proxy.")
    if eval_model not in MODEL_CFG:
        raise ValueError(f"Unsupported eval_model: {eval_model}")
    if eval_model_path:
        model_path = str(Path(eval_model_path).expanduser().resolve())
    else:
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


def _resolve_search_budget(requested_max_evals: int) -> int:
    return max(1, int(requested_max_evals))


def _load_cached_target_layout(
    output_dir: Path,
    max_evals: int,
) -> tuple[list[tuple[int, int]], dict] | None:
    layout_path = output_dir / "best_layout.npy"
    metrics_path = output_dir / "best_metrics.json"
    if not layout_path.exists():
        return None

    positions_xy = np.load(layout_path)
    positions_rc: list[tuple[int, int]] = []
    for x, y in positions_xy:
        row = int(round(float(y)))
        col = int(round(float(x)))
        positions_rc.append((row, col))

    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    if isinstance(metrics, dict):
        metrics["requested_max_evals"] = int(max_evals)
        metrics["effective_max_evals"] = int(metrics.get("effective_max_evals", metrics.get("requested_max_evals", max_evals)))
        metrics["reused_from_cache"] = True
    return positions_rc, metrics


def _extract_total_runtime_sec(stdout_text: str, stderr_text: str) -> float | None:
    merged = "\n".join(part for part in [(stdout_text or "").strip(), (stderr_text or "").strip()] if part)
    if not merged:
        return None
    match = re.search(r"total_runtime_sec:\s*([0-9]+(?:\.[0-9]+)?)", merged)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def solve_target_layout(
    planner_name: str,
    height_map_path: str,
    goal: Dict[str, Any],
    constraints: Dict[str, Any],
    eval_model: str,
    output_dir: Path,
    fallback_k: int,
    eval_model_path: str = "",
    max_evals: int = 100,
    candidate_stride: int = 12,
    candidate_limit: int = 256,
    device: str = "mps",
    use_cache: bool = True,
) -> tuple[list[tuple[int, int]], dict]:
    planner_name = PLANNER_ALIASES.get(planner_name, planner_name)
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
    effective_max_evals = _resolve_search_budget(max_evals)
    model_path, network_type = _resolve_model_artifacts(eval_model, eval_model_path=eval_model_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    if use_cache:
        cached = _load_cached_target_layout(output_dir, max_evals=max_evals)
        if cached is not None:
            positions_rc, metrics = cached
            if isinstance(metrics, dict):
                metrics["target_site_count"] = int(k_max)
            return positions_rc, metrics

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
    if planner_name != "heuristic_exhaustive":
        cmd.extend(["--max-evals", str(effective_max_evals)])
    if planner_name in {"heuristic_greedy", "heuristic", "heuristic_candidate_enum"}:
        cmd.extend(["--candidate-stride", str(candidate_stride), "--candidate-limit", str(candidate_limit)])

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = stderr or stdout or f"exit_code={result.returncode}"
        raise RuntimeError(f"Heuristic planner failed ({planner_name}): {details}")

    layout_path = output_dir / "best_layout.npy"
    metrics_path = output_dir / "best_metrics.json"
    positions_xy = np.load(layout_path)
    positions_rc: list[tuple[int, int]] = []
    for x, y in positions_xy:
        row = int(round(float(y)))
        col = int(round(float(x)))
        positions_rc.append((row, col))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    if isinstance(metrics, dict):
        metrics["requested_max_evals"] = int(max_evals)
        metrics["effective_max_evals"] = int(effective_max_evals)
        metrics["target_site_count"] = int(k_max)
        runtime_sec = _extract_total_runtime_sec(result.stdout or "", result.stderr or "")
        if runtime_sec is not None:
            metrics["search_runtime_sec"] = float(runtime_sec)
        metrics["reused_from_cache"] = False
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return positions_rc, metrics


def _project_target_to_obs_candidates(
    target_site: tuple[int, int],
    current_set: set[tuple[int, int]],
    obs_payload: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(obs_payload, dict):
        return None
    candidates = obs_payload.get("candidates") or []
    if not isinstance(candidates, list):
        return None

    feasible: list[dict] = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        try:
            rc = (int(cand.get("row")), int(cand.get("col")))
        except Exception:
            continue
        if cand.get("feasible") is False:
            continue
        feasible.append(cand)
        if rc == (int(target_site[0]), int(target_site[1])):
            return cand

    if not feasible:
        return None

    target_row, target_col = int(target_site[0]), int(target_site[1])
    return min(
        feasible,
        key=lambda cand: (int(cand["row"]) - target_row) ** 2 + (int(cand["col"]) - target_col) ** 2,
    )


def _project_target_layout(
    target_layout: list[tuple[int, int]],
    current_set: set[tuple[int, int]],
    obs_payload: Optional[Dict[str, Any]],
) -> list[tuple[int, int]]:
    if not isinstance(obs_payload, dict):
        return [(int(r), int(c)) for r, c in target_layout]
    candidates = obs_payload.get("candidates") or []
    if not isinstance(candidates, list):
        return [(int(r), int(c)) for r, c in target_layout]

    feasible: list[dict] = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        try:
            rc = (int(cand.get("row")), int(cand.get("col")))
        except Exception:
            continue
        if cand.get("feasible") is False:
            continue
        feasible.append(cand)

    used: set[tuple[int, int]] = set()
    projected: list[tuple[int, int]] = []
    for target_row, target_col in target_layout:
        target_rc = (int(target_row), int(target_col))
        exact = None
        for cand in feasible:
            rc = (int(cand["row"]), int(cand["col"]))
            if rc in used:
                continue
            if rc == target_rc:
                exact = rc
                break
        if exact is not None:
            projected.append(exact)
            used.add(exact)
            continue

        nearest = None
        nearest_dist = None
        for cand in feasible:
            rc = (int(cand["row"]), int(cand["col"]))
            if rc in used:
                continue
            dist = (rc[0] - target_rc[0]) ** 2 + (rc[1] - target_rc[1]) ** 2
            if nearest is None or dist < nearest_dist:
                nearest = rc
                nearest_dist = dist
        if nearest is not None:
            projected.append(nearest)
            used.add(nearest)
        else:
            projected.append(target_rc)
    return projected


def next_action_from_target_layout(
    env,
    target_layout: list[tuple[int, int]],
    obs_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    current_locs = list(env.tx_locs)
    current_set = {(int(r), int(c)) for r, c in current_locs}
    projected_layout = _project_target_layout(target_layout, current_set, obs_payload)
    target_set = {(int(r), int(c)) for r, c in projected_layout}
    site_exact = env.constraints.get("site_exact")
    site_limit = env.constraints.get("site_limit")
    site_limit = None if site_limit is None else int(site_limit)

    missing = [site for site in projected_layout if (int(site[0]), int(site[1])) not in current_set]
    extra = [site for site in current_locs if (int(site[0]), int(site[1])) not in target_set]

    # 固定站点模式下，启发式回放只补点，不做 move/remove 修补；达到精确站点数后直接结束。
    if site_exact is not None:
        if len(current_locs) < int(site_exact) and missing:
            row, col = missing[0]
            z_m = float(env.pixel_map[int(row), int(col)] * (19.8 - 6.6) + 6.6 + 3.0)
            return {"name": "Propose", "args": {"sites": [{"row": int(row), "col": int(col), "z_m": z_m}], "mode": "add"}}

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

    # 非固定站点模式下，若当前站点数仍少于目标布局规模，且未触及 site_limit，
    # 则优先补点；这样可以避免把初始化站点先 move 掉导致站点数长期偏低。
    if missing and (site_limit is None or len(current_locs) < site_limit) and len(current_locs) < len(projected_layout):
        row, col = missing[0]
        z_m = float(env.pixel_map[int(row), int(col)] * (19.8 - 6.6) + 6.6 + 3.0)
        return {"name": "Propose", "args": {"sites": [{"row": int(row), "col": int(col), "z_m": z_m}], "mode": "add"}}

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
