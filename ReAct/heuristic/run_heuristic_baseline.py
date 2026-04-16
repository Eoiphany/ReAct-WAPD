"""
用途:
  启发式基线规划器。思路对齐原始 autobs/react_loop.py：每一步只评估“新增一个站点”带来的覆盖提升，选择当前最优的新增站点。

示例命令:
  无。该文件主要供主入口导入；也可单独运行做一步候选导出。

参数说明:
  plan_action_heuristic(env, candidates, sample_k, seed, objective=None): 输出一步启发式动作。
  --city-map-path: 单独运行时输入地图。
  --candidate-sample: 单独运行时采样候选数量。
  --seed: 随机种子。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np

from env_utils import load_map_normalized
from radiomap_env import RadioMapEnv, build_candidates


def sample_candidates(candidates: List[Dict[str, Any]], k: int, seed: int) -> List[Dict[str, Any]]:
    if k <= 0 or k >= len(candidates):
        return candidates
    rng = random.Random(seed)
    return rng.sample(candidates, k)


def _score_state(
    coverage: float,
    capacity: float,
    site_count: int,
    coverage_target: Optional[float],
    capacity_target: Optional[float],
    objective: Optional[Dict[str, float]] = None,
) -> float:
    objective = objective or {}
    w_cov = float(objective.get("w_cov", 1.0))
    w_cap = float(objective.get("w_cap", 1.0))
    w_sites = float(objective.get("w_sites", 0.0))
    penalty_cov = 0.0 if coverage_target is None else max(0.0, float(coverage_target) - float(coverage))
    penalty_cap = 0.0 if capacity_target is None else max(0.0, float(capacity_target) - float(capacity))
    return (
        float(coverage)
        + float(capacity)
        + w_cov * float(coverage)
        + w_cap * float(capacity)
        - w_cov * penalty_cov
        - w_cap * penalty_cap
        - w_sites * float(site_count)
    )


def plan_action_heuristic(
    env: RadioMapEnv,
    candidates: List[Dict[str, Any]],
    sample_k: int,
    seed: int,
    objective: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    sampled = sample_candidates(candidates, sample_k, seed)
    if not sampled:
        return {"name": "Propose", "args": {"sites": [{"row": 0, "col": 0, "z_m": 3.0}], "mode": "add"}}

    best = None
    best_score = -1e18
    current_locs = list(env.tx_locs)
    current_set = {(int(r), int(c)) for r, c in current_locs}
    coverage_target = env.goal.get("targets", {}).get("coverage_pct")
    capacity_target = env.goal.get("targets", {}).get("capacity")
    site_limit = env.constraints.get("site_limit")

    for cand in sampled:
        rc = (int(cand["row"]), int(cand["col"]))
        if rc in current_set:
            continue
        if site_limit is not None and len(current_locs) + 1 > int(site_limit):
            continue
        env.tx_locs = current_locs + [rc]
        metrics = env._evaluate()
        score = _score_state(
            coverage=metrics.coverage,
            capacity=metrics.capacity,
            site_count=len(env.tx_locs),
            coverage_target=float(coverage_target) if coverage_target is not None else None,
            capacity_target=float(capacity_target) if capacity_target is not None else None,
            objective=objective,
        )
        if score > best_score:
            best_score = score
            best = cand

    env.tx_locs = current_locs
    if best is None:
        best = sampled[0]
    return {
        "name": "Propose",
        "args": {"sites": [{"row": int(best["row"]), "col": int(best["col"]), "z_m": float(best["z_m"])}], "mode": "add"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-map-path", required=True)
    parser.add_argument("--candidate-sample", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pixel_map = load_map_normalized(args.city_map_path)
    env = RadioMapEnv(
        city_map_path=args.city_map_path,
        goal={"primary": "maximize_coverage", "targets": {"coverage_pct": 0.95}},
        constraints={"site_limit": 20},
        candidate_sample=args.candidate_sample,
        seed=args.seed,
    )
    candidates = build_candidates(pixel_map)
    action = plan_action_heuristic(env, candidates, args.candidate_sample, args.seed)
    print(json.dumps(action, ensure_ascii=False))


if __name__ == "__main__":
    main()
