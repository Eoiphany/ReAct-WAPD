"""
用途:
  汇总一组无线接入点决策轨迹，输出成功率、步数、覆盖率、容量、冗余率和站点数统计。

示例命令:
  python ReAct/evaluate_decision_trajectories.py --traj-dirs ReAct/outputs/trajs

参数说明:
  --traj-dirs: 一个或多个轨迹目录。
  --labels: 与 traj-dirs 对应的标签名。
  --by-request: 是否输出按 request 分组的统计。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default


def _iter_traj_files(traj_dir: str) -> Iterable[str]:
    for name in sorted(os.listdir(traj_dir)):
        if name.endswith(".json"):
            yield os.path.join(traj_dir, name)


def _parse_action_name(action_str: str) -> str:
    if not action_str or "DECIDE[" not in action_str:
        return "unknown"
    try:
        payload = action_str[action_str.find("DECIDE[") + 7 :].rstrip("]")
        data = json.loads(payload)
        return data.get("selected_action", {}).get("name", "unknown")
    except Exception:
        return "unknown"


def _request_id_from_path(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    if "__" in base:
        return base.split("__", 1)[1]
    return "unknown"


def _goal_constraints_ok(last_obs: Dict[str, Any]) -> bool:
    goal = last_obs.get("goal", {}) or {}
    constraints = last_obs.get("constraints", {}) or {}
    state = last_obs.get("state", {}) or {}
    metrics = state.get("last_metrics", {}) or {}

    cov = _safe_float(metrics.get("coverage"), default=0.0)
    cap = _safe_float(metrics.get("capacity"), default=0.0)
    site_count = int(state.get("site_count") or 0)

    targets = goal.get("targets", {}) or {}
    cov_t = targets.get("coverage_pct")
    cap_t = targets.get("capacity")
    site_limit = constraints.get("site_limit")
    site_exact = constraints.get("site_exact")

    if cov_t is not None and cov < float(cov_t):
        return False
    if cap_t is not None and cap < float(cap_t):
        return False
    if site_limit is not None and site_count > int(site_limit):
        return False
    if site_exact is not None and site_count != int(site_exact):
        return False
    return True


def _summarize_traj(path: str) -> Tuple[Dict[str, Any], Counter]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.loads(handle.read())
    if not isinstance(data, list) or not data:
        return {}, Counter()
    item = data[0]
    actions = item.get("actions", []) or []
    last_obs = {}
    if item.get("observations"):
        try:
            last_obs = json.loads(item["observations"][-1])
        except Exception:
            last_obs = {}
    state = last_obs.get("state", {}) or {}
    metrics = state.get("last_metrics", {}) or {}
    summary = {
        "path": path,
        "request_id": _request_id_from_path(path),
        "steps": int(item.get("steps") or (len(actions) - 1 if actions else 0)),
        "coverage": _safe_float(metrics.get("coverage"), default=0.0),
        "capacity": _safe_float(metrics.get("capacity"), default=0.0),
        "redundancy_rate": _safe_float(metrics.get("redundancy_rate"), default=0.0),
        "sites": int(state.get("site_count") or 0),
        "ok": _goal_constraints_ok(last_obs),
    }
    action_counts = Counter(_parse_action_name(a) for a in actions)
    return summary, action_counts


def _avg(values: List[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dirs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="*")
    parser.add_argument("--by-request", action="store_true")
    args = parser.parse_args()

    labels = args.labels or [f"set{i}" for i in range(len(args.traj_dirs))]
    if len(labels) != len(args.traj_dirs):
        raise SystemExit("--labels length must match --traj-dirs length")

    rows = []
    per_request = defaultdict(list)
    per_actions = {}
    for label, traj_dir in zip(labels, args.traj_dirs):
        summaries = []
        action_counts = Counter()
        for path in _iter_traj_files(traj_dir):
            summary, counts = _summarize_traj(path)
            if not summary:
                continue
            summaries.append(summary)
            action_counts.update(counts)
            per_request[(label, summary["request_id"])].append(summary)
        per_actions[label] = action_counts
        if summaries:
            rows.append(
                {
                    "label": label,
                    "n": len(summaries),
                    "ok_rate": _avg([1.0 if s["ok"] else 0.0 for s in summaries]),
                    "steps": _avg([s["steps"] for s in summaries]),
                    "coverage": _avg([s["coverage"] for s in summaries]),
                    "capacity": _avg([s["capacity"] for s in summaries]),
                    "redundancy_rate": _avg([s["redundancy_rate"] for s in summaries]),
                    "sites": _avg([s["sites"] for s in summaries]),
                }
            )

    header = "Label".ljust(12) + "N".rjust(6) + "OK%".rjust(8) + "Steps".rjust(8) + "Coverage".rjust(12) + "Capacity".rjust(12) + "Redundancy".rjust(12) + "Sites".rjust(8)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            row["label"].ljust(12)
            + f"{row['n']:6d}"
            + f"{row['ok_rate']*100:8.1f}"
            + f"{row['steps']:8.2f}"
            + f"{row['coverage']:12.4f}"
            + f"{row['capacity']:12.4f}"
            + f"{row['redundancy_rate']:12.4f}"
            + f"{row['sites']:8.2f}"
        )

    print("\nAction distribution:")
    for label in labels:
        counts = per_actions.get(label, Counter())
        total = sum(counts.values()) or 1
        parts = [f"{k}:{counts[k]}({counts[k]/total:.2f})" for k in sorted(counts)]
        print(f"{label}: " + ", ".join(parts))

    if args.by_request:
        print("\nPer-request summary:")
        req_header = "Label".ljust(12) + "Request".ljust(20) + "N".rjust(6) + "OK%".rjust(8) + "Coverage".rjust(12) + "Capacity".rjust(12) + "Redundancy".rjust(12) + "Sites".rjust(8)
        print(req_header)
        print("-" * len(req_header))
        for (label, req_id), summaries in sorted(per_request.items()):
            ok_rate = _avg([1.0 if s["ok"] else 0.0 for s in summaries])
            cov = _avg([s["coverage"] for s in summaries])
            cap = _avg([s["capacity"] for s in summaries])
            red = _avg([s["redundancy_rate"] for s in summaries])
            sites = _avg([s["sites"] for s in summaries])
            print(
                label.ljust(12)
                + req_id.ljust(20)[:20]
                + f"{len(summaries):6d}"
                + f"{ok_rate*100:8.1f}"
                + f"{cov:12.4f}"
                + f"{cap:12.4f}"
                + f"{red:12.4f}"
                + f"{sites:8.2f}"
            )


if __name__ == "__main__":
    main()
