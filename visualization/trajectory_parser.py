"""注释
命令示例:
python -m py_compile visualization/trajectory_parser.py

参数含义:
- `build_dashboard_state(...)`: 从 trajectory JSON 提取 GUI 需要的当前展示状态。
- `determine_flow_state(...)`: 推导右下流程图与双向箭头的高亮状态。

逻辑说明:
本文件把 ReAct trajectory 的 observation/action/step_logs 解析成 GUI 可直接消费的结构，避免界面层到处手写 JSON 细节。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _safe_load_json_text(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Expected observation payload to be a JSON object.")
    return payload


def _format_site_pair(site: dict[str, Any]) -> str:
    return f"({int(site['row'])},{int(site['col'])})"


def _extract_spectral_efficiency(metrics: dict[str, Any]) -> float | None:
    for key in ("spectral_efficiency", "se", "capacity"):
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def _format_percent(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100:.2f}%"


def _format_spectral_efficiency(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f} bps/Hz"


def _describe_goal(goal: dict[str, Any], constraints: dict[str, Any]) -> str:
    lines: list[str] = []
    primary = str((goal or {}).get("primary") or "").strip()
    targets = (goal or {}).get("targets") or {}
    if primary == "balance_coverage_capacity":
        lines.append("主目标：在覆盖率与信道容量之间进行综合平衡。")
    elif primary:
        lines.append(f"主目标：{primary}")
    coverage_pct = targets.get("coverage_pct")
    capacity = targets.get("capacity")
    if coverage_pct is not None:
        lines.append(f"覆盖率目标：不低于 {float(coverage_pct) * 100:.2f}%。")
    if capacity is not None:
        lines.append(f"信道容量目标：不低于 {float(capacity):.2f} Mbps。")
    site_exact = constraints.get("site_exact")
    site_limit = constraints.get("site_limit")
    if site_exact is not None:
        lines.append(f"站点约束：最终必须部署 {int(site_exact)} 个站点。")
    elif site_limit is not None:
        lines.append(f"站点约束：最终站点数不超过 {int(site_limit)} 个。")
    if not lines:
        lines.append("暂未解析到显式结构化目标。")
    return "\n".join(lines)


def _location_delta(previous_sites: list[dict[str, Any]], current_sites: list[dict[str, Any]]) -> str:
    previous_keys = {(int(site["row"]), int(site["col"])) for site in previous_sites}
    added_sites = [
        site for site in current_sites
        if (int(site["row"]), int(site["col"])) not in previous_keys
    ]
    if added_sites:
        return "; ".join(_format_site_pair(site) for site in added_sites)
    if current_sites and not previous_sites:
        return "; ".join(_format_site_pair(site) for site in current_sites)
    return ""


def _method_label(planner: str, rationale_text: str) -> str:
    text = rationale_text.lower()
    if planner == "llamafactory" and ("weight" in text or "explain" in text):
        return "LLM可解释性权重"
    if planner == "llamafactory":
        return "LLM可解释性权重"
    return "LLM可解释性权重"


def determine_flow_state(*, has_trajectory: bool, process_running: bool, step_count: int, finished: bool, has_initial_render: bool = False) -> dict[str, str]:
    flow = {
        "select_request": "complete" if has_trajectory else "current",
        "request_structuring": "complete" if has_trajectory else "pending",
        "initial_deployment": "complete" if has_trajectory else "pending",
        "decision_loop": "pending",
        "decision_complete_arrow": "pending",
        "complete": "pending",
    }
    if has_trajectory:
        # 修复：只要初始化站点已经渲染（has_initial_render=True），就立即切换到 decision_loop 状态
        # 不再等待 step_count > 0
        if has_initial_render:
            # INIT 出图后立即进入逐步闭环决策状态
            flow["initial_deployment"] = "complete"
            flow["decision_loop"] = "complete" if finished else "current"
            flow["decision_complete_arrow"] = "current" if finished else "pending"
            flow["complete"] = "current" if finished else "pending"
        elif step_count <= 0 and not finished:
            # 还没有渲染 INIT 图片时，显示初始站点部署状态
            flow["initial_deployment"] = "current"
        elif step_count > 0:
            # 已经有后续 step 时，显示逐步闭环决策状态
            flow["decision_loop"] = "complete" if finished else "current"
            flow["decision_complete_arrow"] = "current" if finished else "pending"
            flow["complete"] = "current" if finished else "pending"
    return flow


def build_dashboard_state(
    traj_path: Path,
    process_running: bool,
    observation_index: int | None = None,
    has_initial_render: bool = False,
) -> dict[str, Any]:
    payload = json.loads(Path(traj_path).read_text(encoding="utf-8"))
    if not payload:
        raise ValueError("Empty trajectory payload.")
    latest = payload[-1]
    observations = latest.get("observations", [])
    if not observations:
        raise ValueError("Trajectory has no observations.")
    latest_observation_index = max(0, len(observations) - 1)
    target_observation_index = latest_observation_index
    if observation_index is not None:
        target_observation_index = max(0, min(int(observation_index), latest_observation_index))
    current_obs = _safe_load_json_text(observations[target_observation_index])
    state = current_obs.get("state", {}) or {}
    metrics = state.get("last_metrics", {}) or {}
    sites = state.get("sites", []) or []
    step_logs = latest.get("step_logs", []) or []
    total_elapsed = 0.0
    table_rows = []
    metric_history = []
    running_sites: list[dict[str, Any]] = []
    for index, observation_text in enumerate(observations[: target_observation_index + 1]):
        parsed = _safe_load_json_text(observation_text)
        parsed_state = parsed.get("state", {}) or {}
        # 修复：区分 sites 不存在（None）和 sites 为空列表（[]）的情况
        parsed_sites = parsed_state.get("sites") if "sites" in parsed_state else None
        parsed_metrics = parsed_state.get("last_metrics", {}) or {}
        parsed_diagnosis = parsed.get("diagnosis", {}) or {}
        if index == 0:
            step_label = "init"
        else:
            step_label = str(index)
            if index - 1 < len(step_logs):
                total_elapsed += float(step_logs[index - 1].get("step_total_time_sec", 0.0) or 0.0)
        # 修复：只有当 parsed_sites 为 None（不存在）时才使用 running_sites
        # 如果 parsed_sites 是空列表，也应该使用它
        current_sites = parsed_sites if parsed_sites is not None else running_sites
        if current_sites is None:
            current_sites = []
        table_rows.append(
            {
                "step": step_label,
                "coverage": _format_percent(parsed_metrics.get("coverage")),
                "spectral_efficiency": _format_spectral_efficiency(_extract_spectral_efficiency(parsed_metrics)),
                "location": _location_delta(running_sites, current_sites),
                "ok": "OK" if bool(parsed_diagnosis.get("ok")) else "NO",
                "time": f"T+{total_elapsed:.2f} s",
                "is_current": index == target_observation_index,
            }
        )
        metric_history.append(
            {
                "step": step_label,
                "step_index": index,
                "coverage": float(parsed_metrics.get("coverage", 0.0) or 0.0),
                "spectral_efficiency": float(_extract_spectral_efficiency(parsed_metrics) or 0.0),
                "site_count": len(current_sites or []),
            }
        )
        # 修复：只有当 parsed_sites 不为 None 时才更新 running_sites
        if parsed_sites is not None:
            running_sites = parsed_sites
    rationale_text = "\n".join(latest.get("rationales", []) or [])
    planner = ((latest.get("perf", {}) or {}).get("planner")) or ""
    done = bool(step_logs[-1].get("done")) if step_logs else False
    diagnosis_payload = current_obs.get("diagnosis", {}) or {}
    site_count_raw = state.get("site_count")
    current_ap_count = len(sites)
    if site_count_raw is not None:
        try:
            parsed_site_count = int(site_count_raw)
        except (TypeError, ValueError):
            parsed_site_count = 0
        if parsed_site_count > 0 or not sites:
            current_ap_count = parsed_site_count
    return {
        "current_step": target_observation_index,
        "latest_available_step": latest_observation_index,
        "current_ap_count": int(current_ap_count),
        "request_text": current_obs.get("user_request", ""),
        "goal": current_obs.get("goal", {}) or {},
        "constraints": current_obs.get("constraints", {}) or {},
        "goal_human_readable": _describe_goal(
            current_obs.get("goal", {}) or {},
            current_obs.get("constraints", {}) or {},
        ),
        "diagnosis": (current_obs.get("diagnosis", {}) or {}).get("summary", ""),
        "ok": bool(diagnosis_payload.get("ok")),
        "metrics": {
            "coverage": float(metrics.get("coverage", 0.0) or 0.0),
            "spectral_efficiency": float(_extract_spectral_efficiency(metrics) or 0.0),
            "redundancy_rate": float(metrics.get("redundancy_rate", 0.0) or 0.0),
        },
        "sites": sites,
        "table_rows": table_rows,
        "metric_history": metric_history,
        "method_labels": {
            "init": "TSPL",
            "follow_up": _method_label(planner, rationale_text),
        },
        "flow": determine_flow_state(
            has_trajectory=True,
            process_running=process_running,
            step_count=len(step_logs),
            finished=done and not process_running,
            has_initial_render=has_initial_render,
        ),
    }
