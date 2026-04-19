"""注释
用途:
  ReAct 运行性能与调试日志工具。统一处理选中站点提取、LLM 文本预览、CSV/JSONL 落盘，以及批量任务的耗时统计汇总。

示例命令:
  python -m ReAct.run_access_point_decision --planner llamafactory --print-step --print-timing --print-llm
  python -m ReAct.run_experiment_suite --planner llamafactory --maps-list ReAct/data/maps_test_paths.txt

参数说明:
  extract_sites_from_action(action): 从 selected_action 中提取标准化站点列表。
  preview_llm_text(text, max_len=1200): 生成适合终端展示的 LLM 文本预览。
  append_jsonl(path, row): 追加写入一行 JSONL。
  write_csv(path, rows, fieldnames): 按列顺序写出 CSV。
  summarize_run_records(records): 汇总批量任务的总耗时、均值、LLM/候选评分/env 步耗时与调用次数。

逻辑说明:
  该模块不参与决策，只负责把运行时日志和性能指标组织成稳定结构，
  便于单任务脚本打印、轨迹文件回写，以及批量实验生成 run_records.csv / run_events.jsonl / summary.json。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def extract_sites_from_action(action: Dict[str, Any] | None) -> List[Dict[str, float]]:
    if not isinstance(action, dict):
        return []
    args = action.get("args", {})
    if not isinstance(args, dict):
        return []
    sites = args.get("sites")
    if not isinstance(sites, list):
        return []
    normalized: List[Dict[str, float]] = []
    for site in sites:
        if not isinstance(site, dict):
            continue
        try:
            normalized.append(
                {
                    "row": int(site.get("row", 0)),
                    "col": int(site.get("col", 0)),
                    "z_m": float(site.get("z_m", 3.0)),
                }
            )
        except (TypeError, ValueError):
            continue
    return normalized


def preview_llm_text(text: str, max_len: int = 1200) -> str:
    clean = str(text or "").strip()
    if len(clean) <= max_len:
        return clean
    return clean[:max_len] + "...<truncated>"


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def summarize_run_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    tasks_count = len(rows)

    def _sum_float(key: str) -> float:
        return float(sum(float(row.get(key, 0.0) or 0.0) for row in rows))

    def _sum_int(key: str) -> int:
        return int(sum(int(row.get(key, 0) or 0) for row in rows))

    runtime_total = _sum_float("runtime_sec")
    llm_time_total = _sum_float("llm_time_sec")
    llm_calls_total = _sum_int("llm_calls")
    candidate_score_time_total = _sum_float("candidate_score_time_sec")
    candidate_score_calls_total = _sum_int("candidate_score_calls")
    env_step_time_total = _sum_float("env_step_time_sec")
    env_step_calls_total = _sum_int("env_step_calls")

    return {
        "tasks_count": tasks_count,
        "runtime_total_sec": runtime_total,
        "runtime_mean_sec": 0.0 if tasks_count == 0 else runtime_total / tasks_count,
        "llm_time_total_sec": llm_time_total,
        "llm_time_mean_sec": 0.0 if tasks_count == 0 else llm_time_total / tasks_count,
        "llm_calls_total": llm_calls_total,
        "llm_time_per_call_sec": 0.0 if llm_calls_total == 0 else llm_time_total / llm_calls_total,
        "candidate_score_time_total_sec": candidate_score_time_total,
        "candidate_score_time_mean_sec": 0.0 if tasks_count == 0 else candidate_score_time_total / tasks_count,
        "candidate_score_calls_total": candidate_score_calls_total,
        "candidate_score_time_per_call_sec": 0.0 if candidate_score_calls_total == 0 else candidate_score_time_total / candidate_score_calls_total,
        "env_step_time_total_sec": env_step_time_total,
        "env_step_time_mean_sec": 0.0 if tasks_count == 0 else env_step_time_total / tasks_count,
        "env_step_calls_total": env_step_calls_total,
        "env_step_time_per_call_sec": 0.0 if env_step_calls_total == 0 else env_step_time_total / env_step_calls_total,
    }
