"""注释
命令示例:
python -m ReAct.experiments.summarize_exp1_results
python -m ReAct.experiments.summarize_exp1_results --runs-root /Users/epiphanyer/Desktop/coding/paper_experiment/ReAct/exp/exp1_fixed_vs_unfixed/runs

参数说明:
- --runs-root: exp1 各方法运行结果目录，脚本会递归读取其中的 `summary.json`。
- --output-root: 汇总表输出目录，会生成 csv/json/md 三份文件。

逻辑说明:
该脚本把 exp1 的各方法结果汇总成统一表格，并按目录名区分当前统一配置
`llm_top_k_candidates=8, candidate_sample=16` 与旧版 `llm_top_k_candidates=16, candidate_sample=32`
的 `LLM-FT-ReAct` 结果，避免两者在最终表中混淆。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ReAct.experiments.summary_utils import planner_label, rows_to_markdown


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "ReAct" / "exp" / "exp1_fixed_vs_unfixed" / "runs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "ReAct" / "exp" / "exp1_fixed_vs_unfixed" / "all_result_summary"

FIELDNAMES = [
    "Setting",
    "Method",
    "OK (%)",
    "eta_cov",
    "avg_se",
    "eta_red",
    "Avg. Sites",
    "Avg. Steps",
    "Total Time(s)",
]

SETTING_ORDER = {"fixed": 0, "unfixed": 1}
METHOD_ORDER = {
    "Greedy": 0,
    "SA": 1,
    "GA": 2,
    "PSO": 3,
    "Exhaustive": 4,
    "LLM-Action": 5,
    "LLM-FT-Action": 6,
    "LLM-FT-ReAct": 7,
    "LLM-FT-ReAct (topk=16, sample=32)": 8,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Experiment 1 fixed vs unfixed results.")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def _infer_setting(label: str) -> str:
    if label.startswith("fixed_"):
        return "fixed"
    if label.startswith("unfixed_"):
        return "unfixed"
    return "unknown"


def _infer_llm_mode(label: str) -> str:
    if "_explain_weighted" in label:
        return "explain_weighted"
    return "decide"


def _suite_to_row(summary_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    label = summary_path.parent.name
    planner = str(summary.get("planner", ""))
    llm_mode = _infer_llm_mode(label)
    method = planner_label(planner, llm_mode)
    if label == "unfixed_llamafactory_explain_weighted32-16":
        method = "LLM-FT-ReAct (topk=16, sample=32)"
    perf = summary.get("perf", {}) or {}
    return {
        "Setting": _infer_setting(label),
        "Method": method,
        "OK (%)": round(float(summary.get("ok_rate", 0.0)) * 100.0, 4),
        "eta_cov": round(float(summary.get("coverage", 0.0)), 6),
        "avg_se": round(float(summary.get("capacity", 0.0)), 6),
        "eta_red": round(float(summary.get("redundancy_rate", 0.0)), 6),
        "Avg. Sites": round(float(summary.get("sites", 0.0)), 4),
        "Avg. Steps": round(float(summary.get("steps", 0.0)), 4),
        "Total Time(s)": round(float(perf.get("suite_runtime_sec", 0.0)), 4),
    }


def load_rows(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(runs_root.glob("*/*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(_suite_to_row(summary_path, summary))
    rows.sort(key=lambda row: (SETTING_ORDER.get(str(row["Setting"]), 99), METHOD_ORDER.get(str(row["Method"]), 99), str(row["Method"])))
    return rows


def write_outputs(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "exp1_run_metrics_summary.csv"
    json_path = output_root / "exp1_run_metrics_summary.json"
    md_path = output_root / "exp1_run_metrics_summary.md"

    csv_lines = [",".join(FIELDNAMES)]
    for row in rows:
        csv_lines.append(",".join(str(row.get(name, "")) for name in FIELDNAMES))
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(rows_to_markdown(rows, FIELDNAMES, "Experiment 1 Fixed vs Unfixed Summary"), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    runs_root = Path(args.runs_root).resolve()
    output_root = Path(args.output_root).resolve()
    rows = load_rows(runs_root)
    write_outputs(output_root, rows)
    print(f"[DONE] rows={len(rows)} output_root={output_root}")


if __name__ == "__main__":
    main()
