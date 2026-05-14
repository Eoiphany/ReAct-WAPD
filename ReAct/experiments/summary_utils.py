"""注释
命令示例:
python -m ReAct.experiments.run_exp1_fixed_vs_unfixed --help
python -m ReAct.experiments.run_exp2_init_decision_matrix --help
python -m ReAct.experiments.run_exp3_generalization --help

参数说明:
- make_suite_args(...): 生成 run_experiment_suite.run_suite 所需参数。
- ensure_request_file(...): 在实验输出目录下生成需求文本。
- summary_to_metric_row(...): 把批量实验 summary 转成论文表格行。
- write_table_outputs(...): 同时输出 csv/json/md 三种表格结果。

逻辑说明:
本文件复用 ReAct 现有 batch suite 能力，只负责拼装实验参数、统一指标命名与导出表格。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


REACT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REACT_ROOT.parent
DEFAULT_EXPERIMENT_OUTPUT_ROOT = REACT_ROOT / "exp"
DEFAULT_TWO_STAGE_MODULE_STATE = PROJECT_ROOT / "Autobs" / "outputs" / "rerank" / "best_module_state.pt"
DEFAULT_QWEN_MODEL_PATH = PROJECT_ROOT / "Qwen2.5-7B"
DEFAULT_LLAMAFACTORY_ROOT = PROJECT_ROOT / "Qwen" / "LLaMA-Factory"
DEFAULT_LLAMAFACTORY_ADAPTER = (
    PROJECT_ROOT / "LLaMA-Factory" / "saves" / "Qwen2.5-7B" / "lora" / "train_2026-02-14-14-09-21"
)
DEFAULT_RADIOMAP3DSEER_MAPS_DIR = PROJECT_ROOT / "dataset" / "png" / "buildingsWHeight"
DEFAULT_RADIOMAPSEER_ROOT = PROJECT_ROOT / "RadioMapSeer"
DEFAULT_RADIOMAPSEER_MAPS_DIR = DEFAULT_RADIOMAPSEER_ROOT / "png" / "buildings_complete"
DEFAULT_RADIOMAPSEER_GAIN_DIR = DEFAULT_RADIOMAPSEER_ROOT / "gain" / "IRT4"
DEFAULT_RMNET_RADIOMAP3DSEER = PROJECT_ROOT / "surrogate" / "checkpoints" / "rmnet_radiomap3dseer.pt"
DEFAULT_REQUESTS_DIR = REACT_ROOT / "requests"
DEFAULT_TEST_MAPS_LIST = REACT_ROOT / "data" / "maps_test_paths.txt"

TABLE_COLUMNS = [
    "OK (%)",
    "eta_cov",
    "avg_se",
    "eta_red",
    "Avg. Sites",
    "Avg. Steps",
    "Total Time(s)",
]


def ensure_request_file(output_dir: Path, name: str, text: str) -> Path:
    request_dir = output_dir / "_generated_requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    path = request_dir / name
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def render_fixed_request(site_count: int) -> str:
    return (
        "Deploy exactly the required number of sites under the fixed-site constraint.\n"
        "Stop when the exact site count is reached; prioritize maximizing final coverage, with capacity as secondary.\n"
        f"Set to {int(site_count)} sites."
    )


def render_unfixed_request(site_limit: int) -> str:
    return (
        "Coverage and capacity are both primary goals. "
        "Target coverage >= 92% and average Spectral Efficiency >= 1.85 bps/Hz in ROI.\n"
        f"Keep site count <= {int(site_limit)}. "
        "If one target is already met, prefer actions that reduce the remaining gap without blindly adding sites.\n"
        "Do not optimize redundancy in this request."
    )


def write_path_list(output_dir: Path, name: str, paths: Iterable[Path]) -> Path:
    list_dir = output_dir / "_generated_lists"
    list_dir.mkdir(parents=True, exist_ok=True)
    list_path = list_dir / name
    lines = [str(Path(path).resolve()) for path in paths]
    list_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return list_path


def build_scene_maps_from_gain_dir(
    gain_dir: Path,
    maps_dir: Path,
    num_maps: int,
) -> list[Path]:
    if not gain_dir.is_dir():
        raise FileNotFoundError(f"RadioMapSeer gain dir not found: {gain_dir}")
    if not maps_dir.is_dir():
        raise FileNotFoundError(f"RadioMapSeer maps dir not found: {maps_dir}")

    def _sort_key(value: str) -> tuple[int, int | str]:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    scene_ids: list[str] = []
    seen: set[str] = set()
    for gain_path in sorted(gain_dir.glob("*.png"), key=lambda path: _sort_key(path.stem.split("_", 1)[0])):
        scene_id = gain_path.stem.split("_", 1)[0]
        if scene_id in seen:
            continue
        scene_map = maps_dir / f"{scene_id}.png"
        if scene_map.is_file():
            seen.add(scene_id)
            scene_ids.append(scene_id)
        if len(scene_ids) >= max(1, int(num_maps)):
            break
    if len(scene_ids) < max(1, int(num_maps)):
        raise ValueError(
            f"Only found {len(scene_ids)} usable RadioMapSeer scenes from {gain_dir}, fewer than requested {num_maps}"
        )
    return [maps_dir / f"{scene_id}.png" for scene_id in scene_ids]


def make_suite_args(
    *,
    maps_dir: Path,
    maps_list: Path | None = None,
    num_maps: int,
    request_file: Path,
    output_root: Path,
    suite_name: str,
    planner: str,
    eval_device: str,
    init_mode: str = "none",
    init_k: int = 1,
    two_stage_module_state: Path = DEFAULT_TWO_STAGE_MODULE_STATE,
    two_stage_init_k: int = 1,
    qwen_model_path: Path = DEFAULT_QWEN_MODEL_PATH,
    llamafactory_root: Path = DEFAULT_LLAMAFACTORY_ROOT,
    llamafactory_model: Path = DEFAULT_QWEN_MODEL_PATH,
    llamafactory_adapter: Path = DEFAULT_LLAMAFACTORY_ADAPTER,
    llm_decision_mode: str = "decide",
    max_steps: int = 8,
    candidate_sample: int = 16,
    heuristic_online_candidate_sample: int = 128,
    llm_top_k_candidates: int = 8,
    eval_model_path: Path | None = None,
    heuristic_search_budget: int = 100,
    heuristic_candidate_stride: int = 12,
    heuristic_candidate_limit: int = 256,
    use_heuristic_cache: bool = True,
    replay_traj_dir: Path | None = None,
    seed: int = 42,
) -> argparse.Namespace:
    resolved_search_budget = int(heuristic_search_budget)
    return argparse.Namespace(
        maps_dir="" if maps_list is not None else str(maps_dir),
        maps_list="" if maps_list is None else str(maps_list),
        num_maps=int(num_maps),
        requests_dir=str(DEFAULT_REQUESTS_DIR),
        request_file=str(request_file),
        output_root=str(output_root),
        suite_name=suite_name,
        traj_dir="",
        planner=planner,
        prompt_path=str(REACT_ROOT / "prompts" / "radiomap.json"),
        prompt_key="react_radiomap_decide",
        max_steps=int(max_steps),
        auto_steps=False,
        candidate_sample=int(candidate_sample),
        heuristic_online_candidate_sample=int(heuristic_online_candidate_sample),
        llm_top_k_candidates=int(llm_top_k_candidates),
        llm_decision_mode=llm_decision_mode,
        eval_model="rmnet",
        eval_model_path="" if eval_model_path is None else str(eval_model_path),
        eval_device=eval_device,
        init_mode=init_mode,
        init_k=int(init_k),
        seed=int(seed),
        print_llm=False,
        print_timing=False,
        llm_dump_path="",
        print_step=False,
        visualization_sync_dir="",
        visualization_sync_timeout_sec=0.0,
        openai_api_key="",
        openai_model="gpt-4o-mini",
        openai_base_url="https://api.openai.com",
        openai_response_format="none",
        qwen_model_path=str(qwen_model_path),
        qwen_device=eval_device,
        qwen_dtype="auto",
        qwen_max_new_tokens=320,
        llamafactory_root=str(llamafactory_root),
        llamafactory_model=str(llamafactory_model),
        llamafactory_adapter=str(llamafactory_adapter),
        llamafactory_template="qwen",
        llamafactory_backend="huggingface",
        llamafactory_dtype="auto",
        two_stage_module_state=str(two_stage_module_state),
        two_stage_version="auto",
        two_stage_init_k=int(two_stage_init_k),
        heuristic_search_budget=resolved_search_budget,
        heuristic_candidate_stride=int(heuristic_candidate_stride),
        heuristic_candidate_limit=int(heuristic_candidate_limit),
        use_heuristic_cache=bool(use_heuristic_cache),
        replay_traj_dir="" if replay_traj_dir is None else str(replay_traj_dir),
    )


def run_named_suite(args: argparse.Namespace) -> Dict[str, Any]:
    from ReAct.run_experiment_suite import run_suite
    from ReAct.qwen_adapter import clear_qwen_cache

    try:
        return run_suite(args)
    finally:
        clear_qwen_cache()


def summary_to_metric_row(summary: Dict[str, Any]) -> Dict[str, float]:
    perf = summary.get("perf", {}) or {}
    return {
        "OK (%)": round(float(summary.get("ok_rate", 0.0)) * 100.0, 4),
        "eta_cov": round(float(summary.get("coverage", 0.0)), 6),
        "avg_se": round(float(summary.get("capacity", 0.0)), 6),
        "eta_red": round(float(summary.get("redundancy_rate", 0.0)), 6),
        "Avg. Sites": round(float(summary.get("sites", 0.0)), 4),
        "Avg. Steps": round(float(summary.get("steps", 0.0)), 4),
        "Total Time(s)": round(float(perf.get("suite_runtime_sec", 0.0)), 4),
    }


def rows_to_markdown(rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str], title: str) -> str:
    header = f"# {title}\n\n"
    cols = list(fieldnames)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(name, "")) for name in cols) + " |")
    return header + "\n".join(lines) + "\n"


def write_table_outputs(
    output_dir: Path,
    stem: str,
    rows: Sequence[Dict[str, Any]],
    fieldnames: Sequence[str],
    title: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"

    csv_lines = [",".join(fieldnames)]
    for row in rows:
        csv_lines.append(",".join(str(row.get(name, "")) for name in fieldnames))
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(rows_to_markdown(rows, fieldnames, title), encoding="utf-8")


def planner_label(planner: str, llm_decision_mode: str = "") -> str:
    mapping = {
        "heuristic_greedy": "Greedy",
        "heuristic_ga": "GA",
        "heuristic_pso": "PSO",
        "heuristic_sa": "SA",
        "heuristic_exhaustive": "Exhaustive",
        "qwen": "LLM-Action",
        "llamafactory:decide": "LLM-FT-Action",
        "llamafactory:explain_weighted": "LLM-FT-ReAct",
    }
    if planner == "llamafactory":
        return mapping[f"{planner}:{llm_decision_mode}"]
    return mapping.get(planner, planner)


def init_label(init_mode: str) -> str:
    mapping = {
        "none": "",
        "random": "Random",
        "greedy": "Greedy",
        "heuristic_ga": "GA",
        "heuristic_pso": "PSO",
        "heuristic_sa": "SA",
        "two_stage": "TSPL",
    }
    return mapping.get(init_mode, init_mode)
