"""注释
命令示例:
1. 使用 LLaMA-Factory LoRA 权重批量评估测试集，并自动输出轨迹与 OK 率汇总:
python autodl-tmp/code/ReAct/run_experiment_suite.py \
  --planner llamafactory \
  --maps-list autodl-tmp/code/ReAct/data/maps_test_paths.txt \
  --requests-file autodl-tmp/code/ReAct/requests/coverage_capacity_budget_v2.txt \
  --output-root autodl-tmp/code/ReAct/outputs/experiment_suites \
  --suite-name llamafactory_eval_202604181404 \
   --llamafactory-root autodl-tmp/LLaMA-Factory \
  --llamafactory-model autodl-tmp/Qwen2.5-7B \
  --llamafactory-adapter autodl-tmp/LLaMA-Factory/saves/Qwen2.5-7B/lora/train_2026-02-14-14-09-21 \
  --llamafactory-template qwen \
  --llamafactory-backend huggingface \
  --llamafactory-dtype auto \
  --eval-device mps \
  --max-steps 5 \
  --auto-steps \
  --candidate-sample 64 \
  --llm-top-k-candidates 16 \
  --llm-decision-mode explain_weighted \
  --print-step
2. 使用地图目录前 N 张图和全部需求文件跑启发式基线:
   python -m ReAct.run_experiment_suite \
     --planner heuristic \
     --maps-dir /abs/path/to/maps \
     --num-maps 10 \
     --requests-dir /Users/epiphanyer/Desktop/coding/paper_experiment/ReAct/requests \
     --output-root /abs/path/to/experiment_outputs \
     --suite-name heuristic_smoke
3. 只对单个需求文件执行实验:
   python -m ReAct.run_experiment_suite \
     --planner heuristic \
     --maps-list /abs/path/to/test_maps.txt \
     --request-file /Users/epiphanyer/Desktop/coding/paper_experiment/ReAct/requests/capacity_priority.txt \
     --output-root /abs/path/to/experiment_outputs \
     --suite-name heuristic_single_request

参数含义:
- --maps-dir: 测试地图目录；与 --maps-list 二选一。
- --maps-list: 测试地图路径列表文件，每行一个绝对路径或相对路径；适合直接指定测试集子集。
- --num-maps: 只取前多少张地图；0 表示不截断。
- --requests-dir: 需求文本目录；脚本会遍历其中全部 .txt，并通过 --user-request-path 传给单任务脚本。
- --request-file: 单个需求文本文件；传入时优先于 --requests-dir，用于只跑一个需求。
- --output-root: 实验输出根目录；默认写到 ReAct/outputs/experiment_suites。
- --suite-name: 实验名；默认按 planner 和时间戳生成。
- --traj-dir: 可选，单独指定轨迹目录；不传则写到 <output-root>/<suite-name>/trajs。
- --planner: 单任务决策器，支持 heuristic/random/openai/qwen/llamafactory 等。
- --prompt-path / --prompt-key: 透传给单任务脚本的 prompt 配置。
- --max-steps / --auto-steps: 单任务最大步数，或按需求文本自动推断步数。
- --candidate-sample / --llm-top-k-candidates / --llm-decision-mode: 单任务候选点采样和 LLM 决策模式配置。
- --eval-model / --eval-device / --init-mode / --init-k / --seed / --print-step / --print-llm / --print-timing / --llm-dump-path: 单任务评估模型、评估设备、初始化策略、日志与随机种子等配置。
- --openai-*: planner=openai 时透传的接口参数。
- --qwen-*: planner=qwen 时透传的本地 Qwen 参数。
- --llamafactory-*: planner=llamafactory 时透传的 base model、adapter、template、backend、dtype。
- --ppo-*: planner 需要 PPO 初始化时透传的 checkpoint 配置。

逻辑说明:
该脚本不改动 ReAct 的单任务主流程，而是把“测试地图集合 × 需求文件集合”展开成批量任务，逐个调用 run_access_point_decision.py 生成轨迹，然后复用 evaluate_decision_trajectories.py 与其轨迹统计逻辑，输出 summary.txt、summary.json 与 task_list.jsonl。
终端新增 SUITE / TASK / DONE / SUMMARY / ERROR 彩色状态行，便于观察当前跑到哪一个地图与需求组合；OK 率定义为：轨迹最终 observation 同时满足 coverage/capacity/site 约束的任务数，占全部任务数的比例。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

if __package__:
    from .cli_colors import status_line
    from .env_utils import resolve_city_map_paths
    from .evaluate_decision_trajectories import _summarize_traj
    from .perf_logging import append_jsonl, summarize_run_records, write_csv
    from .run_access_point_decision import run_task as run_single_task
else:
    from cli_colors import status_line
    from env_utils import resolve_city_map_paths
    from evaluate_decision_trajectories import _summarize_traj
    from perf_logging import append_jsonl, summarize_run_records, write_csv
    from run_access_point_decision import run_task as run_single_task


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_REQUESTS_DIR = ROOT_DIR / "requests"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "outputs" / "experiment_suites"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-dir", default="")
    parser.add_argument("--maps-list", default="")
    parser.add_argument("--num-maps", type=int, default=0)
    parser.add_argument("--requests-dir", default=str(DEFAULT_REQUESTS_DIR))
    parser.add_argument("--request-file", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--suite-name", default="")
    parser.add_argument("--traj-dir", default="")
    parser.add_argument(
        "--planner",
        choices=[
            "heuristic",
            "heuristic_greedy",
            "heuristic_sa",
            "heuristic_ga",
            "heuristic_pso",
            "heuristic_bruteforce",
            "heuristic_full_enum",
            "random",
            "openai",
            "qwen",
            "llamafactory",
        ],
        default="heuristic",
    )
    parser.add_argument("--prompt-path", default=str(ROOT_DIR / "prompts" / "radiomap.json"))
    parser.add_argument("--prompt-key", default="react_radiomap_decide")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--auto-steps", action="store_true")
    parser.add_argument("--candidate-sample", type=int, default=64)
    parser.add_argument("--llm-top-k-candidates", type=int, default=16)
    parser.add_argument("--llm-decision-mode", choices=["decide", "explain_weighted"], default="decide")
    parser.add_argument("--eval-model", choices=["pmnet", "rmnet", "proxy"], default="pmnet")
    parser.add_argument("--eval-device", choices=["auto", "cpu", "cuda", "mps"], default="mps")
    parser.add_argument("--init-mode", choices=["none", "random", "greedy", "ppo"], default="none")
    parser.add_argument("--init-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-llm", action="store_true")
    parser.add_argument("--print-timing", action="store_true")
    parser.add_argument("--llm-dump-path", default="")
    parser.add_argument("--print-step", action="store_true")
    parser.add_argument("--openai-api-key", default="")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--openai-base-url", default="https://api.openai.com")
    parser.add_argument("--openai-response-format", choices=["none", "json_object"], default="none")
    parser.add_argument("--qwen-model-path", default=str((ROOT_DIR.parent / "Qwen" / "Qwen2.5-7B").resolve()))
    parser.add_argument("--qwen-device", choices=["auto", "cpu", "cuda", "mps"], default="mps")
    parser.add_argument("--qwen-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=320)
    parser.add_argument("--llamafactory-root", default="")
    parser.add_argument("--llamafactory-model", default="")
    parser.add_argument("--llamafactory-adapter", default="")
    parser.add_argument("--llamafactory-template", default="qwen")
    parser.add_argument("--llamafactory-backend", default="huggingface")
    parser.add_argument("--llamafactory-dtype", default="auto")
    parser.add_argument("--ppo-checkpoint", default=str((ROOT_DIR / "checkpoints").resolve()))
    parser.add_argument("--ppo-version", choices=["single", "multi"], default="single")
    parser.add_argument("--ppo-init-k", type=int, default=1)
    return parser

# 遍历每个字符：
# 如果是：字母 (a-zA-Z)、数字 (0-9)、- 或 _ 则保留
# 否则 → 替换为 -
def _safe_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in text.strip())
    cleaned = cleaned.strip("-")
    return cleaned or "experiment"


def resolve_suite_name(args: argparse.Namespace) -> str:
    if args.suite_name:
        return _safe_name(args.suite_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{_safe_name(args.planner)}_{timestamp}"


def resolve_map_paths(args: argparse.Namespace) -> List[str]:
    if args.maps_list:
        maps_list_path = Path(args.maps_list)
        maps = [line.strip() for line in maps_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif args.maps_dir:
        maps = resolve_city_map_paths(args.maps_dir, default_path=args.maps_dir)
    else:
        raise ValueError("--maps-dir or --maps-list is required")
    if args.num_maps > 0:
        maps = maps[: args.num_maps]
    if not maps:
        raise FileNotFoundError("No map paths found for experiment suite")
    return maps


def resolve_request_paths(requests_dir: str, request_file: str = "") -> List[str]:
    request_file = request_file.strip()
    if request_file:
        request_path = Path(request_file)
        if not request_path.is_file():
            raise FileNotFoundError(f"Request file not found: {request_path}")
        return [str(request_path.resolve())]

    request_root = Path(requests_dir)
    requests = sorted(str(path.resolve()) for path in request_root.glob("*.txt"))
    if not requests:
        raise FileNotFoundError(f"No request files found in {request_root}")
    return requests


def build_tasks(map_paths: Iterable[str], request_paths: Iterable[str]) -> List[Dict[str, str]]:
    tasks: List[Dict[str, str]] = []
    # {"city_map_path": "/root/autodl-tmp/coding/test/buildingsWHeight/8.png", "user_request_path": "/root/autodl-tmp/coding/ReAct/requests/coverage_min_sites.txt"}
    for map_path in map_paths:
        for request_path in request_paths:
            tasks.append(
                {
                    "city_map_path": str(Path(map_path).resolve()),
                    "user_request_path": str(Path(request_path).resolve()),
                }
            )
    return tasks


def build_task_command(
    python_bin: str,
    run_script: Path,
    task: Dict[str, str],
    args: argparse.Namespace,
) -> List[str]:
    command = [
        python_bin,
        str(run_script),
        "--city-map-path",
        task["city_map_path"],
        "--user-request-path",
        task["user_request_path"],
        "--planner",
        args.planner,
        "--prompt-path",
        args.prompt_path,
        "--prompt-key",
        args.prompt_key,
        "--traj-dir",
        args.traj_dir,
        "--max-steps",
        str(args.max_steps),
        "--candidate-sample",
        str(args.candidate_sample),
        "--llm-top-k-candidates",
        str(args.llm_top_k_candidates),
        "--llm-decision-mode",
        args.llm_decision_mode,
        "--eval-model",
        args.eval_model,
        "--eval-device",
        args.eval_device,
        "--init-mode",
        args.init_mode,
        "--init-k",
        str(args.init_k),
        "--seed",
        str(args.seed),
        "--openai-api-key",
        args.openai_api_key,
        "--openai-model",
        args.openai_model,
        "--openai-base-url",
        args.openai_base_url,
        "--openai-response-format",
        args.openai_response_format,
        "--qwen-model-path",
        args.qwen_model_path,
        "--qwen-device",
        args.qwen_device,
        "--qwen-dtype",
        args.qwen_dtype,
        "--qwen-max-new-tokens",
        str(args.qwen_max_new_tokens),
        "--llamafactory-root",
        args.llamafactory_root,
        "--llamafactory-model",
        args.llamafactory_model,
        "--llamafactory-adapter",
        args.llamafactory_adapter,
        "--llamafactory-template",
        args.llamafactory_template,
        "--llamafactory-backend",
        args.llamafactory_backend,
        "--llamafactory-dtype",
        args.llamafactory_dtype,
        "--ppo-checkpoint",
        args.ppo_checkpoint,
        "--ppo-version",
        args.ppo_version,
        "--ppo-init-k",
        str(args.ppo_init_k),
    ]
    if getattr(args, "print_llm", False):
        command.append("--print-llm")
    if args.auto_steps:
        command.append("--auto-steps")
    if getattr(args, "print_timing", False):
        command.append("--print-timing")
    if args.print_step:
        command.append("--print-step")
    llm_dump_path = getattr(args, "llm_dump_path", "")
    if llm_dump_path:
        command.extend(["--llm-dump-path", llm_dump_path])
    return command


def summarize_traj_dir(traj_dir: Path, label: str) -> Dict[str, Any]:
    summaries: List[Dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    by_request: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for path in sorted(traj_dir.glob("*.json")):
        # Label            N     OK%   Steps    Coverage    Capacity  Redundancy   Sites
        # ------------------------------------------------------------------------------
        summary, counts = _summarize_traj(str(path))
        if not summary:
            continue
        summaries.append(summary)
        action_counts.update(counts)
        by_request[str(summary["request_id"])].append(summary)

    if not summaries:
        return {
            "label": label,
            "n": 0,
            "ok_rate": 0.0,
            "steps": 0.0,
            "coverage": 0.0,
            "capacity": 0.0,
            "redundancy_rate": 0.0,
            "sites": 0.0,
            "action_distribution": {},
            "by_request": {},
        }

    def _avg(key: str) -> float:
        return float(sum(float(item[key]) for item in summaries) / len(summaries))

    distribution_total = sum(action_counts.values()) or 1
    by_request_summary: Dict[str, Dict[str, Any]] = {}
    for request_id, request_rows in sorted(by_request.items()):
        request_text = next((str(row.get("request_text") or "").strip() for row in request_rows if row.get("request_text")), "")
        by_request_summary[request_id] = {
            "request_text": request_text,
            "n": len(request_rows),
            "ok_rate": float(sum(1.0 if row["ok"] else 0.0 for row in request_rows) / len(request_rows)),
            "steps": float(sum(float(row["steps"]) for row in request_rows) / len(request_rows)),
            "coverage": float(sum(float(row["coverage"]) for row in request_rows) / len(request_rows)),
            "capacity": float(sum(float(row["capacity"]) for row in request_rows) / len(request_rows)),
            "redundancy_rate": float(sum(float(row["redundancy_rate"]) for row in request_rows) / len(request_rows)),
            "sites": float(sum(float(row["sites"]) for row in request_rows) / len(request_rows)),
        }

    return {
        "label": label,
        "n": len(summaries),
        "ok_rate": _avg("ok"),
        "steps": _avg("steps"),
        "coverage": _avg("coverage"),
        "capacity": _avg("capacity"),
        "redundancy_rate": _avg("redundancy_rate"),
        "sites": _avg("sites"),
        "action_distribution": {
            name: {
                "count": int(count),
                "ratio": float(count / distribution_total),
            }
            for name, count in sorted(action_counts.items())
        },
        "by_request": by_request_summary,
    }


def write_task_list(path: Path, tasks: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")


def _fmt_ratio(value: float) -> str:
    return f"{float(value) * 100.0:.2f}%"


def _task_name(task: Dict[str, str]) -> str:
    map_id = Path(task["city_map_path"]).stem
    request_id = Path(task["user_request_path"]).stem
    return f"map={map_id} request={request_id}"


def _task_traj_path(traj_dir: Path, task: Dict[str, str]) -> Path:
    map_id = Path(task["city_map_path"]).stem
    request_id = Path(task["user_request_path"]).stem
    return traj_dir / f"{map_id}__{request_id}.json"


def _build_task_args(args: argparse.Namespace, task: Dict[str, str], traj_dir: Path) -> argparse.Namespace:
    task_args = argparse.Namespace(**vars(args))
    task_args.city_map_path = task["city_map_path"]
    task_args.user_request_path = task["user_request_path"]
    task_args.user_request = ""
    task_args.traj_dir = str(traj_dir)
    task_args.traj_id = f"{Path(task['city_map_path']).stem}__{Path(task['user_request_path']).stem}"
    return task_args


def _task_record(task: Dict[str, str], task_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "map_id": Path(task["city_map_path"]).stem,
        "request_id": Path(task["user_request_path"]).stem,
        "traj_path": task_result.get("traj_path", ""),
        "runtime_sec": float(task_result.get("runtime_sec", 0.0) or 0.0),
        "llm_time_sec": float(task_result.get("llm_time_sec", 0.0) or 0.0),
        "llm_calls": int(task_result.get("llm_calls", 0) or 0),
        "candidate_score_time_sec": float(task_result.get("candidate_score_time_sec", 0.0) or 0.0),
        "candidate_score_calls": int(task_result.get("candidate_score_calls", 0) or 0),
        "env_step_time_sec": float(task_result.get("env_step_time_sec", 0.0) or 0.0),
        "env_step_calls": int(task_result.get("env_step_calls", 0) or 0),
        "steps_completed": int(task_result.get("steps_completed", 0) or 0),
        "planner": str(task_result.get("planner", "")),
        "status": "ok",
    }


def run_suite(args: argparse.Namespace) -> Dict[str, Any]:
    # output_dir / suite_dir / traj_dir
    suite_name = resolve_suite_name(args)
    suite_dir = Path(args.output_root).expanduser().resolve() / suite_name
    traj_dir = Path(args.traj_dir).expanduser().resolve() if args.traj_dir else suite_dir / "trajs"
    traj_dir.mkdir(parents=True, exist_ok=True)
    suite_dir.mkdir(parents=True, exist_ok=True)
    args.traj_dir = str(traj_dir)
    if args.planner in {"openai", "qwen", "llamafactory"}:
        args.print_llm = True
        args.print_timing = True
        args.print_step = True

    map_paths = resolve_map_paths(args)
    request_paths = resolve_request_paths(args.requests_dir, args.request_file)
    tasks = build_tasks(map_paths, request_paths)
    write_task_list(suite_dir / "task_list.jsonl", tasks)
    print(
        status_line(
            "SUITE",
            (
                f"name={suite_name} planner={args.planner} maps={len(map_paths)} "
                f"requests={len(request_paths)} tasks={len(tasks)} traj_dir={traj_dir}"
            ),
            tone="accent",
        )
    )

    python_bin = sys.executable
    suite_start = time.perf_counter()
    run_records: List[Dict[str, Any]] = []
    run_events_path = suite_dir / "run_events.jsonl"
    if run_events_path.exists():
        run_events_path.unlink()
    append_jsonl(
        run_events_path,
        {
            "event": "suite_start",
            "suite_name": suite_name,
            "planner": args.planner,
            "tasks_count": len(tasks),
            "start_time": time.time(),
            "traj_dir": str(traj_dir),
        },
    )
    for task_idx, task in enumerate(tasks, 1):
        print(status_line("TASK", f"{task_idx}/{len(tasks)} {_task_name(task)}", tone="progress"))
        task_start = time.perf_counter()
        append_jsonl(
            run_events_path,
            {
                "event": "task_start",
                "suite_name": suite_name,
                "task_index": task_idx,
                "tasks_count": len(tasks),
                "map_id": Path(task["city_map_path"]).stem,
                "request_id": Path(task["user_request_path"]).stem,
                "start_time": time.time(),
            },
        )
        try:
            task_args = _build_task_args(args, task, traj_dir)
            task_result = run_single_task(task_args)
            record = _task_record(task, task_result)
            run_records.append(record)
            append_jsonl(
                run_events_path,
                {
                    "event": "task_end",
                    "suite_name": suite_name,
                    "task_index": task_idx,
                    "tasks_count": len(tasks),
                    "map_id": record["map_id"],
                    "request_id": record["request_id"],
                    "runtime_sec": float(record["runtime_sec"]),
                    "llm_time_sec": float(record["llm_time_sec"]),
                    "candidate_score_time_sec": float(record["candidate_score_time_sec"]),
                    "env_step_time_sec": float(record["env_step_time_sec"]),
                    "status": "ok",
                    "end_time": time.time(),
                },
            )
        except Exception as exc:
            append_jsonl(
                run_events_path,
                {
                    "event": "task_end",
                    "suite_name": suite_name,
                    "task_index": task_idx,
                    "tasks_count": len(tasks),
                    "map_id": Path(task["city_map_path"]).stem,
                    "request_id": Path(task["user_request_path"]).stem,
                    "runtime_sec": time.perf_counter() - task_start,
                    "status": "failed",
                    "error": str(exc),
                    "end_time": time.time(),
                },
            )
            print(status_line("ERROR", f"{task_idx}/{len(tasks)} {_task_name(task)}", tone="error"))
            raise
        print(
            status_line(
                "DONE",
                (
                    f"{task_idx}/{len(tasks)} {_task_name(task)} traj={_task_traj_path(traj_dir, task)} "
                    f"runtime={float(record['runtime_sec']):.2f}s llm={float(record['llm_time_sec']):.2f}s"
                ),
                tone="success",
            )
        )

    eval_command = [
        python_bin,
        str(ROOT_DIR / "evaluate_decision_trajectories.py"),
        "--traj-dirs",
        str(traj_dir),
        "--by-request",
    ]
    eval_result = subprocess.run(eval_command, check=True, capture_output=True, text=True)
    summary = summarize_traj_dir(traj_dir, label=suite_name)
    summary["suite_dir"] = str(suite_dir)
    summary["traj_dir"] = str(traj_dir)
    summary["maps_count"] = len(map_paths)
    summary["requests_count"] = len(request_paths)
    summary["tasks_count"] = len(tasks)
    summary["planner"] = args.planner
    summary["perf"] = summarize_run_records(run_records)
    summary["perf"]["suite_runtime_sec"] = time.perf_counter() - suite_start

    write_csv(
        suite_dir / "run_records.csv",
        run_records,
        [
            "map_id",
            "request_id",
            "traj_path",
            "runtime_sec",
            "llm_time_sec",
            "llm_calls",
            "candidate_score_time_sec",
            "candidate_score_calls",
            "env_step_time_sec",
            "env_step_calls",
            "steps_completed",
            "planner",
            "status",
        ],
    )
    append_jsonl(
        run_events_path,
        {
            "event": "suite_end",
            "suite_name": suite_name,
            "planner": args.planner,
            "tasks_count": len(tasks),
            "runtime_sec": float(summary["perf"]["suite_runtime_sec"]),
            "llm_time_sec": float(summary["perf"]["llm_time_total_sec"]),
            "candidate_score_time_sec": float(summary["perf"]["candidate_score_time_total_sec"]),
            "env_step_time_sec": float(summary["perf"]["env_step_time_total_sec"]),
            "status": "ok",
            "end_time": time.time(),
        },
    )

    (suite_dir / "summary.txt").write_text(eval_result.stdout, encoding="utf-8")
    (suite_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        status_line(
            "SUMMARY",
            (
                f"ok_rate={_fmt_ratio(summary['ok_rate'])} avg_steps={float(summary['steps']):.2f} "
                f"avg_coverage={float(summary['coverage']):.3f} "
                f"runtime={float(summary['perf']['suite_runtime_sec']):.2f}s "
                f"llm={float(summary['perf']['llm_time_total_sec']):.2f}s suite_dir={suite_dir}"
            ),
            tone="success",
        )
    )
    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_suite(args)
    print(
        json.dumps(
            {
                "suite_dir": summary["suite_dir"],
                "traj_dir": summary["traj_dir"],
                "tasks_count": summary["tasks_count"],
                "ok_rate_pct": round(float(summary["ok_rate"]) * 100.0, 2),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
