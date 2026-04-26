"""注释
命令示例:
1. 复现旧目录风格的 `trajs_llamafactory_v4` 轨迹:
100*num_requests*num_heuristic
python -m ReAct.generate_decision_trajectories \
  --planner heuristic_ga \
  --maps-list /Users/epiphanyer/Desktop/coding/paper_experiment/ReAct/data/maps_test_paths.txt \
  --requests-dir /Users/epiphanyer/Desktop/coding/paper_experiment/ReAct/requests \
  --traj-dir-root /Users/epiphanyer/Desktop/coding/paper_experiment/ReAct \
  --traj-dir-name trajs_heuristic \
  --llm-decision-mode explain_weighted \
  --eval-model rmnet

  --planner heuristic_greedy heuristic_sa heuristic_pso

2. 一次性跑全部启发式轨迹，并按方法分别构建“只学原因”的数据集:
python -m ReAct.generate_decision_trajectories \
  --all-heuristics \
  --maps-list /Users/epiphanyer/Desktop/coding/paper_experiment/ReAct/data/maps_test_paths.txt \
  --requests-dir /Users/epiphanyer/Desktop/coding/paper_experiment/ReAct/requests \
  --eval-model rmnet \
  --build-dataset-output /Users/epiphanyer/Desktop/coding/paper_experiment/ReAct/train_data/radiomap_reason_heuristics.json \
  --dataset-mode rationale \
  --eval-device cpu

参数含义:
- --planner: 规划器名称，可重复传入多次；支持 llamafactory、qwen、openai、random 以及各类 heuristic_*。
- --all-heuristics: 自动展开当前 ReAct 已桥接的全部启发式方法。
- --maps-dir / --maps-list / --num-maps: 地图来源配置，语义与 `run_experiment_suite.py` 一致。
- --requests-dir / --request-file: 需求来源配置，二选一或以后者优先。
- --traj-dir-root: 轨迹目录根路径；单 planner 时可配合 --traj-dir-name 精确控制最终输出目录。
- --traj-dir-name: 仅单 planner 时生效，直接指定轨迹目录名，例如 `trajs_llamafactory_v4`。
- --output-root: suite 汇总目录根路径，保存 task_list、summary、run_records 等附加产物。
- --build-dataset-output: 可选；若传入则在轨迹生成完成后自动构建数据集；多 planner 时默认按方法分别生成多个文件。
- --merge-datasets: 多 planner 时把所有轨迹合并成一个数据集文件；默认关闭。
- --dataset-mode: `rationale` 只学习原因；`rationale_weights` 学习原因+权重。
- --eval-model / --eval-device: 代理模型与设备；当前默认使用 rmnet。
- --heuristic-max-evals / --heuristic-candidate-stride / --heuristic-candidate-limit: 传给外部启发式方法的关键搜索预算。
- --llamafactory-* / --qwen-* / --openai-*: 透传给对应 planner 的推理参数。

脚本逻辑说明:
这个脚本参考老仓库 `/Users/epiphanyer/Desktop/coding/ReAct/tools/generate_trajs.py` 的“批量生成轨迹”职责，
但底层不再直接 shell 调旧 `run_radiomap.py`，而是复用当前子项目的 `run_experiment_suite.py`。
这样可以直接吃到 `paper_experiment/Heuristic` 中已经桥接进来的 greedy / SA / GA / PSO /
candidate-enumeration / exhaustive 等多种启发式实现，并把每个 planner 的轨迹稳定落成
`map__request.json`。如果同时给了 `--build-dataset-output`，脚本会继续调用
`build_explain_weighted_dataset.py` 生成数据集；默认是“每个方法一个文件”，更适合把不同启发式方法当作数据增强来源。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

if __package__:
    from .build_explain_weighted_dataset import build_dataset
    from .run_experiment_suite import build_parser as build_suite_parser
    from .run_experiment_suite import run_suite
else:
    from build_explain_weighted_dataset import build_dataset
    from run_experiment_suite import build_parser as build_suite_parser
    from run_experiment_suite import run_suite


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "outputs" / "trajectory_generation"
DEFAULT_REQUESTS_DIR = ROOT_DIR / "requests"
ALL_HEURISTIC_PLANNERS = [
    "heuristic_greedy",
    "heuristic_sa",
    "heuristic_ga",
    "heuristic_pso",
    "heuristic_candidate_enum",
    "heuristic_exhaustive",
    "heuristic_bruteforce",
    "heuristic_full_enum",
]
ALL_PLANNERS = [
    "heuristic",
    *ALL_HEURISTIC_PLANNERS,
    "random",
    "openai",
    "qwen",
    "llamafactory",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", action="append", choices=ALL_PLANNERS, default=[])
    parser.add_argument("--all-heuristics", action="store_true")
    parser.add_argument("--maps-dir", default="")
    parser.add_argument("--maps-list", default="")
    parser.add_argument("--num-maps", type=int, default=0)
    parser.add_argument("--requests-dir", default=str(DEFAULT_REQUESTS_DIR))
    parser.add_argument("--request-file", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--traj-dir-root", default=str(ROOT_DIR / "outputs"))
    parser.add_argument("--traj-dir-name", default="")
    parser.add_argument("--suite-name-prefix", default="trajgen")
    parser.add_argument("--prompt-path", default=str(ROOT_DIR / "prompts" / "radiomap.json"))
    parser.add_argument("--prompt-key", default="react_radiomap_decide")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--auto-steps", action="store_true")
    parser.add_argument("--candidate-sample", type=int, default=64)
    parser.add_argument("--llm-top-k-candidates", type=int, default=16)
    parser.add_argument("--llm-decision-mode", choices=["decide", "explain_weighted"], default="decide")
    parser.add_argument("--eval-model", choices=["pmnet", "rmnet", "proxy"], default="rmnet")
    parser.add_argument("--eval-device", choices=["auto", "cpu", "cuda", "mps"], default="mps")
    parser.add_argument("--heuristic-max-evals", type=int, default=200)
    parser.add_argument("--heuristic-candidate-stride", type=int, default=8)
    parser.add_argument("--heuristic-candidate-limit", type=int, default=500)
    parser.add_argument("--init-mode", choices=["none", "random", "greedy", "two_stage"], default="none")
    parser.add_argument("--init-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-llm", action="store_true")
    parser.add_argument("--print-timing", action="store_true")
    parser.add_argument("--print-step", action="store_true")
    parser.add_argument("--llm-dump-path", default="")
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
    parser.add_argument("--two-stage-module-state", default=str((ROOT_DIR / "../Autobs/bandit_policy/best_module_state.pt").resolve()))
    parser.add_argument("--two-stage-version", choices=["auto", "single", "multi"], default="auto")
    parser.add_argument("--two-stage-init-k", type=int, default=1)
    parser.add_argument("--build-dataset-output", default="")
    parser.add_argument("--merge-datasets", action="store_true")
    parser.add_argument("--dataset-mode", choices=["rationale", "rationale_weights"], default="rationale")
    parser.add_argument("--dataset-prompt-key", default="")
    parser.add_argument("--dataset-max-trajs", type=int, default=0)
    parser.add_argument("--dataset-max-steps", type=int, default=0)
    parser.add_argument("--dataset-max-samples", type=int, default=0)
    return parser


def _resolve_planners(args: argparse.Namespace) -> List[str]:
    planners: List[str] = list(args.planner or [])
    if args.all_heuristics:
        planners.extend(ALL_HEURISTIC_PLANNERS)
    if not planners:
        planners = ["llamafactory"]
    unique: List[str] = []
    for planner in planners:
        if planner not in unique:
            unique.append(planner)
    return unique


def _suite_args() -> argparse.Namespace:
    return build_suite_parser().parse_args([])


def _traj_dir_for_planner(args: argparse.Namespace, planner: str, planners_count: int) -> Path:
    traj_root = Path(args.traj_dir_root).expanduser().resolve()
    if planners_count == 1 and args.traj_dir_name:
        return traj_root / args.traj_dir_name
    return traj_root / f"trajs_{planner}"


def _suite_name_for_planner(args: argparse.Namespace, planner: str) -> str:
    return f"{args.suite_name_prefix}_{planner}"


def _dataset_prompt_key(args: argparse.Namespace) -> str:
    if args.dataset_prompt_key:
        return args.dataset_prompt_key
    if args.dataset_mode == "rationale":
        return "react_radiomap_explain_json"
    return "react_radiomap_explain_weighted_json"


def _dataset_output_for_planner(base_output: str | Path, planner: str) -> Path:
    base = Path(base_output).expanduser().resolve()
    if base.suffix.lower() not in {".json", ".jsonl"}:
        raise ValueError("--build-dataset-output must end with .json or .jsonl")
    return base.with_name(f"{base.stem}__{planner}{base.suffix}")


def _configure_suite_args(base: argparse.Namespace, cli_args: argparse.Namespace, planner: str, traj_dir: Path) -> argparse.Namespace:
    base.maps_dir = cli_args.maps_dir
    base.maps_list = cli_args.maps_list
    base.num_maps = cli_args.num_maps
    base.requests_dir = cli_args.requests_dir
    base.request_file = cli_args.request_file
    base.output_root = cli_args.output_root
    base.suite_name = _suite_name_for_planner(cli_args, planner)
    base.traj_dir = str(traj_dir)
    base.planner = planner
    base.prompt_path = cli_args.prompt_path
    base.prompt_key = cli_args.prompt_key
    base.max_steps = cli_args.max_steps
    base.auto_steps = cli_args.auto_steps
    base.candidate_sample = cli_args.candidate_sample
    base.llm_top_k_candidates = cli_args.llm_top_k_candidates
    base.llm_decision_mode = cli_args.llm_decision_mode
    base.eval_model = cli_args.eval_model
    base.eval_device = cli_args.eval_device
    base.init_mode = cli_args.init_mode
    base.init_k = cli_args.init_k
    base.seed = cli_args.seed
    base.print_llm = cli_args.print_llm
    base.print_timing = cli_args.print_timing
    base.print_step = cli_args.print_step
    base.llm_dump_path = cli_args.llm_dump_path
    base.openai_api_key = cli_args.openai_api_key
    base.openai_model = cli_args.openai_model
    base.openai_base_url = cli_args.openai_base_url
    base.openai_response_format = cli_args.openai_response_format
    base.qwen_model_path = cli_args.qwen_model_path
    base.qwen_device = cli_args.qwen_device
    base.qwen_dtype = cli_args.qwen_dtype
    base.qwen_max_new_tokens = cli_args.qwen_max_new_tokens
    base.llamafactory_root = cli_args.llamafactory_root
    base.llamafactory_model = cli_args.llamafactory_model
    base.llamafactory_adapter = cli_args.llamafactory_adapter
    base.llamafactory_template = cli_args.llamafactory_template
    base.llamafactory_backend = cli_args.llamafactory_backend
    base.llamafactory_dtype = cli_args.llamafactory_dtype
    base.two_stage_module_state = cli_args.two_stage_module_state
    base.two_stage_version = cli_args.two_stage_version
    base.two_stage_init_k = cli_args.two_stage_init_k
    # run_task 会读这些字段；run_experiment_suite 自己的 parser 里没有全部暴露，所以这里手动补齐。
    base.heuristic_max_evals = cli_args.heuristic_max_evals
    base.heuristic_candidate_stride = cli_args.heuristic_candidate_stride
    base.heuristic_candidate_limit = cli_args.heuristic_candidate_limit
    return base


def main() -> None:
    args = build_parser().parse_args()
    planners = _resolve_planners(args)
    generated_traj_dirs: List[Path] = []
    planner_summaries: List[Dict[str, Any]] = []

    for planner in planners:
        suite_args = _configure_suite_args(
            _suite_args(),
            args,
            planner,
            _traj_dir_for_planner(args, planner, planners_count=len(planners)),
        )
        summary = run_suite(suite_args)
        generated_traj_dirs.append(Path(summary["traj_dir"]).expanduser().resolve())
        planner_summaries.append(
            {
                "planner": planner,
                "traj_dir": summary["traj_dir"],
                "suite_dir": summary["suite_dir"],
                "tasks_count": summary["tasks_count"],
                "ok_rate": summary["ok_rate"],
            }
        )

    result: Dict[str, Any] = {
        "planners": planner_summaries,
        "traj_dirs": [str(path) for path in generated_traj_dirs],
    }

    if args.build_dataset_output:
        prompt_key = _dataset_prompt_key(args)
        if args.merge_datasets or len(planners) == 1:
            dataset_path = build_dataset(
                traj_dirs=generated_traj_dirs,
                prompt_path=args.prompt_path,
                prompt_key=prompt_key,
                output=args.build_dataset_output,
                output_mode=args.dataset_mode,
                max_trajs=args.dataset_max_trajs,
                max_steps=args.dataset_max_steps,
                max_samples=args.dataset_max_samples,
            )
            result["dataset_output"] = str(dataset_path)
        else:
            dataset_outputs: Dict[str, str] = {}
            for planner_summary in planner_summaries:
                planner = str(planner_summary["planner"])
                dataset_path = build_dataset(
                    traj_dirs=[planner_summary["traj_dir"]],
                    prompt_path=args.prompt_path,
                    prompt_key=prompt_key,
                    output=_dataset_output_for_planner(args.build_dataset_output, planner),
                    output_mode=args.dataset_mode,
                    max_trajs=args.dataset_max_trajs,
                    max_steps=args.dataset_max_steps,
                    max_samples=args.dataset_max_samples,
                )
                dataset_outputs[planner] = str(dataset_path)
            result["dataset_outputs"] = dataset_outputs

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
