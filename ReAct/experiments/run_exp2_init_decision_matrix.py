"""注释
命令示例:
python -m ReAct.experiments.run_exp2_init_decision_matrix
python -m ReAct.experiments.run_exp2_init_decision_matrix --group heuristic --num-maps 1
python -m ReAct.experiments.run_exp2_init_decision_matrix --group llm --num-maps 1 --candidate-sample 32 --llm-top-k-candidates 16
python -m ReAct.experiments.run_exp2_init_decision_matrix --num-maps 20 --candidate-sample 32 --llm-top-k-candidates 16

参数说明:
- --maps-dir: 实验 2 使用的地图目录。
- --num-maps: 参与实验的地图数量。
- --site-limit: 特定需求下允许的最大站点数。
- --heuristic-search-budget: Exhaustive 重新运行时使用的搜索预算；其余启发式默认走 exp1 cache。
- --use-heuristic-cache: 是否允许 exp2 直接复用既有 heuristic cache；启用时 Exhaustive 仍会被脚本内强制 fresh run。
- --group: `heuristic` 只跑启发式+穷举，`llm` 只跑调用 LLM 的组合，`all` 全部执行。
- --candidate-sample: 每步从合法动作中抽取并由代理模型评分的候选数量，默认 32。
- --llm-top-k-candidates: 评分后提供给 LLM 的高分候选数量，默认 16。
- --eval-device: 代理模型、模型初始化、Qwen 与 LLaMA-Factory 的推理设备，默认 CUDA。

逻辑说明:
该脚本按第二张图的矩阵组织实验：上半部分是“初始化方法与决策方法相同”的纯启发式/穷举组，
下半部分是不同初始化方法与 LLM 决策方法的组合组。
其中 exp2 的启发式组会先使用对应方法完成一次初始化，再由同名启发式继续决策；
`Exhaustive` 也会先执行一次穷举初始化，再进入穷举决策阶段。
实验目录与汇总文件名包含 `cs<候选数>_topk<保留数>` 后缀，避免不同候选配置相互覆盖。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ReAct.cli_colors import status_line
from ReAct.experiments.summary_utils import (
    DEFAULT_EXPERIMENT_OUTPUT_ROOT,
    DEFAULT_RADIOMAP3DSEER_MAPS_DIR,
    DEFAULT_TEST_MAPS_LIST,
    TABLE_COLUMNS,
    ensure_request_file,
    init_label,
    make_suite_args,
    planner_label,
    render_unfixed_request,
    run_named_suite,
    summary_to_metric_row,
    write_table_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    planner_choices = ["all", "heuristic_greedy", "heuristic_ga", "heuristic_pso", "heuristic_sa", "heuristic_exhaustive", "qwen", "llamafactory"]
    init_choices = ["all", "random", "greedy", "heuristic_ga", "heuristic_pso", "heuristic_sa", "two_stage"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-dir", default=str(DEFAULT_RADIOMAP3DSEER_MAPS_DIR))
    parser.add_argument("--maps-list", default=str(DEFAULT_TEST_MAPS_LIST))
    parser.add_argument("--num-maps", type=int, default=100)
    parser.add_argument("--site-limit", type=int, default=6)
    parser.add_argument("--heuristic-search-budget", type=int, default=600)
    parser.add_argument("--use-heuristic-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--group", choices=["all", "heuristic", "llm"], default="all")
    parser.add_argument("--planner", choices=planner_choices, default="all")
    parser.add_argument("--llm-mode", choices=["all", "decide", "explain_weighted"], default="all")
    parser.add_argument("--init-mode", choices=init_choices, default="all")
    parser.add_argument("--candidate-sample", type=int, default=32)
    parser.add_argument("--llm-top-k-candidates", type=int, default=16)
    parser.add_argument("--eval-device", choices=["auto", "cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--output-root", default=str(DEFAULT_EXPERIMENT_OUTPUT_ROOT / "exp2_init_decision_matrix"))
    return parser


def candidate_config_suffix(candidate_sample: int, llm_top_k_candidates: int) -> str:
    return f"cs{int(candidate_sample)}_topk{int(llm_top_k_candidates)}"


def experiment_suite_name(prefix: str, candidate_sample: int, llm_top_k_candidates: int) -> str:
    return f"{prefix}_{candidate_config_suffix(candidate_sample, llm_top_k_candidates)}"


def summary_output_stem(group: str, candidate_sample: int, llm_top_k_candidates: int) -> str:
    prefix = "exp2_init_decision_matrix" if group == "all" else f"exp2_init_decision_matrix_{group}"
    return experiment_suite_name(prefix, candidate_sample, llm_top_k_candidates)


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).resolve()
    request_path = ensure_request_file(output_root, "coverage_capacity_budget_site6.txt", render_unfixed_request(args.site_limit))

    rows = []
    same_method_planners = ["heuristic_greedy", "heuristic_ga", "heuristic_pso", "heuristic_sa", "heuristic_exhaustive"]
    exp1_unfixed_greedy_traj_dir = (
        DEFAULT_EXPERIMENT_OUTPUT_ROOT / "exp1_fixed_vs_unfixed" / "runs" / "heuristic" / "unfixed_heuristic_greedy_decide" / "trajs"
    )
    if args.planner != "all":
        same_method_planners = [planner for planner in same_method_planners if planner == args.planner]
    if args.group in {"all", "heuristic"}:
        print(status_line("GROUP", "exp2 group=heuristic_same_method", tone="accent"))
        for planner in same_method_planners:
            print(status_line("INFO", f"init={planner_label(planner)} decision=same_method", tone="info"))
            use_heuristic_cache = bool(args.use_heuristic_cache) and planner != "heuristic_exhaustive"
            if planner == "heuristic_greedy":
                heuristic_init_mode = "greedy"
            elif planner in {"heuristic_ga", "heuristic_pso", "heuristic_sa", "heuristic_exhaustive"}:
                heuristic_init_mode = planner
            else:
                heuristic_init_mode = "none"
            replay_traj_dir = exp1_unfixed_greedy_traj_dir if planner == "heuristic_greedy" else None
            suite_args = make_suite_args(
                maps_dir=Path(args.maps_dir),
                maps_list=Path(args.maps_list) if args.maps_list else None,
                num_maps=args.num_maps,
                request_file=request_path,
                output_root=output_root / "runs" / "heuristic",
                suite_name=experiment_suite_name(
                    f"same_{planner}", args.candidate_sample, args.llm_top_k_candidates
                ),
                planner=planner,
                eval_device=args.eval_device,
                init_mode=heuristic_init_mode,
                llm_decision_mode="decide",
                candidate_sample=args.candidate_sample,
                llm_top_k_candidates=args.llm_top_k_candidates,
                heuristic_search_budget=args.heuristic_search_budget,
                use_heuristic_cache=use_heuristic_cache,
                replay_traj_dir=replay_traj_dir,
            )
            summary = run_named_suite(suite_args)
            row = {
                "Init Method": planner_label(planner),
                "Decision Method": "",
            }
            row.update(summary_to_metric_row(summary))
            rows.append(row)

    init_modes = ["random", "greedy", "heuristic_ga", "heuristic_pso", "heuristic_sa", "two_stage"]
    llm_groups = [
        ("qwen", "decide"),
        ("llamafactory", "decide"),
        ("llamafactory", "explain_weighted"),
    ]
    if args.init_mode != "all":
        init_modes = [mode for mode in init_modes if mode == args.init_mode]
    if args.planner != "all":
        llm_groups = [item for item in llm_groups if item[0] == args.planner]
    if args.llm_mode != "all":
        llm_groups = [item for item in llm_groups if item[1] == args.llm_mode]
    if args.group in {"all", "llm"}:
        print(status_line("GROUP", "exp2 group=llm_cross_matrix", tone="accent"))
        for planner, llm_mode in llm_groups:
            for init_mode in init_modes:
                print(
                    status_line(
                        "INFO",
                        f"init={init_label(init_mode)} decision={planner_label(planner, llm_mode)} llm_mode={llm_mode}",
                        tone="info",
                    )
                )
                suite_args = make_suite_args(
                    maps_dir=Path(args.maps_dir),
                    maps_list=Path(args.maps_list) if args.maps_list else None,
                    num_maps=args.num_maps,
                    request_file=request_path,
                    output_root=output_root / "runs" / "llm",
                    suite_name=experiment_suite_name(
                        f"{init_mode}_{planner}_{llm_mode}",
                        args.candidate_sample,
                        args.llm_top_k_candidates,
                    ),
                    planner=planner,
                    eval_device=args.eval_device,
                    init_mode=init_mode,
                    init_k=1,
                    two_stage_init_k=1,
                    llm_decision_mode=llm_mode,
                    candidate_sample=args.candidate_sample,
                    llm_top_k_candidates=args.llm_top_k_candidates,
                )
                summary = run_named_suite(suite_args)
                row = {
                    "Init Method": init_label(init_mode),
                    "Decision Method": planner_label(planner, llm_mode),
                }
                row.update(summary_to_metric_row(summary))
                rows.append(row)

    stem = summary_output_stem(args.group, args.candidate_sample, args.llm_top_k_candidates)
    title = "Experiment 2 Init Decision Matrix" if args.group == "all" else f"Experiment 2 Init Decision Matrix ({args.group})"

    write_table_outputs(
        output_dir=output_root,
        stem=stem,
        rows=rows,
        fieldnames=["Init Method", "Decision Method", *TABLE_COLUMNS],
        title=title,
    )


if __name__ == "__main__":
    main()
