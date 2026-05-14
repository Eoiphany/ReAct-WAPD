"""注释
命令示例:
python -m ReAct.experiments.run_exp1_fixed_vs_unfixed --group heuristic
python -m ReAct.experiments.run_exp1_fixed_vs_unfixed --group llm --setting fixed --eval-device cuda
python -m ReAct.experiments.run_exp1_fixed_vs_unfixed --group llm --setting unfixed --eval-device cuda --num-maps 1

参数说明:
- --maps-dir: 实验 1 默认测试地图目录。
- --num-maps: 参与实验的地图数量。
- --fixed-sites: 固定站点实验的精确站点数。
- --unfixed-site-limit: 非固定站点实验的最大站点数阈值。
- --heuristic-search-budget: clustered heuristic 在单张地图上的统一搜索预算。
- --use-heuristic-cache: 是否允许直接复用既有 heuristic cache；若要重建 exp1 正式时间，应显式关闭。
- --group: `heuristic` 只跑启发式，`llm` 只跑 Qwen/微调 Qwen，`all` 全部执行。
- --setting: `fixed` 只跑固定站点，`unfixed` 只跑非固定站点，`all` 全部执行。
- --eval-device: rmnet/Qwen/LLaMA-Factory 推理设备。

Root cause: _QWEN_CACHE 持有前序 planner 的 Qwen 实例未释放，导致同进程内重复加载 Qwen2.5-7B（+LoRA）时显存叠加，触发 OOM。
candidate_sample是这一步先从全体合法候选站点里抽多少个出来做真实评分，llm_top_k_candidates是候选评分排完序之后，最终只把前 k 个候选发给 LLM 看



命令示例:

python -m ReAct.experiments.run_exp1_fixed_vs_unfixed --group heuristic
conda activate qwen
python -m ReAct.experiments.run_exp1_fixed_vs_unfixed --group llm --eval-device cuda
cd autodl-tmp && python -m ReAct.experiments.run_exp1_fixed_vs_unfixed --group llm --eval-device cuda

conda activate qwen
cd autodl-tmp && python -m ReAct.experiments.run_exp1_fixed_vs_unfixed --group llm --setting unfixed --eval-device cuda --num-maps 1


逻辑说明:
该脚本固定使用 TSPL 初始化，分别在 fixed 与 unfixed 两种需求下切换启发式与 LLM 决策方法，
输出与你第一张图对应的结果表。
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
    render_fixed_request,
    render_unfixed_request,
    run_named_suite,
    summary_to_metric_row,
    write_table_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    planner_choices = ["all", "heuristic_greedy", "heuristic_ga", "heuristic_pso", "heuristic_sa", "qwen", "llamafactory"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-dir", default=str(DEFAULT_RADIOMAP3DSEER_MAPS_DIR))
    parser.add_argument("--maps-list", default=str(DEFAULT_TEST_MAPS_LIST))
    parser.add_argument("--num-maps", type=int, default=100)
    parser.add_argument("--fixed-sites", type=int, default=4)
    parser.add_argument("--unfixed-site-limit", type=int, default=6)
    parser.add_argument("--heuristic-search-budget", type=int, default=600)
    parser.add_argument("--use-heuristic-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--group", choices=["all", "heuristic", "llm"], default="all")
    parser.add_argument("--setting", choices=["all", "fixed", "unfixed"], default="all")
    parser.add_argument("--planner", choices=planner_choices, default="all")
    parser.add_argument("--llm-mode", choices=["all", "decide", "explain_weighted"], default="all")
    parser.add_argument("--init-mode", choices=["two_stage", "random", "greedy", "none"], default="two_stage")
    parser.add_argument("--eval-device", choices=["auto", "cpu", "cuda", "mps"], default="mps")
    parser.add_argument("--output-root", default=str(DEFAULT_EXPERIMENT_OUTPUT_ROOT / "exp1_fixed_vs_unfixed"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).resolve()
    request_fixed = ensure_request_file(output_root, "fixed_sites_4.txt", render_fixed_request(args.fixed_sites))
    request_unfixed = ensure_request_file(output_root, "coverage_capacity_budget_site6.txt", render_unfixed_request(args.unfixed_site_limit))

    heuristic_planners = [
        ("heuristic_greedy", "decide"),
        ("heuristic_ga", "decide"),
        ("heuristic_pso", "decide"),
        ("heuristic_sa", "decide"),
    ]
    llm_planners = [
        ("qwen", "decide"),
        ("llamafactory", "decide"),
        ("llamafactory", "explain_weighted"),
    ]
    if args.planner != "all":
        heuristic_planners = [item for item in heuristic_planners if item[0] == args.planner]
        llm_planners = [item for item in llm_planners if item[0] == args.planner]
    if args.llm_mode != "all":
        heuristic_planners = [item for item in heuristic_planners if item[1] == args.llm_mode]
        llm_planners = [item for item in llm_planners if item[1] == args.llm_mode]
    settings = [
        ("fixed", request_fixed),
        ("unfixed", request_unfixed),
    ]
    if args.setting != "all":
        settings = [item for item in settings if item[0] == args.setting]

    rows = []
    for setting_name, request_path in settings:
        if args.group in {"all", "heuristic"}:
            print(status_line("GROUP", f"exp1 group=heuristic setting={setting_name} request={request_path.name}", tone="accent"))
            for planner, llm_mode in heuristic_planners:
                print(
                    status_line(
                        "INFO",
                        f"setting={setting_name} init=TSPL planner={planner_label(planner, llm_mode)} llm_mode={llm_mode}",
                        tone="info",
                    )
                )
                suite_args = make_suite_args(
                    maps_dir=Path(args.maps_dir),
                    maps_list=Path(args.maps_list) if args.maps_list else None,
                    num_maps=args.num_maps,
                    request_file=request_path,
                    output_root=output_root / "runs" / "heuristic",
                    suite_name=f"{setting_name}_{planner}_{llm_mode}",
                    planner=planner,
                    eval_device=args.eval_device,
                    init_mode=args.init_mode,
                    two_stage_init_k=1,
                    llm_decision_mode=llm_mode,
                    candidate_sample=16,
                    llm_top_k_candidates=8,
                    heuristic_search_budget=args.heuristic_search_budget,
                    use_heuristic_cache=args.use_heuristic_cache,
                )
                summary = run_named_suite(suite_args)
                metric_row = summary_to_metric_row(summary)
                row = {
                    "Setting": setting_name,
                    "Method": planner_label(planner, llm_mode),
                }
                row.update(metric_row)
                rows.append(row)

        if args.group in {"all", "llm"}:
            print(status_line("GROUP", f"exp1 group=llm setting={setting_name} request={request_path.name}", tone="accent"))
            for planner, llm_mode in llm_planners:
                print(
                    status_line(
                        "INFO",
                        f"setting={setting_name} init=TSPL planner={planner_label(planner, llm_mode)} llm_mode={llm_mode}",
                        tone="info",
                    )
                )
                suite_args = make_suite_args(
                    maps_dir=Path(args.maps_dir),
                    maps_list=Path(args.maps_list) if args.maps_list else None,
                    num_maps=args.num_maps,
                    request_file=request_path,
                    output_root=output_root / "runs" / "llm",
                    suite_name=f"{setting_name}_{planner}_{llm_mode}",
                    planner=planner,
                    eval_device=args.eval_device,
                    init_mode=args.init_mode,
                    two_stage_init_k=1,
                    llm_decision_mode=llm_mode,
                    candidate_sample=16,
                    llm_top_k_candidates=8,
                )
                summary = run_named_suite(suite_args)
                metric_row = summary_to_metric_row(summary)
                row = {
                    "Setting": setting_name,
                    "Method": planner_label(planner, llm_mode),
                }
                row.update(metric_row)
                rows.append(row)

    stem = "exp1_fixed_vs_unfixed"
    if args.group != "all":
        stem += f"_{args.group}"
    if args.setting != "all":
        stem += f"_{args.setting}"
    title = "Experiment 1 Fixed vs Unfixed"
    if args.group != "all" or args.setting != "all":
        title += f" ({args.group}, {args.setting})"

    write_table_outputs(
        output_dir=output_root,
        stem=stem,
        rows=rows,
        fieldnames=["Setting", "Method", *TABLE_COLUMNS],
        title=title,
    )


if __name__ == "__main__":
    main()
