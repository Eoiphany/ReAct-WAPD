"""注释
命令示例:
python -m ReAct.experiments.run_exp3_generalization
python -m ReAct.experiments.run_exp3_generalization --dataset radiomap3dseer --num-maps 100 --eval-device cuda
python -m ReAct.experiments.run_exp3_generalization --dataset usc --num-maps 100 --eval-device cuda

参数说明:
- --radiomap3dseer-maps-dir: RadioMap3DSeer 建筑高度图目录。
- --radiomap3dseer-maps-list: RadioMap3DSeer 测试集列表。
- --usc-data-root: USC 数据集根目录，内部应包含 map/、Tx/、pmap/。
- --usc-maps-dir: USC 地图目录；通常为 usc-data/map。
- --dataset: `radiomap3dseer`、`usc` 或 `all`。
- --num-maps: 每个数据集评估的地图数量。
- --site-limit: 非固定站点任务的最大站点数。
- --eval-device: 代理模型与 LLM 推理设备。

逻辑说明:
该脚本用于实验 3，验证同一无线接入配置下 LLM-FT-ReAct 在不同场景数据集上的跨场景泛化能力。
脚本不使用 TSPL 初始化，统一采用空初始化直接进入闭环决策；RadioMap3DSeer 使用既有测试列表，
USC 使用与代理模型训练一致的固定随机种子划分得到测试集样本列表，然后分别调用统一的 batch suite。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ReAct.cli_colors import status_line
from ReAct.experiments.summary_utils import (
    DEFAULT_EXPERIMENT_OUTPUT_ROOT,
    DEFAULT_QWEN_MODEL_PATH,
    DEFAULT_RADIOMAP3DSEER_MAPS_DIR,
    DEFAULT_TEST_MAPS_LIST,
    TABLE_COLUMNS,
    ensure_request_file,
    make_suite_args,
    planner_label,
    render_unfixed_request,
    run_named_suite,
    summary_to_metric_row,
    write_path_list,
    write_table_outputs,
)
from surrogate.data_surrogate import resolve_usc_sample_ids
from surrogate.train_usc_surrogate import split_sample_ids_deterministically


REACT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REACT_ROOT.parent
DEFAULT_USC_DATA_ROOT = PROJECT_ROOT / "usc-data"
DEFAULT_USC_MAPS_DIR = DEFAULT_USC_DATA_ROOT / "map"
DEFAULT_RMNET_RADIOMAP3DSEER = PROJECT_ROOT / "surrogate" / "checkpoints" / "rmnet_radiomap3dseer.pt"
DEFAULT_RMNET_USC = PROJECT_ROOT / "surrogate" / "checkpoints" / "rmnet_usc.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radiomap3dseer-maps-dir", default=str(DEFAULT_RADIOMAP3DSEER_MAPS_DIR))
    parser.add_argument("--radiomap3dseer-maps-list", default=str(DEFAULT_TEST_MAPS_LIST))
    parser.add_argument("--usc-data-root", default=str(DEFAULT_USC_DATA_ROOT))
    parser.add_argument("--usc-maps-dir", default=str(DEFAULT_USC_MAPS_DIR))
    parser.add_argument("--dataset", choices=["all", "radiomap3dseer", "usc"], default="all")
    parser.add_argument("--num-maps", type=int, default=100)
    parser.add_argument("--site-limit", type=int, default=6)
    parser.add_argument("--eval-device", choices=["auto", "cpu", "cuda", "mps"], default="mps")
    parser.add_argument("--output-root", default=str(DEFAULT_EXPERIMENT_OUTPUT_ROOT / "exp3_generalization"))
    return parser


def _build_usc_test_list(output_root: Path, usc_data_root: Path, usc_maps_dir: Path, num_maps: int) -> Path:
    sample_ids = resolve_usc_sample_ids(str(usc_data_root), None)
    _, _, test_ids = split_sample_ids_deterministically(
        sample_ids,
        train_ratio=0.7,
        test_ratio=0.2,
        seed=42,
    )
    selected_ids = test_ids[: max(1, int(num_maps))]
    map_paths = [usc_maps_dir / f"{sample_id}.png" for sample_id in selected_ids]
    return write_path_list(output_root, "usc_test_paths.txt", map_paths)


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).resolve()
    request_path = ensure_request_file(
        output_root,
        "coverage_capacity_budget_site6.txt",
        render_unfixed_request(args.site_limit),
    )

    datasets = []
    if args.dataset in {"all", "radiomap3dseer"}:
        datasets.append(
            {
                "name": "RadioMap3DSeer",
                "maps_dir": Path(args.radiomap3dseer_maps_dir),
                "maps_list": Path(args.radiomap3dseer_maps_list),
                "eval_model_path": DEFAULT_RMNET_RADIOMAP3DSEER,
            }
        )
    if args.dataset in {"all", "usc"}:
        usc_list = _build_usc_test_list(
            output_root=output_root,
            usc_data_root=Path(args.usc_data_root),
            usc_maps_dir=Path(args.usc_maps_dir),
            num_maps=args.num_maps,
        )
        datasets.append(
            {
                "name": "USC-data",
                "maps_dir": Path(args.usc_maps_dir),
                "maps_list": usc_list,
                "eval_model_path": DEFAULT_RMNET_USC,
            }
        )

    rows = []
    for dataset_spec in datasets:
        print(
            status_line(
                "GROUP",
                f"exp3 dataset={dataset_spec['name']} method={planner_label('llamafactory', 'explain_weighted')}",
                tone="accent",
            )
        )
        suite_args = make_suite_args(
            maps_dir=dataset_spec["maps_dir"],
            maps_list=dataset_spec["maps_list"],
            num_maps=args.num_maps,
            request_file=request_path,
            output_root=output_root / "runs" / dataset_spec["name"],
            suite_name=f"{dataset_spec['name']}_llm_ft_react",
            planner="llamafactory",
            eval_device=args.eval_device,
            init_mode="none",
            llm_decision_mode="explain_weighted",
            candidate_sample=16,
            llm_top_k_candidates=8,
            qwen_model_path=DEFAULT_QWEN_MODEL_PATH,
            llamafactory_model=DEFAULT_QWEN_MODEL_PATH,
            eval_model_path=dataset_spec["eval_model_path"],
        )
        summary = run_named_suite(suite_args)
        row = {
            "Dataset": dataset_spec["name"],
            "Method": planner_label("llamafactory", "explain_weighted"),
        }
        row.update(summary_to_metric_row(summary))
        rows.append(row)

    write_table_outputs(
        output_dir=output_root,
        stem="exp3_generalization",
        rows=rows,
        fieldnames=["Dataset", "Method", *TABLE_COLUMNS],
        title="Experiment 3 Generalization",
    )


if __name__ == "__main__":
    main()
