"""
用途:
  为无线接入点决策批量生成 map/request 组合任务清单，可选直接顺序执行主入口脚本。

示例命令:
  仅生成 task list:
    python ReAct/generate_decision_tasks.py \
      --maps-dir ../test/dataset/png/buildingsWHeight \
      --requests-dir ReAct/requests \
      --output-task-list ReAct/outputs/task_list.jsonl
  生成并执行:
    python ReAct/generate_decision_tasks.py \
      --maps-dir ../test/dataset/png/buildingsWHeight \
      --requests-dir ReAct/requests \
      --run \
      --planner heuristic

参数说明:
  --maps-dir: 地图目录。
  --maps-list: 地图列表文件，每行一个路径。
  --num-maps: 只取前多少张地图，0 表示全部。
  --requests-dir: 需求文本目录。
  --output-task-list: 输出 task_list.jsonl 路径。
  --run: 生成后顺序执行任务。
  --planner: 执行时使用的规划器。
  --traj-dir: 执行时轨迹输出目录。
  --max-steps: 执行时最大步数。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from env_utils import resolve_city_map_paths


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_TASK_LIST = ROOT_DIR / "outputs" / "task_list.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-dir", default="")
    parser.add_argument("--maps-list", default="")
    parser.add_argument("--num-maps", type=int, default=0)
    parser.add_argument("--requests-dir", required=True)
    parser.add_argument("--output-task-list", default=str(DEFAULT_TASK_LIST))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--planner", choices=["heuristic", "random", "openai"], default="heuristic")
    parser.add_argument("--traj-dir", default=str(ROOT_DIR / "outputs" / "trajs"))
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()

    if args.maps_list:
        list_path = Path(args.maps_list)
        maps = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif args.maps_dir:
        maps = resolve_city_map_paths(args.maps_dir, default_path=args.maps_dir)
    else:
        raise ValueError("--maps-dir or --maps-list is required")

    if args.num_maps > 0:
        maps = maps[: args.num_maps]

    req_path = Path(args.requests_dir)
    requests = sorted(str(path) for path in req_path.glob("*.txt"))
    if not requests:
        raise FileNotFoundError(f"No request files found in {req_path}")

    output_path = Path(args.output_task_list)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = []
    with output_path.open("w", encoding="utf-8") as handle:
        for map_path in maps:
            for req in requests:
                task = {"city_map_path": str(map_path), "user_request_path": str(req)}
                tasks.append(task)
                handle.write(json.dumps(task, ensure_ascii=False) + "\n")

    if not args.run:
        print(f"Saved task list to: {output_path}")
        return

    run_script = ROOT_DIR / "run_access_point_decision.py"
    for task in tasks:
        cmd = [
            sys.executable,
            str(run_script),
            "--city-map-path",
            task["city_map_path"],
            "--user-request-path",
            task["user_request_path"],
            "--planner",
            args.planner,
            "--traj-dir",
            args.traj_dir,
            "--max-steps",
            str(args.max_steps),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
