"""注释
命令示例:
python visualization/generate_live_deployment_pred.py --map-path /abs/path/to/map.png --sites-file /abs/path/to/sites.json --output-dir /abs/path/to/pred_dir

参数含义:
- `--map-path`: 当前地图路径。
- `--sites-file`: 当前 step 的站点集合 JSON。
- `--output-dir`: 右图输出目录。

逻辑说明:
本脚本是 visualization 目录下的右图实时包装入口，复用现有 eval_radiomap 合并站点预测链路，输出与示例风格一致的部署效果图。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import json

if __package__:
    from visualization.live_step_renderer import render_deployment_prediction
else:
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_PARENT = CURRENT_DIR.parent
    if str(PROJECT_PARENT) not in sys.path:
        sys.path.insert(0, str(PROJECT_PARENT))
    from visualization.live_step_renderer import render_deployment_prediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-path", required=True)
    parser.add_argument("--sites-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-model", choices=["pmnet", "rmnet"], default="rmnet")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sites_payload = json.loads(Path(args.sites_file).read_text(encoding="utf-8"))
    output_path = output_dir / "latest_pred.png"
    render_deployment_prediction(
        map_path=Path(args.map_path).resolve(),
        sites=sites_payload.get("sites", []),
        output_path=output_path,
        eval_model=args.eval_model,
    )
    print(f"saved_pred={output_path}")


if __name__ == "__main__":
    main()
