"""注释
命令示例:
python visualization/generate_live_roi_coverage.py --map-path /abs/path/to/map.png --sites-file /abs/path/to/sites.json --output-path /abs/path/to/out.png

参数含义:
- `--map-path`: 当前地图路径。
- `--sites-file`: 当前 step 的站点集合 JSON。
- `--output-path`: 左图输出路径。

逻辑说明:
本脚本是 visualization 目录下的左图实时包装入口，读取当前站点集合后转成 overlay 点，调用现有 RoI/coverage 脚本生成当前展示图。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from visualization.live_step_renderer import render_roi_coverage
else:
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_PARENT = CURRENT_DIR.parent
    if str(PROJECT_PARENT) not in sys.path:
        sys.path.insert(0, str(PROJECT_PARENT))
    from visualization.live_step_renderer import render_roi_coverage

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-path", required=True)
    parser.add_argument("--sites-file", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--eval-model", choices=["pmnet", "rmnet"], default="rmnet")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sites_payload = json.loads(Path(args.sites_file).read_text(encoding="utf-8"))
    sites = sites_payload.get("sites", [])
    render_roi_coverage(
        map_path=Path(args.map_path).resolve(),
        sites=sites,
        output_path=Path(args.output_path).resolve(),
        eval_model=args.eval_model,
    )
    print(f"saved_roi={Path(args.output_path).resolve()}")


if __name__ == "__main__":
    main()
