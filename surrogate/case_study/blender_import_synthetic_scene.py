"""
用途:
  在 Blender 中读取 case_study 生成的 manifest，导入 PLY 楼体，创建地面、Tx 标记与基础相机，
  并保存为可继续编辑的 .blend 文件。

直接运行命令:
  /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --python \
    /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/case_study/blender_import_synthetic_scene.py -- \
    --manifest /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/case_study/output_dataset/blender/synthetic_scene_0_manifest.json \
    --output-blend /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/case_study/output_dataset/blender/synthetic_scene_0.blend

参数说明:
  --manifest: synthetic_scene_generator 导出的 manifest JSON。
  --output-blend: 输出 .blend 路径。
  --ground-margin-m: 地面相对世界边界额外扩展的边距。
  --tx-marker-size-m: 发射机标记立方体边长。

逻辑说明:
  脚本在 Blender Python 中执行。它会清空默认场景，导入 manifest 中列出的 PLY 建筑，
  再生成一个地面平面和一个 Tx 标记物体，最后保存为 .blend，便于后续人工编辑或继续导出。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a synthetic case-study scene into Blender.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-blend", type=Path, required=True)
    parser.add_argument("--ground-margin-m", type=float, default=8.0)
    parser.add_argument("--tx-marker-size-m", type=float, default=2.0)
    return parser.parse_args(argv)


def _argv_after_double_dash() -> list[str]:
    if "--" not in sys.argv:
        return []
    idx = sys.argv.index("--")
    return sys.argv[idx + 1 :]


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args(_argv_after_double_dash())

    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)

    manifest = _load_manifest(args.manifest)
    mesh_paths = [Path(p) for p in manifest["mesh_paths"]]
    tx_x, tx_y, tx_z = manifest["tx_position_m"]
    world_size = float(manifest["world_size_m"])

    imported_names: list[str] = []
    for mesh_path in mesh_paths:
        if not mesh_path.exists():
            raise FileNotFoundError(mesh_path)
        bpy.ops.import_mesh.ply(filepath=str(mesh_path))
        imported_names.extend(obj.name for obj in bpy.context.selected_objects)

    bpy.ops.mesh.primitive_plane_add(
        size=world_size + 2.0 * args.ground_margin_m,
        location=(world_size / 2.0, world_size / 2.0, 0.0),
    )
    ground = bpy.context.active_object
    ground.name = "Ground"

    bpy.ops.mesh.primitive_cube_add(
        size=args.tx_marker_size_m,
        location=(tx_x, tx_y, tx_z),
    )
    tx_marker = bpy.context.active_object
    tx_marker.name = "TxMarker"

    bpy.ops.object.light_add(type="SUN", location=(world_size * 0.5, world_size * 0.5, world_size))
    sun = bpy.context.active_object
    sun.name = "Sun"

    bpy.ops.object.camera_add(
        location=(world_size * 0.5, -world_size * 0.9, world_size * 0.9),
        rotation=(1.05, 0.0, 0.0),
    )
    camera = bpy.context.active_object
    camera.name = "OverviewCamera"
    bpy.context.scene.camera = camera

    args.output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output_blend))
    print(f"Imported {len(imported_names)} mesh objects")
    print(f"Saved blend to {args.output_blend}")


if __name__ == "__main__":
    main()
