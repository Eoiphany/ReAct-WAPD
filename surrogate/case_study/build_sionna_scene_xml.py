"""
用途:
  根据 case_study 导出的 Blender manifest 生成 Mitsuba/Sionna 可加载的 scene XML。

直接运行命令:
  python -m paper_experiment.surrogate.case_study.build_sionna_scene_xml
  python -m paper_experiment.surrogate.case_study.build_sionna_scene_xml \
    --manifest /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/case_study/output_dataset/blender/synthetic_scene_0_manifest.json \
    --output-xml /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/case_study/output_dataset/sionna/synthetic_scene.xml

参数说明:
  --manifest: synthetic_scene_generator 导出的 manifest JSON。
  --output-xml: 生成的 scene XML 路径。
  --ground-z: 地面矩形的 z 高度。
  --building-material-mode: 建筑材质分配策略；`uniform_concrete` 为当前主默认值，`alternating` 保持旧的交替材质，`uniform_*` 则把全部建筑统一成指定 ITU 材质。

逻辑说明:
  该脚本不依赖 Blender GUI；它直接把 manifest 中记录的 mesh 列表写成 Mitsuba scene XML，
  并补一个覆盖整个世界范围的矩形地面。生成结果可直接交给 Sionna 的 load_scene() 使用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .case_study_paths import BLENDER_ROOT, SIONNA_ROOT


DEFAULT_MANIFEST = BLENDER_ROOT / "synthetic_scene_0_manifest.json"
DEFAULT_OUTPUT_XML = SIONNA_ROOT / "synthetic_scene.xml"
SUPPORTED_ITU_BUILDING_TYPES = ("concrete", "marble", "metal", "brick", "wood", "glass", "plywood")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _shape_xml(mesh_relpath: str, mesh_id: str, material_id: str) -> str:
    return (
        f'\t<shape type="ply" id="{mesh_id}" name="{mesh_id}">\n'
        f'\t\t<string name="filename" value="{mesh_relpath}"/>\n'
        '\t\t<boolean name="face_normals" value="true"/>\n'
        f'\t\t<ref id="{material_id}" name="bsdf"/>\n'
        '\t</shape>\n'
    )


def _resolve_building_material(idx: int, mode: str) -> str:
    if mode.startswith("uniform_"):
        itu_type = mode.removeprefix("uniform_")
        if itu_type not in SUPPORTED_ITU_BUILDING_TYPES:
            raise ValueError(f"Unsupported ITU material type: {itu_type}")
        return f"mat-itu_{itu_type}"
    return "mat-itu_marble" if idx % 2 == 0 else "mat-itu_metal"


def build_scene_xml_from_manifest(
    manifest_path: Path,
    output_xml_path: Path,
    ground_z: float = 0.0,
    building_material_mode: str = "uniform_concrete",
) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mesh_paths = [Path(p) for p in manifest["mesh_paths"]]
    world_size_m = float(manifest["world_size_m"])

    output_xml_path = Path(output_xml_path)
    ensure_dir(output_xml_path.parent)

    shapes: list[str] = []
    for idx, mesh_path in enumerate(mesh_paths):
        mesh_rel = Path("..") / "meshes" / mesh_path.name
        material_id = _resolve_building_material(idx, building_material_mode)
        shapes.append(_shape_xml(mesh_rel.as_posix(), f"mesh-{mesh_path.stem}", material_id))

    plane_scale = world_size_m / 2.0
    plane_translate = world_size_m / 2.0
    ground_xml = (
        '\t<shape type="rectangle" id="ground" name="ground">\n'
        '\t\t<transform name="to_world">\n'
        f'\t\t\t<scale x="{plane_scale:.6f}" y="{plane_scale:.6f}" z="1.0"/>\n'
        f'\t\t\t<translate x="{plane_translate:.6f}" y="{plane_translate:.6f}" z="{ground_z:.6f}"/>\n'
        '\t\t</transform>\n'
        '\t\t<ref id="mat-itu_concrete" name="bsdf"/>\n'
        '\t</shape>\n'
    )

    xml = (
        '<scene version="2.1.0">\n\n'
        '\t<integrator type="path" id="elm__0" name="elm__0">\n'
        '\t\t<integer name="max_depth" value="12"/>\n'
        '\t</integrator>\n\n'
        '\t<bsdf type="twosided" id="mat-itu_concrete" name="mat-itu_concrete">\n'
        '\t\t<bsdf type="diffuse" name="bsdf">\n'
        '\t\t\t<rgb value="0.085326 0.085326 0.085326" name="reflectance"/>\n'
        '\t\t</bsdf>\n'
        '\t</bsdf>\n'
        '\t<bsdf type="diffuse" id="mat-itu_marble" name="mat-itu_marble">\n'
        '\t\t<rgb value="0.466965 0.481934 0.466065" name="reflectance"/>\n'
        '\t</bsdf>\n'
        '\t<bsdf type="diffuse" id="mat-itu_metal" name="mat-itu_metal">\n'
        '\t\t<rgb value="0.112802 0.097931 0.082988" name="reflectance"/>\n'
        '\t</bsdf>\n\n'
        '\t<bsdf type="diffuse" id="mat-itu_brick" name="mat-itu_brick">\n'
        '\t\t<rgb value="0.402000 0.112000 0.087000" name="reflectance"/>\n'
        '\t</bsdf>\n'
        '\t<bsdf type="diffuse" id="mat-itu_wood" name="mat-itu_wood">\n'
        '\t\t<rgb value="0.266000 0.109000 0.060000" name="reflectance"/>\n'
        '\t</bsdf>\n'
        '\t<bsdf type="diffuse" id="mat-itu_glass" name="mat-itu_glass">\n'
        '\t\t<rgb value="0.168000 0.139000 0.509000" name="reflectance"/>\n'
        '\t</bsdf>\n'
        '\t<bsdf type="diffuse" id="mat-itu_plywood" name="mat-itu_plywood">\n'
        '\t\t<rgb value="0.136000 0.076000 0.539000" name="reflectance"/>\n'
        '\t</bsdf>\n\n'
        '\t<emitter type="constant" id="World" name="World">\n'
        '\t\t<rgb value="1.000000 1.000000 1.000000" name="radiance"/>\n'
        '\t</emitter>\n\n'
        f"{ground_xml}\n"
        f"{''.join(shapes)}"
        '</scene>\n'
    )
    output_xml_path.write_text(xml, encoding="utf-8")
    return output_xml_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Mitsuba/Sionna scene XML from the synthetic scene manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-xml", type=Path, default=DEFAULT_OUTPUT_XML)
    parser.add_argument("--ground-z", type=float, default=0.0)
    parser.add_argument(
        "--building-material-mode",
        choices=["alternating", *[f"uniform_{name}" for name in SUPPORTED_ITU_BUILDING_TYPES]],
        default="uniform_concrete",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = build_scene_xml_from_manifest(
        args.manifest,
        args.output_xml,
        ground_z=args.ground_z,
        building_material_mode=args.building_material_mode,
    )
    print(f"scene_xml={output}")


if __name__ == "__main__":
    main()
