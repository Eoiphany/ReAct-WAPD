from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring
import html


OUT_DIR = Path(__file__).resolve().parents[1] / "diagram_outputs"
SVG_PATH = OUT_DIR / "unified_visualization_flow.svg"
VDX_PATH = OUT_DIR / "unified_visualization_flow.vdx"


@dataclass
class Node:
    id: str
    text: str
    x: int
    y: int
    w: int
    h: int
    kind: str = "process"  # process, decision, terminator

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2

    @property
    def top(self) -> tuple[int, int]:
        return (self.cx, self.y)

    @property
    def bottom(self) -> tuple[int, int]:
        return (self.cx, self.y + self.h)

    @property
    def left(self) -> tuple[int, int]:
        return (self.x, self.cy)

    @property
    def right(self) -> tuple[int, int]:
        return (self.x + self.w, self.cy)


NODES = [
    Node("start", "开始", 1010, 40, 180, 60, "terminator"),
    Node("http", "启动HTTP服务", 980, 140, 240, 70),
    Node("home", "进入统一Web可视化界面", 920, 250, 360, 70),
    Node("page", "选择页面", 1010, 360, 180, 90, "decision"),

    Node("deploy_page", "进入部署可视化页面", 250, 360, 260, 70),
    Node("d1", "选择地图与需求文件", 250, 470, 260, 70),
    Node("d2", "执行需求结构化预览", 250, 570, 260, 70),
    Node("d3", "配置规划器、代理模型与搜索参数", 180, 670, 400, 70),
    Node("d4", "启动闭环部署任务", 250, 770, 260, 70),
    Node("d5", "后端构建命令并启动决策进程", 180, 870, 400, 70),
    Node("d6", "决策进程持续写入trajectory.json", 170, 970, 420, 70),
    Node("d7", "后端轮询轨迹并解析当前状态", 180, 1070, 400, 70),
    Node("d8", "渲染RoI覆盖率图、部署效果图与指标趋势图", 120, 1170, 520, 70),
    Node("d9", "SSE推送状态到前端", 250, 1270, 260, 70),
    Node("d10", "前端刷新流程状态、图像、表格与日志", 180, 1370, 400, 70),
    Node("d11", "任务是否结束", 250, 1470, 260, 90, "decision"),
    Node("d12", "展示最终部署结果", 250, 1600, 260, 70),
    Node("d13", "继续操作", 250, 1700, 260, 90, "decision"),

    Node("s_page", "进入代理模型可视化页面", 1500, 360, 300, 70),
    Node("s_mode", "选择模式", 1560, 470, 180, 90, "decision"),

    Node("t1", "预训练模式", 1370, 600, 240, 60, "terminator"),
    Node("t2", "配置数据集、模型与训练超参数", 1270, 700, 440, 70),
    Node("t3", "启动代理模型训练任务", 1370, 800, 240, 70),
    Node("t4", "后端启动训练进程", 1370, 900, 240, 70),
    Node("t5", "训练过程写入history.json等结果文件", 1250, 1000, 480, 70),
    Node("t6", "后端检测训练指标变化并生成训练曲线", 1240, 1100, 500, 70),
    Node("t7", "SSE推送训练状态与可视化结果", 1290, 1200, 400, 70),
    Node("t8", "前端刷新日志、摘要与训练指标图", 1300, 1300, 380, 70),
    Node("t9", "任务是否结束", 1370, 1400, 240, 90, "decision"),
    Node("t10", "展示训练结果与运行摘要", 1330, 1530, 320, 70),
    Node("t11", "继续操作", 1370, 1630, 240, 90, "decision"),

    Node("e1", "Checkpoint评测模式", 1850, 600, 260, 60, "terminator"),
    Node("e2", "填写Checkpoint输出目录", 1840, 700, 280, 70),
    Node("e3", "检测测试集并绑定样本", 1840, 800, 280, 70),
    Node("e4", "选择测试样本", 1840, 900, 280, 70),
    Node("e5", "添加一个或多个Checkpoint", 1810, 1000, 340, 70),
    Node("e6", "启动单样本多模型评测", 1840, 1100, 280, 70),
    Node("e7", "后端执行多Checkpoint预测", 1820, 1200, 320, 70),
    Node("e8", "生成预测对比图及RMSE、MAE、R2指标", 1760, 1300, 440, 70),
    Node("e9", "SSE推送评测状态与结果", 1840, 1400, 280, 70),
    Node("e10", "前端刷新对比图、指标表与模型列表", 1770, 1500, 420, 70),
    Node("e11", "任务是否结束", 1840, 1600, 280, 90, "decision"),
    Node("e12", "展示评测结果与模型对比结论", 1780, 1730, 400, 70),
    Node("e13", "继续操作", 1840, 1830, 280, 90, "decision"),

    Node("end", "结束", 1010, 1945, 180, 60, "terminator"),
]

NODE_MAP = {n.id: n for n in NODES}


def p(*pts: tuple[int, int]) -> str:
    return " ".join(f"{x},{y}" for x, y in pts)


CONNECTORS = [
    ("start", "bottom", [(1100, 140)]),
    ("http", "bottom", [(1100, 250)]),
    ("home", "bottom", [(1100, 360)]),
    ("page", "left", [(920, 405), (510, 405)]),
    ("page", "right", [(1280, 405), (1500, 405)]),

    ("deploy_page", "bottom", [(380, 470)]),
    ("d1", "bottom", [(380, 570)]),
    ("d2", "bottom", [(380, 670)]),
    ("d3", "bottom", [(380, 770)]),
    ("d4", "bottom", [(380, 870)]),
    ("d5", "bottom", [(380, 970)]),
    ("d6", "bottom", [(380, 1070)]),
    ("d7", "bottom", [(380, 1170)]),
    ("d8", "bottom", [(380, 1270)]),
    ("d9", "bottom", [(380, 1370)]),
    ("d10", "bottom", [(380, 1470)]),
    ("d11", "bottom", [(380, 1600)]),
    ("d12", "bottom", [(380, 1700)]),
    ("d13", "bottom", [(380, 1840), (380, 1975), (1010, 1975)]),

    ("s_page", "bottom", [(1650, 470)]),
    ("s_mode", "left", [(1370, 515)]),
    ("s_mode", "right", [(1850, 515)]),

    ("t1", "bottom", [(1490, 700)]),
    ("t2", "bottom", [(1490, 800)]),
    ("t3", "bottom", [(1490, 900)]),
    ("t4", "bottom", [(1490, 1000)]),
    ("t5", "bottom", [(1490, 1100)]),
    ("t6", "bottom", [(1490, 1200)]),
    ("t7", "bottom", [(1490, 1300)]),
    ("t8", "bottom", [(1490, 1400)]),
    ("t9", "bottom", [(1490, 1530)]),
    ("t10", "bottom", [(1490, 1630)]),
    ("t11", "bottom", [(1490, 1830), (1490, 1975), (1190, 1975)]),

    ("e1", "bottom", [(1980, 700)]),
    ("e2", "bottom", [(1980, 800)]),
    ("e3", "bottom", [(1980, 900)]),
    ("e4", "bottom", [(1980, 1000)]),
    ("e5", "bottom", [(1980, 1100)]),
    ("e6", "bottom", [(1980, 1200)]),
    ("e7", "bottom", [(1980, 1300)]),
    ("e8", "bottom", [(1980, 1400)]),
    ("e9", "bottom", [(1980, 1500)]),
    ("e10", "bottom", [(1980, 1600)]),
    ("e11", "bottom", [(1980, 1730)]),
    ("e12", "bottom", [(1980, 1830)]),
    ("e13", "bottom", [(1980, 1975), (1190, 1975)]),

    # loops and mode switches
    ("d11", "left", [(90, 1515), (90, 405), (1010, 405)]),   # 否，继续部署轮询
    ("t9", "left", [(1180, 1445), (1180, 1035), (1250, 1035)]),  # 否，继续训练
    ("e11", "left", [(1700, 1645), (1700, 1235), (1820, 1235)]),  # 否，继续评测

    ("d13", "left", [(90, 1745), (90, 285), (920, 285)]),   # 重置
    ("d13", "right", [(760, 1745), (760, 405), (1010, 405)]),  # 切换页面

    ("t11", "left", [(1180, 1675), (1180, 630), (1370, 630)]),  # 切换到预训练起点占位
    ("t11", "right", [(1740, 1675), (1740, 630), (1850, 630)]),  # 切换到评测模式
    ("t11", "top", [(1490, 590), (1100, 590), (1100, 320)]),  # 重置/页面切换复用到主入口附近

    ("e13", "left", [(1650, 1875), (1650, 630), (1610, 630)]),  # 切换到预训练模式
    ("e13", "right", [(2270, 1875), (2270, 405), (1190, 405)]),  # 切换页面
    ("e13", "top", [(1980, 590), (1100, 590), (1100, 320)]),  # 重置/页面切换复用到主入口附近
]


ARROW_LABELS = [
    ((930, 392), "部署"),
    ((1325, 392), "代理模型"),
    ((60, 950), "否"),
    ((1120, 1240), "否"),
    ((1670, 1440), "否"),
    ((150, 1030), "重置"),
    ((790, 980), "切换页面"),
    ((1760, 980), "切换到评测模式"),
    ((1600, 1230), "切换到预训练模式"),
    ((2210, 980), "切换页面"),
]


def wrap_lines(text: str, width: int) -> list[str]:
    max_chars = max(5, width // 18)
    lines: list[str] = []
    current = ""
    for ch in text:
        current += ch
        if len(current) >= max_chars and ch not in "，,、":
            lines.append(current)
            current = ""
    if current:
        lines.append(current)
    return lines


def shape_path(node: Node) -> str:
    x, y, w, h = node.x, node.y, node.w, node.h
    if node.kind == "decision":
        pts = [(x + w // 2, y), (x + w, y + h // 2), (x + w // 2, y + h), (x, y + h // 2)]
        return f"M {pts[0][0]} {pts[0][1]} L {pts[1][0]} {pts[1][1]} L {pts[2][0]} {pts[2][1]} L {pts[3][0]} {pts[3][1]} Z"
    if node.kind == "terminator":
        r = h / 2
        return (
            f"M {x+r} {y} H {x+w-r} "
            f"A {r} {r} 0 0 1 {x+w-r} {y+h} H {x+r} "
            f"A {r} {r} 0 0 1 {x+r} {y} Z"
        )
    return f"M {x} {y} H {x+w} V {y+h} H {x} Z"


def build_svg() -> str:
    width = 2400
    height = 2050
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#4b5563"/>',
        "</marker>",
        '<style>',
        'text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#111827}',
        '.title{font-size:22px;font-weight:700}',
        '.node{fill:#ffffff;stroke:#334155;stroke-width:2}',
        '.line{fill:none;stroke:#4b5563;stroke-width:2.2;marker-end:url(#arrow)}',
        '.label{font-size:15px;fill:#374151}',
        '.nodeText{font-size:16px;font-weight:500;text-anchor:middle;dominant-baseline:middle}',
        '</style>',
        "</defs>",
        '<rect x="0" y="0" width="2400" height="2050" fill="#f8fafc"/>',
        '<text x="1200" y="28" class="title" text-anchor="middle">统一Web可视化系统操作流程图</text>',
    ]
    for node in NODES:
        parts.append(f'<path class="node" d="{shape_path(node)}"/>')
        lines = wrap_lines(node.text, node.w)
        line_y = node.cy - (len(lines) - 1) * 11
        for i, line in enumerate(lines):
            parts.append(
                f'<text class="nodeText" x="{node.cx}" y="{line_y + i*22}">{html.escape(line)}</text>'
            )
    for src_id, anchor, pts in CONNECTORS:
        src = NODE_MAP[src_id]
        start = getattr(src, anchor)
        all_pts = [start, *pts]
        parts.append(f'<polyline class="line" points="{p(*all_pts)}"/>')
    for (x, y), text in ARROW_LABELS:
        parts.append(f'<text class="label" x="{x}" y="{y}">{html.escape(text)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def build_vdx() -> str:
    root = Element(
        "VisioDocument",
        {
            "xmlns": "urn:schemas-microsoft-com:office:visio",
            "xml:space": "preserve",
        },
    )
    pages = SubElement(root, "Pages")
    page = SubElement(pages, "Page", {"ID": "0", "NameU": "Page-1", "Name": "Page-1"})
    page_sheet = SubElement(page, "PageSheet")
    page_props = SubElement(page_sheet, "PageProps")
    SubElement(page_props, "PageWidth").text = "33.3333"
    SubElement(page_props, "PageHeight").text = "28.4722"

    shapes = SubElement(page, "Shapes")
    sid = 1
    scale = 72.0

    def add_shape(node: Node) -> None:
        nonlocal sid
        shape = SubElement(shapes, "Shape", {"ID": str(sid), "NameU": f"Node.{sid}", "Type": "Shape"})
        sid += 1
        x = node.x / scale
        y = (2050 - node.y - node.h) / scale
        w = node.w / scale
        h = node.h / scale
        xform = SubElement(shape, "XForm")
        SubElement(xform, "PinX").text = f"{x + w/2:.4f}"
        SubElement(xform, "PinY").text = f"{y + h/2:.4f}"
        SubElement(xform, "Width").text = f"{w:.4f}"
        SubElement(xform, "Height").text = f"{h:.4f}"
        SubElement(xform, "LocPinX").text = f"{w/2:.4f}"
        SubElement(xform, "LocPinY").text = f"{h/2:.4f}"
        line = SubElement(shape, "Line")
        SubElement(line, "LineWeight").text = "0.018"
        fill = SubElement(shape, "Fill")
        SubElement(fill, "FillForegnd").text = "#ffffff"
        char = SubElement(shape, "Char", {"IX": "0"})
        SubElement(char, "Size").text = "0.1667"
        para = SubElement(shape, "Para", {"IX": "0"})
        SubElement(para, "HorzAlign").text = "1"
        text = SubElement(shape, "Text")
        text.text = node.text
        geom = SubElement(shape, "Geom", {"IX": "0"})
        if node.kind == "decision":
            pts = [(w / 2, h), (w, h / 2), (w / 2, 0), (0, h / 2), (w / 2, h)]
        else:
            pts = [(0, h), (w, h), (w, 0), (0, 0), (0, h)]
        for i, (px, py) in enumerate(pts):
            tag = "MoveTo" if i == 0 else "LineTo"
            row = SubElement(geom, tag)
            SubElement(row, "X").text = f"{px:.4f}"
            SubElement(row, "Y").text = f"{py:.4f}"

    def add_line(points: list[tuple[int, int]]) -> None:
        nonlocal sid
        shape = SubElement(shapes, "Shape", {"ID": str(sid), "NameU": f"Line.{sid}", "Type": "Shape"})
        sid += 1
        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        x = min_x / scale
        y = (2050 - max_y) / scale
        w = max((max_x - min_x) / scale, 0.01)
        h = max((max_y - min_y) / scale, 0.01)
        xform = SubElement(shape, "XForm")
        SubElement(xform, "PinX").text = f"{x + w/2:.4f}"
        SubElement(xform, "PinY").text = f"{y + h/2:.4f}"
        SubElement(xform, "Width").text = f"{w:.4f}"
        SubElement(xform, "Height").text = f"{h:.4f}"
        SubElement(xform, "LocPinX").text = f"{w/2:.4f}"
        SubElement(xform, "LocPinY").text = f"{h/2:.4f}"
        line = SubElement(shape, "Line")
        SubElement(line, "LineWeight").text = "0.014"
        SubElement(line, "EndArrow").text = "4"
        geom = SubElement(shape, "Geom", {"IX": "0"})
        for i, (px, py) in enumerate(points):
            lx = (px - min_x) / scale
            ly = (max_y - py) / scale
            tag = "MoveTo" if i == 0 else "LineTo"
            row = SubElement(geom, tag)
            SubElement(row, "X").text = f"{lx:.4f}"
            SubElement(row, "Y").text = f"{ly:.4f}"

    for node in NODES:
        add_shape(node)
    for src_id, anchor, pts in CONNECTORS:
        src = NODE_MAP[src_id]
        add_line([getattr(src, anchor), *pts])

    return tostring(root, encoding="unicode")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(build_svg(), encoding="utf-8")
    VDX_PATH.write_text(build_vdx(), encoding="utf-8")
    print(SVG_PATH)
    print(VDX_PATH)


if __name__ == "__main__":
    main()
