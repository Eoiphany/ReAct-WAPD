"""
用途:
  PMNet 的消融实验模型定义，支持单独开启 SE、gated fusion、residual refine。

直接运行命令:
  无。该文件是模型定义模块，不单独运行。

导出对象与参数:
  PMNetAblation(n_blocks, atrous_rates, multi_grids, output_stride, use_se=False, use_gated_fusion=False, use_residual_refine=False)
    n_blocks: 编码器每个 stage 的 block 数。
    atrous_rates: ASPP 的空洞卷积 rate。
    multi_grids: 最后一层编码器的 multi-grid。
    output_stride: 输出步长，8 或 16。
    use_se: 是否启用 SE 模块。
    use_gated_fusion: 是否启用 gated skip fusion。
    use_residual_refine: 是否启用 residual refine。
  build_pmnet_ablation(...)
    上述参数的便捷构造函数。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pmnet import PMNet, _ConvBnReLU


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class ResidualRefine(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            _ConvBnReLU(channels, channels, 3, 1, 1, 1, True),
            _ConvBnReLU(channels, channels, 3, 1, 1, 1, False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x), inplace=True)


class SkipGate(nn.Module):
    def __init__(self, dec_ch: int, skip_ch: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dec_ch + skip_ch, skip_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_ch),
            nn.Sigmoid(),
        )

    def forward(self, dec: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([dec, skip], dim=1))
        return skip * gate


class PMNetAblation(PMNet):
    def __init__(
        self,
        n_blocks,
        atrous_rates,
        multi_grids,
        output_stride,
        use_se: bool = False,
        use_gated_fusion: bool = False,
        use_residual_refine: bool = False,
    ):
        super().__init__(n_blocks, atrous_rates, multi_grids, output_stride)
        self.use_se = use_se
        self.use_gated_fusion = use_gated_fusion
        self.use_residual_refine = use_residual_refine

        self.se_aspp = SEBlock(512)
        self.se_up4 = SEBlock(512)
        self.se_up3 = SEBlock(256)
        self.se_up2 = SEBlock(256)
        self.se_up1 = SEBlock(256)
        self.se_up0 = SEBlock(128)

        self.refine_up4 = ResidualRefine(512)
        self.refine_up3 = ResidualRefine(256)
        self.refine_up2 = ResidualRefine(256)
        self.refine_up1 = ResidualRefine(256)
        self.refine_up0 = ResidualRefine(128)

        self.gate5 = SkipGate(512, 512)
        self.gate4 = SkipGate(512, 512)
        self.gate3 = SkipGate(256, 256)
        self.gate2 = SkipGate(256, 256)
        self.gate1 = SkipGate(256, 64)

    def _maybe_gate(self, dec: torch.Tensor, skip: torch.Tensor, gate: SkipGate) -> torch.Tensor:
        if not self.use_gated_fusion:
            return skip
        return gate(dec, skip)

    def _maybe_se(self, x: torch.Tensor, block: SEBlock) -> torch.Tensor:
        if not self.use_se:
            return x
        return block(x)

    def _maybe_refine(self, x: torch.Tensor, block: ResidualRefine) -> torch.Tensor:
        if not self.use_residual_refine:
            return x
        return block(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.reduce(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.aspp(x6)
        x8 = self.fc1(x7)
        x8 = self._maybe_se(x8, self.se_aspp)

        xup5 = self.conv_up5(x8)
        xup5 = torch.cat([xup5, self._maybe_gate(xup5, x5, self.gate5)], dim=1)

        xup4 = self.conv_up4(xup5)
        xup4 = self._maybe_se(xup4, self.se_up4)
        xup4 = self._maybe_refine(xup4, self.refine_up4)
        xup4 = torch.cat([xup4, self._maybe_gate(xup4, x4, self.gate4)], dim=1)

        xup3 = self.conv_up3(xup4)
        xup3 = self._maybe_se(xup3, self.se_up3)
        xup3 = self._maybe_refine(xup3, self.refine_up3)
        xup3 = torch.cat([xup3, self._maybe_gate(xup3, x3, self.gate3)], dim=1)

        xup2 = self.conv_up2(xup3)
        xup2 = self._maybe_se(xup2, self.se_up2)
        xup2 = self._maybe_refine(xup2, self.refine_up2)
        xup2 = torch.cat([xup2, self._maybe_gate(xup2, x2, self.gate2)], dim=1)

        xup1 = self.conv_up1(xup2)
        xup1 = self._maybe_se(xup1, self.se_up1)
        xup1 = self._maybe_refine(xup1, self.refine_up1)
        xup1 = torch.cat([xup1, self._maybe_gate(xup1, x1, self.gate1)], dim=1)

        xup0 = self.conv_up0(xup1)
        xup0 = self._maybe_se(xup0, self.se_up0)
        xup0 = self._maybe_refine(xup0, self.refine_up0)

        xup0 = F.interpolate(xup0, size=x.shape[2:], mode="bilinear", align_corners=False)
        xup0 = torch.cat([xup0, x], dim=1)
        return self.conv_up00(xup0)


def build_pmnet_ablation(
    output_stride: int = 16,
    use_se: bool = False,
    use_gated_fusion: bool = False,
    use_residual_refine: bool = False,
) -> PMNetAblation:
    return PMNetAblation(
        n_blocks=[3, 3, 27, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=output_stride,
        use_se=use_se,
        use_gated_fusion=use_gated_fusion,
        use_residual_refine=use_residual_refine,
    )
