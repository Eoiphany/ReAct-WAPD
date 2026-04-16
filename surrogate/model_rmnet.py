"""
用途:
  RMNet 模型定义文件。这里的 RMNet 就是你原始代码中的 PMNetPlus。

直接运行命令:
  无。该文件是模型定义模块，供其他脚本导入。

导出对象与参数:
  RMNet(n_blocks, atrous_rates, multi_grids, output_stride, in_ch=2)
    n_blocks: ResNet 主干各 stage 的 block 数。
    atrous_rates: ASPP 的空洞卷积 rate 列表。
    multi_grids: 最后一层 encoder 的 multi-grid 设置。
    output_stride: 输出步长，只支持 8 或 16。
    in_ch: 输入通道数，当前默认 2。
  build_rmnet(output_stride=16)
    output_stride: 便捷构造函数参数。
"""

from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(channels: int, max_groups: int = 16) -> nn.Module:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        relu: bool = True,
    ):
        layers = [
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            _make_norm(out_ch),
        ]
        if relu:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch: int, out_ch: int, stride: int, dilation: int, downsample: bool):
        super().__init__()
        mid_ch = out_ch // self.expansion
        self.reduce = ConvNormAct(in_ch, mid_ch, 1, stride=stride)
        self.conv3x3 = ConvNormAct(
            mid_ch,
            mid_ch,
            3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        )
        self.increase = ConvNormAct(mid_ch, out_ch, 1, relu=False)
        self.shortcut = ConvNormAct(in_ch, out_ch, 1, stride=stride, relu=False) if downsample else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.reduce(x)
        out = self.conv3x3(out)
        out = self.increase(out)
        out = out + self.shortcut(x)
        return self.act(out)


class ResLayer(nn.Sequential):
    def __init__(self, n_layers: int, in_ch: int, out_ch: int, stride: int, dilation: int, multi_grids=None):
        super().__init__()
        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert len(multi_grids) == n_layers

        for i in range(n_layers):
            self.add_module(
                f"block{i + 1}",
                Bottleneck(
                    in_ch=in_ch if i == 0 else out_ch,
                    out_ch=out_ch,
                    stride=stride if i == 0 else 1,
                    dilation=dilation * multi_grids[i],
                    downsample=(i == 0),
                ),
            )


class Stem(nn.Sequential):
    def __init__(self, out_ch: int, in_ch: int = 2):
        super().__init__()
        self.add_module("conv1", ConvNormAct(in_ch, out_ch, 7, stride=2, padding=3))
        self.add_module("pool", nn.MaxPool2d(2, 2, 0, ceil_mode=True))


class ImagePool(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = ConvNormAct(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        out = self.pool(x)
        out = self.conv(out)
        return F.interpolate(out, size=(height, width), mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates):
        super().__init__()
        self.stages = nn.ModuleList([ConvNormAct(in_ch, out_ch, 1)])
        for rate in rates:
            self.stages.append(ConvNormAct(in_ch, out_ch, 3, padding=rate, dilation=rate))
        self.stages.append(ImagePool(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([stage(x) for stage in self.stages], dim=1)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = F.silu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class ResidualRefine(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, 3, padding=1),
            ConvNormAct(channels, channels, 3, padding=1, relu=False),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class FusionBlock(nn.Module):
    def __init__(self, dec_in: int, skip_in: int, out_ch: int):
        super().__init__()
        self.dec_proj = ConvNormAct(dec_in, out_ch, 1)
        self.skip_proj = ConvNormAct(skip_in, out_ch, 1)
        self.gate = nn.Sequential(
            ConvNormAct(out_ch * 2, out_ch, 1),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            ConvNormAct(out_ch * 2, out_ch, 3, padding=1),
            ResidualRefine(out_ch),
            SEBlock(out_ch),
        )

    def forward(self, dec: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        dec = F.interpolate(dec, size=skip.shape[2:], mode="bilinear", align_corners=False)
        dec_feat = self.dec_proj(dec)
        skip_feat = self.skip_proj(skip)
        gate = self.gate(torch.cat([dec_feat, skip_feat], dim=1))
        fused_skip = skip_feat * gate
        return self.refine(torch.cat([dec_feat, fused_skip], dim=1))


class RMNet(nn.Module):
    def __init__(self, n_blocks, atrous_rates, multi_grids, output_stride, in_ch: int = 2):
        super().__init__()

        if output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        elif output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        else:
            raise ValueError(f"Unsupported output_stride: {output_stride}")

        ch = [64 * 2**p for p in range(6)]

        self.layer1 = Stem(ch[0], in_ch=in_ch)
        self.layer2 = ResLayer(n_blocks[0], ch[0], ch[2], strides[0], dilations[0])
        self.reduce = ConvNormAct(256, 256, 1)
        self.layer3 = ResLayer(n_blocks[1], ch[2], ch[3], strides[1], dilations[1])
        self.layer4 = ResLayer(n_blocks[2], ch[3], ch[3], strides[2], dilations[2])
        self.layer5 = ResLayer(n_blocks[3], ch[3], ch[4], strides[3], dilations[3], multi_grids=multi_grids)

        self.aspp = ASPP(ch[4], 256, atrous_rates)
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.context = nn.Sequential(
            ConvNormAct(concat_ch, 512, 1),
            SEBlock(512),
            nn.Dropout2d(p=0.1),
        )

        self.fuse5 = FusionBlock(512, 1024, 512)
        self.fuse4 = FusionBlock(512, 512, 256)
        self.fuse3 = FusionBlock(256, 512, 256)
        self.fuse2 = FusionBlock(256, 256, 192)
        self.fuse1 = FusionBlock(192, 64, 128)

        self.input_fuse = nn.Sequential(
            ConvNormAct(128 + in_ch, 128, 3, padding=1),
            ResidualRefine(128),
            ConvNormAct(128, 64, 3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(x)
        x2_raw = self.layer2(x1)
        x2 = self.reduce(x2_raw)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        context = self.context(self.aspp(x5))
        d5 = self.fuse5(context, x5)
        d4 = self.fuse4(d5, x4)
        d3 = self.fuse3(d4, x3)
        d2 = self.fuse2(d3, x2)
        d1 = self.fuse1(d2, x1)

        out = F.interpolate(d1, size=x.shape[2:], mode="bilinear", align_corners=False)
        out = torch.cat([out, x], dim=1)
        return self.input_fuse(out)


PMNetPlus = RMNet


def build_rmnet(output_stride: int = 16) -> RMNet:
    return RMNet(
        n_blocks=[3, 3, 27, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=output_stride,
    )
