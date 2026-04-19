"""
用途:
  标准 U-Net 模型定义文件，用于与 PMNet / RMNet 做对比实验。

直接运行命令:
  无。该文件是模型定义模块，供其他脚本导入。

导出对象与参数:
  UNet(in_channels=2, out_channels=1, base_channels=64)
    in_channels: 输入通道数，当前任务默认 2。
    out_channels: 输出通道数，当前任务默认 1。
    base_channels: 第一层特征通道数。
  build_unet(in_channels=2)
    in_channels: 便捷构造函数输入通道数。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Down(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]

        self.inc = DoubleConv(in_channels, channels[0])
        self.down1 = Down(channels[0], channels[1])
        self.down2 = Down(channels[1], channels[2])
        self.down3 = Down(channels[2], channels[3])
        self.down4 = Down(channels[3], channels[4])

        self.up1 = Up(channels[4], channels[3], channels[3])
        self.up2 = Up(channels[3], channels[2], channels[2])
        self.up3 = Up(channels[2], channels[1], channels[1])
        self.up4 = Up(channels[1], channels[0], channels[0])
        self.outc = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode="bilinear", align_corners=False)
        return x


def build_unet(in_channels: int = 2) -> UNet:
    return UNet(in_channels=in_channels, out_channels=1, base_channels=64)
