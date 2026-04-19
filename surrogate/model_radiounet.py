"""
用途:
  RadioUNet 模型定义文件。该实现依据论文《RadioUNet: Fast Radio Map Estimation with
  Convolutional Neural Networks》作者公开代码中的 RadioWNet 双 U 结构整理，并统一为当前项目的
  单输出接口。

直接运行命令:
  无。该文件是模型定义模块，供其他脚本导入。

导出对象与参数:
  RadioUNet(in_channels=2, return_intermediate=False)
    in_channels: 输入通道数。
    return_intermediate: 是否同时返回第一阶段与第二阶段输出。
  build_radiounet(in_channels=2)
    in_channels: 便捷构造函数输入通道数。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def convrelu(in_channels: int, out_channels: int, kernel: int, padding: int, pool: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool, stride=pool, padding=0),
    )


def convreluT(in_channels: int, out_channels: int, kernel: int, padding: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride=2, padding=padding),
        nn.ReLU(inplace=True),
    )


class RadioUNet(nn.Module):
    def __init__(self, in_channels: int = 2, return_intermediate: bool = False):
        super().__init__()
        self.inputs = in_channels
        self.return_intermediate = return_intermediate

        stem_channels = 6 if in_channels <= 3 else 10
        self.layer00 = convrelu(in_channels, stem_channels, 3, 1, 1)
        self.layer0 = convrelu(stem_channels, 40, 5, 2, 2)
        self.layer1 = convrelu(40, 50, 5, 2, 2)
        self.layer10 = convrelu(50, 60, 5, 2, 1)
        self.layer2 = convrelu(60, 100, 5, 2, 2)
        self.layer20 = convrelu(100, 100, 3, 1, 1)
        self.layer3 = convrelu(100, 150, 5, 2, 2)
        self.layer4 = convrelu(150, 300, 5, 2, 2)
        self.layer5 = convrelu(300, 500, 5, 2, 2)

        self.conv_up5 = convreluT(500, 300, 4, 1)
        self.conv_up4 = convreluT(600, 150, 4, 1)
        self.conv_up3 = convreluT(300, 100, 4, 1)
        self.conv_up20 = convrelu(200, 100, 3, 1, 1)
        self.conv_up2 = convreluT(200, 60, 6, 2)
        self.conv_up10 = convrelu(120, 50, 5, 2, 1)
        self.conv_up1 = convreluT(100, 40, 6, 2)
        self.conv_up0 = convreluT(80, 20, 6, 2)
        self.conv_up00 = convrelu(20 + stem_channels + in_channels, 20, 5, 2, 1)
        self.conv_up000 = convrelu(20 + in_channels, 1, 5, 2, 1)

        self.Wlayer00 = convrelu(in_channels + 1, 20, 3, 1, 1)
        self.Wlayer0 = convrelu(20, 30, 5, 2, 2)
        self.Wlayer1 = convrelu(30, 40, 5, 2, 2)
        self.Wlayer10 = convrelu(40, 50, 5, 2, 1)
        self.Wlayer2 = convrelu(50, 60, 5, 2, 2)
        self.Wlayer20 = convrelu(60, 70, 3, 1, 1)
        self.Wlayer3 = convrelu(70, 90, 5, 2, 2)
        self.Wlayer4 = convrelu(90, 110, 5, 2, 2)
        self.Wlayer5 = convrelu(110, 150, 5, 2, 2)

        self.Wconv_up5 = convreluT(150, 110, 4, 1)
        self.Wconv_up4 = convreluT(220, 90, 4, 1)
        self.Wconv_up3 = convreluT(180, 70, 4, 1)
        self.Wconv_up20 = convrelu(140, 60, 3, 1, 1)
        self.Wconv_up2 = convreluT(120, 50, 6, 2)
        self.Wconv_up10 = convrelu(100, 40, 5, 2, 1)
        self.Wconv_up1 = convreluT(80, 30, 6, 2)
        self.Wconv_up0 = convreluT(60, 20, 6, 2)
        self.Wconv_up00 = convrelu(20 + 20 + in_channels + 1, 20, 5, 2, 1)
        self.Wconv_up000 = convrelu(20 + in_channels + 1, 1, 5, 2, 1)

    def _match(self, x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] != reference.shape[2:]:
            x = F.interpolate(x, size=reference.shape[2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, input_tensor: torch.Tensor):
        input0 = input_tensor[:, 0 : self.inputs, :, :]

        layer00 = self.layer00(input0)
        layer0 = self.layer0(layer00)
        layer1 = self.layer1(layer0)
        layer10 = self.layer10(layer1)
        layer2 = self.layer2(layer10)
        layer20 = self.layer20(layer2)
        layer3 = self.layer3(layer20)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)

        layer4u = self._match(self.conv_up5(layer5), layer4)
        layer4u = torch.cat([layer4u, layer4], dim=1)
        layer3u = self._match(self.conv_up4(layer4u), layer3)
        layer3u = torch.cat([layer3u, layer3], dim=1)
        layer20u = self._match(self.conv_up3(layer3u), layer20)
        layer20u = torch.cat([layer20u, layer20], dim=1)
        layer2u = self.conv_up20(layer20u)
        layer2u = torch.cat([self._match(layer2u, layer2), layer2], dim=1)
        layer10u = self._match(self.conv_up2(layer2u), layer10)
        layer10u = torch.cat([layer10u, layer10], dim=1)
        layer1u = self.conv_up10(layer10u)
        layer1u = torch.cat([self._match(layer1u, layer1), layer1], dim=1)
        layer0u = self._match(self.conv_up1(layer1u), layer0)
        layer0u = torch.cat([layer0u, layer0], dim=1)
        layer00u = self._match(self.conv_up0(layer0u), layer00)
        layer00u = torch.cat([layer00u, layer00, self._match(input0, layer00)], dim=1)
        layer000u = self.conv_up00(layer00u)
        layer000u = torch.cat([self._match(layer000u, input0), input0], dim=1)
        output1 = self.conv_up000(layer000u)

        Winput = torch.cat([self._match(output1, input_tensor), input_tensor], dim=1)
        Wlayer00 = self.Wlayer00(Winput)
        Wlayer0 = self.Wlayer0(Wlayer00)
        Wlayer1 = self.Wlayer1(Wlayer0)
        Wlayer10 = self.Wlayer10(Wlayer1)
        Wlayer2 = self.Wlayer2(Wlayer10)
        Wlayer20 = self.Wlayer20(Wlayer2)
        Wlayer3 = self.Wlayer3(Wlayer20)
        Wlayer4 = self.Wlayer4(Wlayer3)
        Wlayer5 = self.Wlayer5(Wlayer4)

        Wlayer4u = self._match(self.Wconv_up5(Wlayer5), Wlayer4)
        Wlayer4u = torch.cat([Wlayer4u, Wlayer4], dim=1)
        Wlayer3u = self._match(self.Wconv_up4(Wlayer4u), Wlayer3)
        Wlayer3u = torch.cat([Wlayer3u, Wlayer3], dim=1)
        Wlayer20u = self._match(self.Wconv_up3(Wlayer3u), Wlayer20)
        Wlayer20u = torch.cat([Wlayer20u, Wlayer20], dim=1)
        Wlayer2u = self.Wconv_up20(Wlayer20u)
        Wlayer2u = torch.cat([self._match(Wlayer2u, Wlayer2), Wlayer2], dim=1)
        Wlayer10u = self._match(self.Wconv_up2(Wlayer2u), Wlayer10)
        Wlayer10u = torch.cat([Wlayer10u, Wlayer10], dim=1)
        Wlayer1u = self.Wconv_up10(Wlayer10u)
        Wlayer1u = torch.cat([self._match(Wlayer1u, Wlayer1), Wlayer1], dim=1)
        Wlayer0u = self._match(self.Wconv_up1(Wlayer1u), Wlayer0)
        Wlayer0u = torch.cat([Wlayer0u, Wlayer0], dim=1)
        Wlayer00u = self._match(self.Wconv_up0(Wlayer0u), Wlayer00)
        Wlayer00u = torch.cat([Wlayer00u, Wlayer00, self._match(Winput, Wlayer00)], dim=1)
        Wlayer000u = self.Wconv_up00(Wlayer00u)
        Wlayer000u = torch.cat([self._match(Wlayer000u, Winput), Winput], dim=1)
        output2 = self.Wconv_up000(Wlayer000u)

        output1 = self._match(output1, input_tensor)
        output2 = self._match(output2, input_tensor)
        if self.return_intermediate:
            return output1, output2
        return output2


def build_radiounet(in_channels: int = 2) -> RadioUNet:
    return RadioUNet(in_channels=in_channels, return_intermediate=False)
