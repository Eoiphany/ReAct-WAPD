"""
用途:
  TransUNet 模型定义文件。该实现遵循官方 2D TransUNet 的“CNN 编码器 + Transformer 编码器 +
  U-Net 解码器”主结构，并适配当前 2 通道输入、1 通道回归输出的代理建模任务。

直接运行命令:
  无。该文件是模型定义模块，供其他脚本导入。

导出对象与参数:
  TransUNet(
    in_channels=2,
    out_channels=1,
    encoder_channels=(64, 128, 256, 512),
    hidden_size=512,
    num_heads=8,
    num_layers=4,
    mlp_dim=2048
  )
    in_channels: 输入通道数。
    out_channels: 输出通道数。
    encoder_channels: CNN 编码器各 stage 通道数。
    hidden_size: Transformer token 维度。
    num_heads: 多头注意力头数。
    num_layers: Transformer encoder 层数。
    mlp_dim: FFN 隐层维度。
  build_transunet(in_channels=2)
    in_channels: 便捷构造函数输入通道数。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class EncoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            ConvBNReLU(in_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBNReLU(out_channels + skip_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


class PatchTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        mlp_dim: int,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=1)
        self.position_embeddings: nn.Parameter | None = None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

    def _set_position_embeddings(self, position_embeddings: torch.Tensor) -> nn.Parameter:
        reference_weight = self.proj.weight
        position_embeddings = position_embeddings.to(
            device=reference_weight.device,
            dtype=reference_weight.dtype,
        )
        self.position_embeddings = nn.Parameter(position_embeddings)
        return self.position_embeddings

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        key = f"{prefix}position_embeddings"
        if key not in state_dict:
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            return

        checkpoint_embeddings = state_dict[key]
        if self.position_embeddings is None or self.position_embeddings.shape != checkpoint_embeddings.shape:
            self._set_position_embeddings(checkpoint_embeddings.detach().clone())

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _get_position_embeddings(
        self,
        num_tokens: int,
        hidden_size: int,
        device: torch.device,
    ) -> nn.Parameter:
        if self.position_embeddings is None or self.position_embeddings.shape[1:] != (num_tokens, hidden_size):
            position_embeddings = torch.zeros(1, num_tokens, hidden_size, device=device)
            nn.init.trunc_normal_(position_embeddings, std=0.02)
            return self._set_position_embeddings(position_embeddings)
        return self.position_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        batch_size, hidden_size, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        position_embeddings = self._get_position_embeddings(tokens.shape[1], hidden_size, x.device)
        tokens = self.encoder(tokens + position_embeddings)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2).reshape(batch_size, hidden_size, height, width)


class TransUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        encoder_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        hidden_size: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_dim: int = 2048,
    ):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels

        self.stem = EncoderBlock(in_channels, c1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), EncoderBlock(c1, c2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), EncoderBlock(c2, c3))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), EncoderBlock(c3, c4))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), EncoderBlock(c4, c4))

        self.transformer = PatchTransformerEncoder(
            in_channels=c4,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_dim=mlp_dim,
        )
        self.conv_more = ConvBNReLU(hidden_size, c4)

        self.decoder1 = DecoderBlock(c4, c4, 256)
        self.decoder2 = DecoderBlock(256, c3, 128)
        self.decoder3 = DecoderBlock(128, c2, 64)
        self.decoder4 = DecoderBlock(64, c1, 16)
        self.segmentation_head = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.stem(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        bottleneck = self.down4(skip4)

        tokens = self.transformer(bottleneck)
        x = self.conv_more(tokens)
        x = self.decoder1(x, skip4)
        x = self.decoder2(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder4(x, skip1)
        x = self.segmentation_head(x)
        if x.shape[2:] != skip1.shape[2:]:
            x = F.interpolate(x, size=skip1.shape[2:], mode="bilinear", align_corners=False)
        return x


def build_transunet(in_channels: int = 2) -> TransUNet:
    return TransUNet(
        in_channels=in_channels,
        out_channels=1,
        encoder_channels=(64, 128, 256, 512),
        hidden_size=512,
        num_heads=8,
        num_layers=4,
        mlp_dim=2048,
    )
