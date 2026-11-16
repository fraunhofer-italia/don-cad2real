# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum
from timm.layers import StdConv2d
from collections import OrderedDict


# ----------- Common block components ------------

class BlockEnum(Enum):
    # Base Enum for nn modules
    def __eq__(self, value):
        if isinstance(value, str):
            return self.name == value.upper()
        
        return super().__eq__(value)

class Normalization(BlockEnum):
    BATCH_NORM = lambda channels: nn.BatchNorm2d(channels)
    GROUP_NORM = lambda channels, num_groups=32: nn.GroupNorm(num_groups, channels)
    NONE = lambda: nn.Identity()


class Activation(BlockEnum):
    GELU = lambda: nn.GELU()
    RELU = lambda inplace=True: nn.ReLU(inplace=inplace)
    SILU = lambda inplace=True: nn.SiLU(inplace=inplace)
    SIGMOID = lambda: nn.Sigmoid()
    NONE = lambda: nn.Identity()


class UpsampleBlock(BlockEnum):
    BILINEAR = lambda scale_factor=2: nn.UpsamplingBilinear2d(scale_factor=scale_factor)
    DECONV = lambda in_channels, out_channels: nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)



# ----------- Utility Modules ------------

class ConvModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: Normalization, activation: Activation, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1, bias: bool = False):
        super(ConvModule, self).__init__()

        self.conv_ = StdConv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )
        self.conv_.weight.data.normal_(0, 0.01)
        if bias: self.conv_.bias.data.zero_()

        self.norm_ = norm
        self.activate_ = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_(x)
        x = self.norm_(x)
        x = self.activate_(x)
        return x
    
class UpsampleModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_conv_per_block: int = 2, upsample_block: UpsampleBlock = UpsampleBlock.BILINEAR):
        super(UpsampleModule, self).__init__()

        if upsample_block == UpsampleBlock.BILINEAR:
            upsample_layer = [("bilinear_upsample", upsample_block(scale_factor=2))]
            conv_in_channels = in_channels
        elif upsample_block == UpsampleBlock.DECONV:
            upsample_layer = [("deconv_upsample", upsample_block(in_channels, out_channels))]
            conv_in_channels = out_channels
        else:
            raise ValueError(f"Unknown upsample method: {upsample_block}")

        conv_layers = [
            (
                f"conv_module_{i+1}",
                ConvModule(
                    in_channels=conv_in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    norm=Normalization.BATCH_NORM(channels=out_channels),
                    activation=Activation.GELU(),
                ),
            )
            for i in range(num_conv_per_block)
        ]

        self.upsample_model = nn.Sequential(OrderedDict([*(upsample_layer + conv_layers)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample_model(x)


class DFLIntegral(nn.Module):
    def __init__(self, bins: int):
        super().__init__()
        self.bins = bins
        # [1,1,B+1] buffer to project per-side distributions to expectation
        proj = torch.arange(bins + 1, dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("proj", proj, persistent=False)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, 4*(bins+1), H, W]
        returns: [B, 4, H, W] expected distances in "bin units"
        """
        B, _, H, W = logits.shape
        x = logits.view(B, 4, self.bins + 1, H, W)
        probs = F.softmax(x, dim=2)
        exp = (probs * self.proj[..., None, None]).sum(dim=2)
        return exp

