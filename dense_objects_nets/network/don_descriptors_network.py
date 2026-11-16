# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import timm
import torch.nn as nn

from typing import Any, Dict
from collections import OrderedDict
from dense_objects_nets.network.don_backbone import DonNetwork


class DonNetworkResNet34(DonNetwork):
    """
    DON descriptors network using ResNet34 as backbone.
    """

    def create_backbone(self, network_config: Dict[str, Any]) -> nn.Module:
        resnet34_original = timm.create_model("resnet34", pretrained=True, output_stride=8)

        # Remove the last two layers (avg pooling and fully connected)
        layers = list(resnet34_original.named_children())
        layers = layers[:-2]

        backbone = nn.Sequential(OrderedDict([*layers]))

        return backbone,  resnet34_original.fc.in_features # 512


class DonNetworkResNet50(DonNetwork):
    """
    DON descriptors network using ResNet50 as backbone.
    """

    def create_backbone(self, network_config: Dict[str, Any]) -> nn.Module:
        resnet50_original = timm.create_model("resnet50", pretrained=True, output_stride=8)

        # Remove the last two layers (avg pooling and fully connected)
        layers = list(resnet50_original.named_children())
        layers = layers[:-2]

        backbone = nn.Sequential(OrderedDict([*layers]))

        return backbone,  resnet50_original.fc.in_features # 2048


class DonNetworkConvNext(DonNetwork):
    """
    DON descriptors network using ConvNext as backbone.
    """

    def create_backbone(self, network_config: Dict[str, Any]) -> nn.Module:
        convnext_base = timm.create_model("convnext_base.fb_in22k_ft_in1k", pretrained=True, output_stride=8)

        # Remove the last two layers (avg pooling and fully connected)
        layers = list(convnext_base.named_children())
        layers = layers[:-2]

        backbone = nn.Sequential(OrderedDict([*layers]))

        return backbone,  convnext_base.num_features # 1024
        