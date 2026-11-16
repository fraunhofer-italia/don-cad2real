# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>


import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from torchvision.transforms import v2
from typing import Dict, Any, List, Tuple

from dense_objects_nets.network.network import Network

class DonNetwork(Network, ABC):

    def __init__(self, network_config: Dict[str, Any]) -> "DonNetwork":
        """
        Base class used to create the Dense Object Nets underlying convolutional network.

        Adds a final convolutional layer to a given backbone network to produce dense descriptors.
        
        Args:
        - network_config (`dict`): dict containing the required network parameters.
        - model_weights (`str`): path to the model state dict object needed to setup a pretrained model. Defaults to 'None'. 
        - model_device (`torch.device`): device for the weights of the input pretrained model. Defaults to 'None'.
        """
        super(DonNetwork, self).__init__(network_config)

        # Create the backbone of the network
        self.backbone, bk_output_channels = self.create_backbone(network_config)

        # Create the final layer of the network
        self.don_head = nn.Conv2d(
            in_channels=bk_output_channels,
            out_channels=int(network_config["descriptors_dim"]),
            stride=1,
            kernel_size=1,
        )
        self.don_head.weight.data.normal_(0, 0.01)
        self.don_head.bias.data.zero_()

    @abstractmethod
    def create_backbone(self, network_config: Dict[str, Any]) -> nn.Module:
        pass

    def forward(self, x: torch.Tensor, upsample: bool = True):
        """
        Forward pass on the network. Output is the rescaled with bilinear interpolation to fit the desired
        output image (OUT_H, out_W) dimensions. 

        Args:
        - x (`torch.Image`): tensor containing the input image with shape (bn, 3, H, W). 

        Returns:
        - `torch.Tensor`: tensor containing the network output rescaled to the desire output size: (bn, DESC_DIM, OUT_H, OUT_W).
        """
        # Save the input image shape
        H, W = x.shape[-2:]

        # Normalize to ImageNet values
        x = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)

        # Forward pass on the backbone
        x = self.backbone(x)

        # Forward pass on the final layer
        x = self.don_head(x)

        if upsample:
            x = nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)

        return x

    def load_state_dict(self, state_dict, strict = True, assign = False):
        # For compatibility with old models, replace some state dict keys if found. # TODO: remove in the future
        def replace_key(key: str) -> str:
            if not key.startswith("model."):
                return key
            elif key.startswith("model.out."):
                return key.replace("model.out.", "don_head.")
            else:
                return key.replace("model.", "backbone.")
            
        state_dict = {
            replace_key(k): v for k, v in state_dict.items()
        }
        return super().load_state_dict(state_dict, strict, assign)

    @classmethod
    def from_checkpoint(cls, filepath: str, device: torch.device = torch.device("cpu")):
        # Overload for compatibility with old models, check if the checkpoint contains the network configuration. # TODO: remove in the future
        checkpoint = torch.load(filepath, map_location=device)

        if "network_config" not in checkpoint:
            new_checkpoint = {}
            new_checkpoint["state_dict"] = checkpoint
            new_checkpoint["network_config"] = {
                "descriptors_dim": checkpoint["don_head.weight"].shape[0],
            }
            checkpoint = new_checkpoint

        model = cls(network_config=checkpoint["network_config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        return model

    @property
    def D(self):
        return int(self.network_config_["descriptors_dim"])