# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import os
import timm
import torch
import torch.nn as nn

from collections import OrderedDict
from typing import Any, Dict, List, Tuple

from dense_objects_nets.utils import evaluation_utils
from dense_objects_nets.network.network import Network
from dense_objects_nets.network import utils as nn_utils
from dense_objects_nets.pipelines.pose_estimation.utils import dynamic_zoom_in as dzi


class Head(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Head, self).__init__()

        self.conv = nn_utils.ConvModule(
            in_channels, out_channels,
            norm=nn_utils.Normalization.NONE(),
            activation=nn_utils.Activation.NONE(),
            kernel_size=3,
            padding=1,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PnPHead(nn.Module):
    def __init__(self, in_channels: int, num_conv_blocks: int = 3, feat_dim: int = 128, final_spatial_size: Tuple[int, int] = (32, 32)):
        super(PnPHead, self).__init__()

        conv_layers = [
            (
                f"conv_block_{i+1}",
                nn_utils.ConvModule(
                    in_channels=in_channels if i == 0 else feat_dim,
                    out_channels=feat_dim,
                    norm=nn_utils.Normalization.GROUP_NORM(channels=feat_dim),
                    activation=nn_utils.Activation.RELU(),
                    kernel_size=3,
                    stride=2,
                ),
            )
            for i in range(num_conv_blocks)
        ]
        self.conv_layers = nn.Sequential(OrderedDict(conv_layers))

        final_h, final_w = final_spatial_size
        fc_in_dim = feat_dim * final_h * final_w

        # First fully connected layer
        self.fc1 = nn.Linear(fc_in_dim, 1024)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.zero_()

        # Second fully connected layer
        self.fc2 = nn.Linear(1024, 256)
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc2.bias.data.zero_()

        # Fully connected layers for rotation (as R6d)
        self.fc_r = nn.Linear(256, 6)
        self.fc_r.weight.data.normal_(0, 0.01)
        self.fc_r.bias.data.zero_()
        
        # Fully connected layers for tsite (translation) [delta_x, delta_y, delta_z]
        self.fc_t = nn.Linear(256, 3)
        self.fc_t.weight.data.normal_(0, 0.01)
        self.fc_t.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_layers(x)

        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, -1)  # Flatten the tensor

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)

        r = self.fc_r(x)  # Rotation output
        t = self.fc_t(x)  # Translation output

        return r, t


class DonPoseNetwork(Network):
    def __init__(self, network_config: Dict[str, Any]):
        super(DonPoseNetwork, self).__init__(network_config)

        num_classes = int(network_config["num_classes"])
        num_blocks = int(network_config["num_blocks"])
        num_conv_per_block = int(network_config["num_conv_per_block"])
        patch_pnp_size = int(network_config["patch_pnp_size"])
        feat_dim = int(network_config["feat_dim"])
        
        # Load the backbone (a pretrained DON model if provided or a standard ResNet34)
        self.backbone, bk_output_channels = self.load_backbone(network_config)

        # Whether to freeze the backbone during training
        self.freeze_backbone = network_config["freeze_backbone"]
        self.backbone_ctx = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        if self.freeze_backbone:
            self.backbone.eval()

        layers = [
            (
                f"upsample_block_{i+1}",
                nn_utils.UpsampleModule(
                    in_channels=bk_output_channels if i == 0 else feat_dim,
                    out_channels=feat_dim,
                    num_conv_per_block=num_conv_per_block,
                    upsample_block=nn_utils.UpsampleBlock.DECONV if i == 0 else nn_utils.UpsampleBlock.BILINEAR,  # First block uses deconv, others use bilinear
                ),
            )
            for i in range(num_blocks)
        ]

        # Create the new model from the above layers (keeping the names)
        self.upsample_model = nn.Sequential(OrderedDict([*layers]))

        # Create the XYZ head
        self.xyz_head = Head(in_channels=feat_dim, out_channels=3 * num_classes)

        # Create the mask head
        self.mask_head = Head(in_channels=feat_dim, out_channels=1 * num_classes)

        # Create the PnP head
        pnp_in_channels = 3 + 1 + 2 # 3 for XYZ, 1 for mask, and 2 for the coordinates mapping
        self.pnp_head = PnPHead(in_channels=pnp_in_channels, final_spatial_size=(patch_pnp_size, patch_pnp_size))

        # Save the number of classes
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, bboxes: List[torch.Tensor], obj_channel_indexes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with self.backbone_ctx: # Gradient tracking for the backbone depends on whether it is frozen or not
            backbone_features = self.backbone(x)

        pose_prediction = self.upsample_model(backbone_features)

        xyz_img = self.xyz_head(pose_prediction)  # (batch_size, 3 * num_classes, H, W)
        mask_img = self.mask_head(pose_prediction) # (batch_size, 1 * num_classes, H, W)

        # Extract xyz and mask images for the objects in the input images
        batch_size, _, H, W = xyz_img.shape
        xyz_img = xyz_img.view(batch_size, self.num_classes, 3, H, W)
        mask_img = mask_img.view(batch_size, self.num_classes, 1, H, W)

        batch_indexes = torch.arange(batch_size, device=xyz_img.device)
        objects_xyz_img = xyz_img[batch_indexes, obj_channel_indexes] # (batch_size, 3, H, W)
        objects_mask_img = mask_img[batch_indexes, obj_channel_indexes] # (batch_size, H, W)

        # Get the 2D-to-3D coordinates mapping images required by the PnP head
        roi_coords_img = torch.stack([
            dzi.get_coordinates_from_bbox(bbox, out_size=(H, W)) for bbox in bboxes
        ])  # (batch_size, 2, H, W)

        # Concatenate the inputs required by the PnP head
        pnp_inputs = torch.cat([objects_xyz_img, objects_mask_img, roi_coords_img], dim=1) # (batch_size, 3 + 1 + 2, H, W)

        # Concatenate the inputs required by the PnP head
        rot, tsite = self.pnp_head(pnp_inputs) # (batch_size, 6), (batch_size, 3)

        return objects_xyz_img, objects_mask_img, rot, tsite

    def load_backbone(self, network_config: Dict[str, Any]) -> Tuple[Network, int]:
        backbone_class = network_config.get("backbone_class", None)

        if backbone_class is None:
            print("No backbone class provided, using standard ResNet34 as backbone.")
            # Load standard ResNet34 as backbone, with last two layers removed
            resnet34_original = timm.create_model("resnet34", pretrained=True, output_stride=32)        

            # Remove the last two layers (avg pooling and fully connected) and replace them with a new Con2D layer
            layers = list(resnet34_original.named_children())
            layers = layers[:-2]

            backbone = nn.Sequential(OrderedDict([*layers]))

            return backbone, resnet34_original.fc.in_features # 512        
        
        # Load a DON backbone model to extract the descriptors from
        backbone_class = evaluation_utils.load_class(f"dense_objects_nets.network.{backbone_class}")
        backbone_checkpoint = network_config.get("checkpoint", None)

        # If checkpoint is not provided or invalid, check for backbone config
        if backbone_checkpoint is None or not os.path.isfile(os.path.expandvars(backbone_checkpoint)):
            backbone_config = network_config.get("backbone_config", None)
            if backbone_config is None:
                raise ValueError("No valid checkpoint or backbone config provided.")
            
            # Create the backbone from the provided config
            backbone = backbone_class(backbone_config)

        else:
            backbone = backbone_class.from_checkpoint(
                os.path.expandvars(network_config["checkpoint"]),
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )

        # Save the backbone network configuration so that it can be reloaded even if the original backbone checkpoint is not available
        self.network_config_["backbone_config"] = backbone.network_config_

        # Modify the forward function of the DON backbone to avoid upsampling
        backbone.forward = lambda x, _orig_forward=backbone.forward: _orig_forward(x, upsample=False)

        return backbone, backbone.D

    def train(self, mode = True):
        if not self.freeze_backbone: self.backbone.train(mode)
        self.upsample_model.train(mode)
        self.xyz_head.train(mode)
        self.mask_head.train(mode)
        self.pnp_head.train(mode)
        self.training = mode

    def named_parameters(self, recurse=True):
        for name, param in super().named_parameters(recurse=recurse):
            # skip backbone if frozen
            if self.freeze_backbone and "backbone" in name:
                continue
            # keep all other parameters
            yield name, param

    def get_optimizer(self, optimizer_config: Dict[str, Any]) -> torch.optim.Optimizer:
        learning_rate = float(optimizer_config["learning_rate"])
        backbone_learning_rate = float(optimizer_config.get("backbone_learning_rate", learning_rate))
        weight_decay = float(optimizer_config.get("weight_decay", 0.0))
        
        decay_backbone, no_decay_backbone = [], []
        decay_head, no_decay_head = [], []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            
            is_backbone = "backbone" in name
            is_no_decay = (p.ndim == 1) or name.endswith(".bias")

            if is_backbone:
                if is_no_decay:
                    no_decay_backbone.append(p)
                else:
                    decay_backbone.append(p)
            else:
                if is_no_decay:
                    no_decay_head.append(p)
                else:
                    decay_head.append(p)

        optimizer = torch.optim.AdamW([
            {"params": decay_backbone,   "lr": backbone_learning_rate, "weight_decay": weight_decay},
            {"params": no_decay_backbone,"lr": backbone_learning_rate, "weight_decay": 0.0},
            {"params": decay_head,       "lr": learning_rate, "weight_decay": weight_decay},
            {"params": no_decay_head,    "lr": learning_rate, "weight_decay": 0.0},
        ])

        return optimizer