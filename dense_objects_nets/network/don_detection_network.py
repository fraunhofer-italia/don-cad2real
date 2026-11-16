# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import os
import timm
import torch
import torch.nn as nn

from torchvision.ops import nms
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

from dense_objects_nets.network import utils as nn_utils
from dense_objects_nets.network.network import Network
from dense_objects_nets.utils import evaluation_utils


# Support function to create a grid of points corresponding to the feature map locations
def make_grid(h: int, w: int, stride: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(h, device=device) + 0.5
    xs = torch.arange(w, device=device) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack((xx, yy), dim=-1) * stride  # Pixel-space centers of stride grid (in absolute pixels coordinates)
    return pts.view(-1, 2) # (H*W, 2)

class ConvBNAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3):
        super(ConvBNAct, self).__init__()

        self.block = nn_utils.ConvModule(
            in_channels=c_in,
            out_channels=c_out,
            norm=nn_utils.Normalization.BATCH_NORM(channels=c_out),
            activation=nn_utils.Activation.SILU(),
            kernel_size=k,
            padding=k//2,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DonDetectionNetwork(Network):
    def __init__(self, network_config: Dict[str, Any]):
        super(DonDetectionNetwork, self).__init__(network_config)

        features_dim = int(network_config.get("features_dim", 128))

        self.num_classes = int(network_config["num_classes"])
        self.bin_size = float(network_config["bin_size"])
        self.num_bins = int(network_config["bins"])
        self.stride = int(network_config["stride"])

       # Create the backbone of the network
        self.backbone, bk_output_channels = self.load_backbone(network_config)

        # Whether to freeze the backbone during training
        self.freeze_backbone = network_config["freeze_backbone"]
        self.backbone_ctx = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        if self.freeze_backbone:
            self.backbone.eval()

        # Shared stem
        self.stem = ConvBNAct(bk_output_channels, features_dim, k=1)

        # Classification and regression towers
        self.cls_tower = nn.Sequential(
            ConvBNAct(features_dim, features_dim),
            ConvBNAct(features_dim, features_dim)
        )

        self.reg_tower = nn.Sequential(
            ConvBNAct(features_dim, features_dim),
            ConvBNAct(features_dim, features_dim)
        )

        # Prediction heads
        self.cls_pred = nn.Conv2d(features_dim, self.num_classes, 1) # Class prediction head
        self.ctr_pred = nn.Conv2d(features_dim, 1, 1) # Center-ness prediction head
        self.reg_pred = nn.Conv2d(features_dim, 4 * (self.num_bins + 1), 1) # Bbox regression prediction head (DFL logits)

        self.dfl = nn_utils.DFLIntegral(bins=self.num_bins)
        self.reg_scale = nn.Parameter(torch.tensor(1.0))

    def load_backbone(self, network_config: Dict[str, Any]) -> Tuple[Network, int]:
        backbone_class = network_config.get("backbone_class", None)

        if backbone_class is None:
            print("No backbone class provided, using standard ResNet34 as backbone.")
            # Load standard ResNet34 as backbone, with last two layers removed
            resnet34_original = timm.create_model("resnet34", pretrained=True, output_stride=8)        

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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        with self.backbone_ctx:
            backbone_features = self.backbone(x)

        # don_descriptors: (B, D, H, W)
        B, _, h, w = backbone_features.shape
        feat = self.stem(backbone_features)  # (B, features_dim, h, w)

        cls_feat = self.cls_tower(feat)
        reg_feat = self.reg_tower(feat)

        cls_logits = self.cls_pred(cls_feat) # (B, K, h, w)
        ctr_logits = self.ctr_pred(reg_feat) # (B, 1, h, w)
        reg_raw = self.reg_pred(reg_feat) * self.reg_scale # (B, 4*(bins+1), h, w)

        return dict(
            cls_logits=cls_logits,
            ctr_logits=ctr_logits,
            reg_raw=reg_raw,
            fmap_size=(h, w),
            stride=self.stride
        )

    @torch.no_grad()
    def predict(self, x: torch.Tensor, score_thresh: float = 0.4, iou_thresh: float = 0.6, max_det: int = 20, debug: bool = False) -> List[Dict[str, torch.Tensor]]:
        # Switch to eval mode
        mode = self.training
        self.eval()
        
        out = self.forward(x)
        B, _, H, W = x.shape

        # Decode regression (DFL)
        dist_bins = self.dfl(out["reg_raw"])                 # (B, 4, h, w) in bins
        dist_px   = dist_bins * self.bin_size * self.stride  # convert to pixels

        # Get feature map grid
        h, w = out["fmap_size"]
        points_xy = make_grid(h, w, out["stride"], x.device)

        cx = points_xy[:, 0].view(1, h, w).expand(B, -1, -1)
        cy = points_xy[:, 1].view(1, h, w).expand(B, -1, -1)

        l, t, r, b = dist_px[:, 0], dist_px[:, 1], dist_px[:, 2], dist_px[:, 3]
        x1 = (cx - l).clamp(min=0)
        y1 = (cy - t).clamp(min=0)
        x2 = cx + r
        y2 = cy + b

        boxes = torch.stack([x1, y1, x2, y2], dim=3).view(B, -1, 4)

        # Scores
        cls_prob = torch.sigmoid(out["cls_logits"]).permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
        ctr_prob = torch.sigmoid(out["ctr_logits"]).permute(0, 2, 3, 1).reshape(B, -1, 1)
        scores   = cls_prob * ctr_prob

        preds = []
        for b in range(B):
            boxes_b  = boxes[b]
            scores_b = scores[b]

            smax, labels = scores_b.max(dim=1)
            keep = smax > score_thresh
            boxes_b = boxes_b[keep]; smax = smax[keep]; labels = labels[keep]

            if boxes_b.numel() == 0: # No detections
                preds.append(dict(
                    boxes=boxes_b,
                    scores=smax,
                    labels=labels,

                    # Include the raw logits and feature maps if debugging is enabled
                    cls_prob=cls_prob if debug else None,
                    ctr_prob=ctr_prob if debug else None,
                    fmap_size=(h, w) if debug else None,
                ))
                continue

            keep_idx = []
            for c in labels.unique():
                idx = (labels == c).nonzero(as_tuple=False).squeeze(1)
                kept = nms(boxes_b[idx], smax[idx], iou_thresh)
                keep_idx.append(idx[kept])
            keep_idx = torch.cat(keep_idx, 0)

            if keep_idx.numel() > max_det:
                topk = smax[keep_idx].topk(max_det).indices
                keep_idx = keep_idx[topk]

            preds.append(dict(
                boxes=boxes_b[keep_idx],
                scores=smax[keep_idx],
                labels=labels[keep_idx] + 1,  # +1 to match the 1-based class indexing

                # Include the raw logits and feature maps if debugging is enabled
                cls_prob=cls_prob if debug else None,
                ctr_prob=ctr_prob if debug else None,
                fmap_size=(h, w) if debug else None,
            ))

        # Restore previous mode
        self.train(mode)

        return preds

    def train(self, mode = True):
        if not self.freeze_backbone: self.backbone.train(mode)
        self.stem.train(mode)
        self.cls_tower.train(mode)
        self.reg_tower.train(mode)
        self.cls_pred.train(mode)
        self.ctr_pred.train(mode)
        self.reg_pred.train(mode)
        self.dfl.train(mode)
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