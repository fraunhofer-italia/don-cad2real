# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>


import torch
import torch.nn as nn
from torch.autograd import Function

from math import exp
from typing import Dict, Any, Tuple

from dense_objects_nets.network.don_backbone import DonNetwork
from dense_objects_nets.network.don_descriptors_network import DonNetworkResNet34, DonNetworkResNet50, DonNetworkConvNext


class GradReverse(Function):
    """
    Gradient Reversal Layer from: "Domain-Adversarial Training of Neural Networks" Ganin, Y. & Lempitsky, V. (2015)
    https://arxiv.org/abs/1505.07818
    """
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

    @staticmethod
    def grad_reverse(x, lambd=1.0):
        return GradReverse.apply(x, lambd)


class DANNDomainClassifier(nn.Module):
    """
    Domain-adversarial head for DON features.
    Input:  [bn, N, D] descriptor vectors (D=24 here)
    Output: [bn, N, 1] domain logit (with sigmoid for probability)
    """
    def __init__(self, descriptors_dim: int, hidden_dims: Tuple[int, int] = (256, 128), dropout: float = 0.2):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(descriptors_dim, hidden_dims[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dims[1], 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, lambd: float = 1.0):
        x = GradReverse.grad_reverse(x, lambd)

        x = self.feature(x)
        x = self.classifier(x)
        return self.sigmoid(x)


class DonDannNetwork(DonNetwork):
    """
    Base class DON network with domain-adversarial head for training with DAAM.

    This class is NOT instantiable since DonNetwork is an abstract base class.
    """
    def __init__(self, network_config: Dict[str, Any]) -> "DonDannNetwork":
        super(DonDannNetwork, self).__init__(network_config)
        
        # Build the domain classifier for adversarial training
        self.domain_classifier = DANNDomainClassifier(descriptors_dim=int(network_config["descriptors_dim"]))
        self.domain_classifier_loss = nn.BCELoss()
        self.domain_classifier_optimizer = torch.optim.AdamW(self.domain_classifier.parameters(), lr=1e-4, weight_decay=5.0e-5)

    def get_lambda(self, p: float, min_lambda: float = 0.2, max_lambda: float = 1.0, lambda_steepness: float = 10.0) -> float:
        """
        Compute the lambda value for the gradient reversal layer based on the training progress p.

        Args:
            p (float): The current training progress (between 0 and 1).
            min_lambda (float, optional): The minimum lambda value. Defaults to 0.2.
            max_lambda (float, optional): The maximum lambda value. Defaults to 1.0.
            lambda_steepness (float, optional): The steepness of the curve. Defaults to 10.0.

        Returns:
            float: The computed lambda value.
        """
        p = min(1.0, max(0.0, p))  # Clamp p to [0, 1]

        # Gradient penalty scale for the domain classifier loss        
        lambd = (2.0 / (1.0 + exp(-lambda_steepness * p))) - 1.0
        lambd = min_lambda + (max_lambda - min_lambda) * lambd
        return lambd

    def get_dann_loss(self, don_descriptors: torch.Tensor, domain_labels: torch.Tensor, lambd: float = 1.0, grad_penalty_scale: float = 0.2) -> torch.Tensor:
        """
        Compute the domain classifier loss with a batch of descriptors and domain labels.

        Args:
        - don_descriptors (`torch.Tensor`): tensor containing the DON descriptors with shape (bn, N, D).
        - domain_labels (`torch.Tensor`): tensor containing the domain labels with shape (bn, N, 1). 1 for real, 0 for sim.
        - lambd (`float`): weight for the gradient reversal layer. Defaults to 1.0.

        Returns:
        - `torch.Tensor`: the computed domain classifier loss.
        """
        domain_preds = self.domain_classifier(don_descriptors, lambd)
        loss = self.domain_classifier_loss(domain_preds, domain_labels)

        # Gradient regularization term (inspired by WGAN-GP)
        # (Similar to https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py#L540)
        if grad_penalty_scale > 0.0 and don_descriptors.requires_grad:
            grad_outputs = torch.ones_like(domain_preds)
            input_grad = torch.autograd.grad(
                outputs=domain_preds,
                inputs=don_descriptors,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]  # (bn, N, D)

            # Gradient norm per sample (sum over feature dim)
            grad_penalty = (input_grad.pow(2).sum(dim=-1)).mean()

            loss = loss + grad_penalty_scale * grad_penalty

        return loss
    
    def dann_update(self, don_descriptors: torch.Tensor, domain_labels: torch.Tensor, lambd: float = 1.0):
        """
        Update the domain classifier parameters based on the computed loss.

        Args:
        - domain_loss (`torch.Tensor`): the computed domain classifier loss.
        """
        if not don_descriptors.requires_grad:
            return
        
        self.domain_classifier_optimizer.zero_grad()
        self.get_dann_loss(don_descriptors, domain_labels, lambd).backward(retain_graph=True)
        self.domain_classifier_optimizer.step()

    def get_optimizer(self, optimizer_config: Dict[str, Any]) -> torch.optim.Optimizer:
        """
        Initialize the optimizer for training.

        Args:
        - optimizer_config (`Dict[str, Any]`): dictionary containing the optimizer configuration.

        Returns:
        - `torch.optim.Optimizer`: the initialized optimizer.
        """
        learning_rate = float(optimizer_config["learning_rate"])
        weight_decay = float(optimizer_config.get("weight_decay", 0.0))

        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            
            # Skip parameters from the domain classifier
            if "domain_classifier" in name:
                continue

            if p.ndim == 1 or name.endswith(".bias"):  # biases and norm params
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = torch.optim.AdamW([
            {"params": decay, "weight_decay": weight_decay, "lr": learning_rate},
            {"params": no_decay, "weight_decay": 0.0, "lr": learning_rate},
        ])

        return optimizer
    
    def load_state_dict(self, state_dict, strict = True, assign = False):
        return super().load_state_dict(state_dict, False, assign)


# ----------------------------------------------------------------------------#
# Actual instantiable classes for different backbone networks                 #
# ----------------------------------------------------------------------------#
class DonDannNetworkResNet34(DonDannNetwork, DonNetworkResNet34):
    pass

class DonDannNetworkResNet50(DonDannNetwork, DonNetworkResNet50):
    pass

class DonDannNetworkConvNext(DonDannNetwork, DonNetworkConvNext):
    pass