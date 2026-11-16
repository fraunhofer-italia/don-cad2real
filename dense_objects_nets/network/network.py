# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>


import torch
import torch.nn as nn

from typing import Dict, Any

class Network(nn.Module):

    def __init__(self, network_config: Dict[str, Any], *args, **kwargs):
        """
        Base network class
        """
        super().__init__()
        self.network_config_ = network_config

    def save_checkpoint(self, filepath: str):
        """
        Save the model weights and configuration to the given directory.

        Args:
        - filepath (`str`): path to the file where to save the model checkpoint.
        """
        checkpoint = {
            "name": f"{self.__module__}.{self.__class__.__name__}",
            "state_dict": self.state_dict(),
            "network_config": self.network_config_
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def from_checkpoint(cls, filepath: str, device: torch.device = torch.device("cpu")):
        """
        Load model from a checkpoint file.

        Args:
        - filepath (`str`): path to the checkpoint file.
        - device (`torch.device`): device to load the model to. Defaults to CPU.
        """
        checkpoint = torch.load(filepath, map_location=device)

        model = cls(network_config=checkpoint["network_config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        return model
    
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
            if p.ndim == 1 or name.endswith(".bias"):  # biases and norm params
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = torch.optim.AdamW([
            {"params": decay, "weight_decay": weight_decay, "lr": learning_rate},
            {"params": no_decay, "weight_decay": 0.0, "lr": learning_rate},
        ])

        return optimizer

    @property
    def device(self):
        return next(self.parameters()).device