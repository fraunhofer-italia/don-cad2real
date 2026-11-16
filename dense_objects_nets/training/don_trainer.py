# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

from torch import Tensor
from typing import Dict, Any
from abc import ABC, abstractmethod

from dense_objects_nets.network.network import Network

class DonTrainer(ABC):
    """
    Abstract class for a Dense Object Nets trainer object.
    """
    
    @abstractmethod
    def __init__(self, config : Dict[str, Any]):
        pass

    @abstractmethod
    def run(self, model : Network, data : Dict[str, Any]) -> Dict[str, Tensor]:
        """
        Run a training step on the model using the provided data.
        """
        pass
    
    def log_training_metrics(self, outputs_dir: str, out_filename: str = "training_metrics.xlsx"):
        """
        Log training metrics to the console or a file.
        """
        pass
        