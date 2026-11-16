# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import os
import openpyxl
import cv2 as cv
import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from torch.utils.tensorboard import SummaryWriter

from dense_objects_nets.network.network import Network

@dataclass
class EvaluationResults:
    class Types:
        METRICS = "metrics"
        IMAGES = "images"

    name: str
    type: str = Types.METRICS

    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, EvaluationResults):
            return self.name == other.name
        else:
            return NotImplemented

@dataclass
class Metric:
    name: str
    log_per_object: bool = True

    def rename(self, new_name: str):
        return Metric(new_name, self.log_per_object)

    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, Metric):
            return self.name == other.name
        else:
            return NotImplemented


class DonEvaluator(ABC):
    """
    Abstract class for a Dense Object Nets evaluator object.
    """

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        pass

    @abstractmethod
    def evaluate(self, model : Network, samples : List[Dict[str, Any]]) -> Dict[EvaluationResults, Any]:
        """
        Evaluate the given input model on the given evaluation data samples returning some metrics
        """
        pass

    def log_on_tensorboard(self, evaluation_results: Dict[str, EvaluationResults], logger: SummaryWriter, epoch: int, log_images: bool = True):
        for ev_res, data in evaluation_results.items():
            if ev_res.type == EvaluationResults.Types.METRICS: # Expecting a Dict[Metric, List[float]]
                for metric, values in data.items():
                    if values:
                        logger.add_scalar(metric.name, np.mean(values), epoch)

            elif ev_res.type == EvaluationResults.Types.IMAGES and log_images: # Expecting a List[np.ndarray] of images
                out_imgs = np.array([cv.cvtColor(img, cv.COLOR_BGR2RGB) for img in data]) # Tensorboard requires RGB
                logger.add_images(f"{ev_res.name}/Epoch {epoch}", out_imgs, global_step=epoch, dataformats="NHWC",)

    def log_on_file(self, evaluation_results: Dict[str, EvaluationResults], epoch: int, outputs_dir: str, outname: str = "metrics.xlsx", overwrite: bool = False):
        metrics_values = evaluation_results.get("Metrics", None)
        if metrics_values is None:
            return  # No metrics to log

        metric_stats = {
            metric.name: {
                "Num samples": len(values),
                "Avg": np.mean(values) if values else np.nan,
                "Std": np.std(values) if values else np.nan,
                "Median": np.median(values) if values else np.nan,
                "Min": np.min(values) if values else np.nan,
                "Max": np.max(values) if values else np.nan,
            }
            for metric, values in metrics_values.items()
        }

        # Return if no metrics to log
        if not metric_stats: return 

        # Load or create workbook and worksheets
        file_path = os.path.join(os.path.expandvars(outputs_dir), outname)
        
        if overwrite and os.path.exists(file_path):
            os.remove(file_path)
        
        if not os.path.exists(file_path):
            wb = openpyxl.Workbook()
            
            ws1 = wb.active
            ws1.title = "Metrics Stats"
            ws1.append(["Metric", "Num samples", "Avg", "Std", "Median", "Min", "Max"])

            ws2 = wb.create_sheet("Values")
            ws2.append(["Metric", "Values"])
        else:
            wb = openpyxl.load_workbook(file_path)
            ws1 = wb["Metrics Stats"]
            ws2 = wb["Values"]
    
        # Write epoch label in first column
        ws1.append([f"Iteration {epoch}"])
        ws2.append([f"Iteration {epoch}"])

        # Log the statistics for each metric
        metrics_names = sorted(metric_stats.keys())
        for name in metrics_names:
            stats = metric_stats[name]
            ws1.append([
                name, stats["Num samples"], stats["Avg"], stats["Std"], stats["Median"], stats["Min"], stats["Max"]
            ])

        # Log the values for each metric
        for name in metrics_names:
            values = metrics_values[name]
            ws2.append([name] + list(values))
        
        wb.save(file_path)