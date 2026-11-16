# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

"""

"""
from collections import defaultdict
import tensorflow as tf
import pandas as pd
import glob
import os

import dense_objects_nets.utils.io_utils as io_utils 


def get_scalars_metrics_values(metrics_tags, filepath):
    metrics_values = defaultdict(lambda : ([], []))
    for e in tf.compat.v1.train.summary_iterator(filepath):
        if len(e.summary.value)>0: #Skip first empty element
            tag = e.summary.value[0].tag
            value = e.summary.value[0].simple_value
            step = e.step

            if tag in metrics_tags:
                tag_values, tag_steps = metrics_values[tag]
                tag_values.append(value)
                tag_steps.append(step)

    return metrics_values

def get_training_run_id(run_path):
    training_config = io_utils.load_config_file(filename=f"{run_path}/training_config.yml")
    D = str(training_config["network"]["descriptors_dim"])
    M = str(training_config["training"]["loss_margin"])
    LR = str(training_config["optimizer"]["learning_rate"])
    WD = str(training_config["optimizer"]["weight_decay"])
    
    dataset_config = io_utils.load_config_file(filename=f"{run_path}/dataset_config.yml")
    dataset_type = "synthetic" if "synthetic" in dataset_config["train"]["dataset_base_path"] else "real"

    ID = f"{D}_{M}_{LR}_{WD}_{dataset_type}"
    return ID

if __name__ == "__main__":
    # Base path to the directory containing the training runs outputs
    base_path = os.path.expandvars("${WS_DIR}/runs/cv_runs/cv_obj1")
    
    metrics_tags = [
        "Descriptors Metrics/Avg Pred Match Descriptors Distance",
        "Descriptors Metrics/Avg Real Match Descriptors Distance",
        "Descriptors Metrics/Avg Img2 Match Descriptors Distance",
        "Pixels Metrics/Avg Match Points Distance",
        "Pixels Metrics/Avg Match Pixels Distance",
        "Pixels Metrics/AUC PCK@k for k in [1, 100]"
    ]


    # ------------------------------------------------ #


    # Load all tfevents
    tfevents = sorted(glob.glob(os.path.join(base_path, "*/*.tfevents*")))

    metrics_values = defaultdict(lambda : [])
    for event_name in tfevents:
        run_dir_path  = event_name[:event_name.rfind("/")]
        params_id = get_training_run_id(run_path=run_dir_path)

        run_metrics = get_scalars_metrics_values(metrics_tags=metrics_tags, filepath=event_name)

        for metric_tag, (metric_values, metric_steps) in run_metrics.items():
            metrics_values[metric_tag].append((params_id, metric_values, metric_steps))

    
    with pd.ExcelWriter('scripts/output.xlsx') as writer:
        for metric in metrics_tags:
            metrics_by_step = defaultdict(lambda : [])
            for params_id, metric_values, metric_steps in metrics_values[metric]:
                metrics_by_step[tuple(metric_steps)].append((params_id, metric_values))

            for metric_steps, metrics_groups  in metrics_by_step.items():
                row_ids = [params_id for params_id, _ in metrics_groups]
                columns = metric_steps
                values = [metric_values for _, metric_values in metrics_groups]
                
                df = pd.DataFrame(values, index=row_ids, columns=columns)
                df.to_excel(writer, sheet_name=metric.split("/")[-1].replace("[", "").replace("]", ""))
                


        



