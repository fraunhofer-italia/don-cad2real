# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

# GUI functions must be imported before any 'cv2' import to avoid conflicts with PyQt5
from dense_objects_nets.utils.gui_utils import select_runs_folders, selection_window, select_files

import os
import argparse
import cv2 as cv
import numpy as np
import pandas as pd

from dense_objects_nets.utils import io_utils, evaluation_utils
from dense_objects_nets.evaluation.don_evaluator import EvaluationResults


def select_metrics(run_folder: str):
    metrics_df = pd.read_excel(os.path.join(run_folder, "metrics.xlsx"), sheet_name="Values", header=0)

    # Find the evaluation metrics computed in this run
    iteration_indices = metrics_df[metrics_df["Metric"].str.startswith("Iteration")].index.tolist()
    metrics_names = metrics_df["Metric"].iloc[iteration_indices[0] + 1:iteration_indices[-1]].tolist()
    metrics_names = set(metrics_names)  # Remove duplicates
    metrics_names = sorted([metric for metric in metrics_names if not metric.startswith("Iteration")]) # Exclude 'Iteration' separators

    selected_metrics, _ = selection_window(title="Select Metrics", values=metrics_names)
    selected_metrics = sorted(selected_metrics)

    return selected_metrics

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_object", "-mo", required=False, action="store_true", default=False,
                        help="Whether to plot descriptors for the whole image or only for the masked object area.")
    parser.add_argument("--augment_dataset", "-aug", required=False, action="store_true", default=False,
                        help="Whether to augment the dataset during evaluation.")
    args = parser.parse_args()

    run_folders, _ = select_runs_folders(base_dir="${WS_DIR}/runs", query_file="evaluation_config.yml")
    if len(run_folders) != 1:
        print("Please select a single run folder containing an evaluation config file.")
        exit(0)

    evaluation_config_filename = os.path.join(run_folders[0], "evaluation_config.yml")
    evaluation_config = io_utils.load_config_file(filename=evaluation_config_filename)

    # Select the evaluation dataset
    eval_dataset_filenames, eval_dataset_names = select_files(base_dir="${WS_DIR}/config/dataset", query_file="**.yml", title="Select Evaluation Dataset")
    if len(eval_dataset_filenames) != 1:
        print("Please select a single dataset for evaluation.")
        exit(0)
    
    # Select metrics to visualize
    evaluation_metrics = select_metrics(run_folders[0])
    if not evaluation_metrics:
        print("No metrics selected for visualization. Exiting.")
        exit(0)

    # Load evaluation data and models
    evaluator, eval_dataset, best_model, last_model, training_params = evaluation_utils.load_evaluation_data(
        evaluation_config, eval_dataset_filenames[0], mask_object=args.mask_object
    )

    if args.augment_dataset:
        augmentation_transforms = [
            evaluation_utils.load_class(f"dense_objects_nets.dataset.transforms.{transform_class_name}")()
            for transform_class_name in evaluation_config.get("transforms", [])
        ]
        eval_dataset.transforms = augmentation_transforms

    best_model_iteration = evaluation_config["model_best"].removesuffix('.pth').split("_")[-1]
    last_model_iteration = evaluation_config["model_last"].removesuffix('.pth').split("_")[-1]
    win_name = f"Best model: iter {best_model_iteration}   /   Last model: iter {last_model_iteration}"

    print(f"Started predictions evaluation with:")
    print(f"- Evaluation config: {evaluation_config_filename}")
    print(f"- Dataset: {eval_dataset_names[0]}")
    print(f"- Models: best (iter {best_model_iteration}), last (iter {last_model_iteration})")
    print("")

    while True:
        # Get an evaluation sample.
        sample = eval_dataset.load_evaluation_samples(num_samples=1)

        if not sample:
            print("Model evaluation error! Failed to get a valid evaluation sample.")
            continue

        # Evaluate both the best and the last model
        best_model_eval_results = evaluator.evaluate(model=best_model, samples=sample)
        last_model_eval_results = evaluator.evaluate(model=last_model, samples=sample)

        best_model_metrics_dict = best_model_eval_results["Metrics"]    
        last_model_metrics_dict = last_model_eval_results["Metrics"]

        for metric in evaluation_metrics:
            if metric not in best_model_metrics_dict or metric not in last_model_metrics_dict:
                continue

            print(f"Metric:  {metric}")
            print(f"- best model (iter: {best_model_iteration}): {np.mean(best_model_metrics_dict[metric])}")
            print(f"- last model (iter: {last_model_iteration}): {np.mean(last_model_metrics_dict[metric])}")

        # Visualize evaluation images
        for eval_res in best_model_eval_results:
            if eval_res.type != EvaluationResults.Types.IMAGES:
                continue
            
            best_model_img = best_model_eval_results[eval_res][0]
            last_model_img = last_model_eval_results[eval_res][0]

            eval_img = cv.hconcat([best_model_img, last_model_img])

            cv.namedWindow(f"{eval_res.name} -- {win_name}", cv.WINDOW_NORMAL)
            cv.imshow(f"{eval_res.name} -- {win_name}", eval_img)

        print(f"Press any key to continue and 'q' to exit the program.")
        if cv.waitKey(0) == ord('q'):
            print("Prediction visualization ended.")
            break

        print("")
        print("")