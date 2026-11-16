# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

# GUI functions must be imported before any 'cv2' import to avoid conflicts with PyQt5
from dense_objects_nets.utils.gui_utils import select_runs_folders, select_files

import os
import sys
import argparse

from tqdm import tqdm
from copy import deepcopy
from typing import Any, Dict
from collections import defaultdict
from dense_objects_nets.utils import io_utils, evaluation_utils

def load_dataset(evaluation_config: Dict[str, Any], dataset_scenes: Dict[str, Any], training_config: Dict[str, Any], augmentation: bool = False):
    eval_dataset = evaluation_utils.load_class(evaluation_config["dataset_class"])(
        scenes_config=dataset_scenes,
        dataset_config=training_config["dataset"],
    )
    if augmentation:
        augmentation_transforms = [
            evaluation_utils.load_class(f"dense_objects_nets.dataset.transforms.{transform_class_name}")()
            for transform_class_name in evaluation_config.get("transforms", [])
        ]
        eval_dataset.transforms = augmentation_transforms
    
    return eval_dataset


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_batch_size", "-bs", default=200, type=int, help="Maximum number of samples to evaluate in a single batch.")
    parser.add_argument("--training_config", "-tc", action="store_true", help="If provided, ")
    parser.add_argument("--keep_samples", "-ks", action="store_true", help="If set, keeps the loaded evaluation samples between different runs.")
    parser.add_argument("--num_samples", "-ns", required=False, type=int, help="Number of samples to load from the dataset for evaluation. If not provided, all samples are loaded.")
    parser.add_argument("--augment_dataset", "-aug", required=False, action="store_true", default=False, help="Whether to augment the dataset during evaluation.")
    args = parser.parse_args()

    run_folders, _ = select_runs_folders(base_dir="${WS_DIR}/runs", query_file="evaluation_config.yml")
    if not run_folders:
        print("No runs selected.")
        exit(0)

    # Select and load evaluation dataset
    eval_dataset_filenames, eval_dataset_names = select_files(
        base_dir="${WS_DIR}/config/dataset", query_file="**.yml", title="Select Evaluation Dataset Config Files"
    )
    if not eval_dataset_filenames:
        print("No evaluation dataset config files selected.")
        exit(0)

    # Load configurations files for all selected runs
    evaluation_configs = [
        io_utils.load_config_file(os.path.join(run_folder, "evaluation_config.yml"))
        for run_folder in run_folders
    ]
    
    if not args.training_config:
        training_configs = [
            io_utils.load_config_file(os.path.join(run_folder, "training_config.yml"))
            for run_folder in run_folders
        ]
    else:
        train_configs_filenames, train_configs_names = select_files(
            base_dir="${WS_DIR}/config/training", query_file="**.yml", title="Select Training Config File"
        )
        if len(train_configs_filenames) != 1:
            print("Please select a single training config file to use for all runs.")
            exit(1)

        print(f"Using selected training config file: {train_configs_names}")
        training_configs = [
            io_utils.load_config_file(os.path.join("$WS_DIR/config/training", train_configs_filenames[0]))
        ] * len(run_folders)
    

    for dataset_cf, dataset_name in tqdm(zip(eval_dataset_filenames, eval_dataset_names), desc="Evaluation datasets", total=len(eval_dataset_filenames), position=2):
        # Load the dataset configuration
        dataset_scenes = io_utils.load_config_file(dataset_cf)

        # Load dataset and samples using the first training config. 
        # Samples will be re-loaded for each run if the dataset/training config are different
        eval_dataset = load_dataset(evaluation_configs[0], dataset_scenes, training_configs[0], augmentation=args.augment_dataset)

        num_samples = min(len(eval_dataset), args.num_samples if args.num_samples is not None else len(eval_dataset))
        tqdm.write(f"Loading {num_samples} samples from dataset '{dataset_name}'...")
        evaluation_samples = eval_dataset.load_evaluation_samples(num_samples)
        sys.stdout.write('\033[F\033[K') # Clear the previous line

        for training_config, evaluation_config, run_folder in tqdm(zip(training_configs, evaluation_configs, run_folders), desc="Evaluating runs", total=len(run_folders), leave=False, position=3):
            # Check if the dataset class is the same
            if evaluation_config["dataset_class"] != evaluation_configs[0]["dataset_class"]:
                tqdm.write("Re-loading dataset samples because the dataset class is different...")
                eval_dataset = load_dataset(evaluation_config, dataset_scenes, training_config["dataset"], augmentation=args.augment_dataset)
                evaluation_samples = eval_dataset.load_evaluation_samples(num_samples)

            # Check if dataset samples need to be re-loaded
            elif eval_dataset.dataset_config != training_config["dataset"] and (not args.keep_samples):
                tqdm.write("Re-loading dataset samples because the dataset or scenes config is different...")
                evaluation_samples = eval_dataset.load_evaluation_samples(num_samples)

            # Load the evaluator and models
            evaluator = evaluation_utils.load_class(evaluation_config["evaluator_class"])(
                training_config["evaluator"],
            )

            best_model, last_model = evaluation_utils.load_models(evaluation_config)
            best_model_name = evaluation_config["model_best"].removesuffix('.pth')
            last_model_name = evaluation_config["model_last"].removesuffix('.pth')
            best_model.eval()
            last_model.eval()

            # Evaluate the models on the dataset
            tqdm.write("Evaluating:")
            tqdm.write(f"- Run folder: {run_folder}")
            tqdm.write(f"- Dataset: {dataset_name} with {num_samples} samples")
            tqdm.write(f"- Models: '{best_model_name}' and '{last_model_name}'")
            best_model_results = {"Metrics": defaultdict(list)}
            last_model_results = {"Metrics": defaultdict(list)}
            samples_per_batch = [min(args.max_batch_size, num_samples - i) for i in range(0, num_samples, args.max_batch_size)]
            for i, batch_size in tqdm(enumerate(samples_per_batch), desc="Samples batches", leave=False, total=len(samples_per_batch), position=4):
                batch_start = sum(samples_per_batch[:i])
                batch_end = batch_start + batch_size
                samples = evaluation_samples[batch_start:batch_end]

                best_model_batch_results = evaluator.evaluate(best_model, deepcopy(samples))
                last_model_batch_results = evaluator.evaluate(last_model, deepcopy(samples))

                best_model_batch_values = best_model_batch_results["Metrics"]
                last_model_batch_values = last_model_batch_results["Metrics"]

                for key in best_model_batch_values.keys():
                    best_model_results["Metrics"][key].extend(best_model_batch_values[key])
                    last_model_results["Metrics"][key].extend(last_model_batch_values[key])

            # Log the evaluation results on file.
            outname = f"metrics_{dataset_name.replace('.yml', '').replace('/', '_')}.xlsx"
            if args.augment_dataset: outname = outname.replace(".xlsx", "_augmented.xlsx")
            evaluator.log_on_file(
                evaluation_results=best_model_results,
                epoch="best",
                outputs_dir=run_folder,
                outname=outname,
                overwrite=True,
            )
            evaluator.log_on_file(
                evaluation_results=last_model_results,
                epoch="last",
                outputs_dir=run_folder,
                outname=outname,
                overwrite=False,
            )

            tqdm.write(f"DONE! Metrics logged to {outname}")
            tqdm.write("")