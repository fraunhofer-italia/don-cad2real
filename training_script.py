# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import os
import torch
import argparse
import importlib
from typing import Dict, Any, List

from dense_objects_nets.training.training import DonTrainingRun
import dense_objects_nets.utils.io_utils as io_utils
import dense_objects_nets.utils.parameters_utils  as params_utils


def load_class(class_name, module_path="dense_objects_nets.pipelines"):
    try:
        if "." in class_name:
            final_module_path, class_name = class_name.rsplit(".", 1)
            module_path = f"{module_path}.{final_module_path}"

        module = importlib.import_module(module_path)
        class_ref = getattr(module, class_name)
    except Exception as e:
        print(f"Error loading class '{class_name}': {e}")
        exit()
    return class_ref

def load_runs_data(components: Dict[str, Any], runs_config: List[Dict[str, Any]], config_base_dir: str) -> List[Dict]:
    runs_data = []
    for run_config in runs_config:
        training_run_components = run_config["components"]

        # Load the training pipeline class references
        pipeline_info = components["pipelines"][training_run_components["pipeline"]]
        class_references = {
            object_class: load_class(pipeline_info[object_class], module_path="dense_objects_nets.pipelines") 
            for object_class in ["trainer_class", "dataset_class", "evaluator_class"]
        }

        # Load network class and state dict (if any)
        network_info = components["networks"][training_run_components["network"]]
        network_class = load_class(network_info["class"], module_path="dense_objects_nets.network")
        checkpoint_path = network_info.get("checkpoint_path", None)
        starting_epoch = network_info.get("starting_epoch", 1)

        # Load the datasets scenes configs and associated transforms
        dataset_info = components["datasets"][training_run_components["dataset"]]
        dataset_scenes_config = {
            split: io_utils.load_config_file(os.path.join(config_base_dir, "dataset", dataset_info[split]))
            for split in ["train", "validation", "evaluation"]
        }
        dataset_transforms = [
            load_class(transform_class_name, module_path="dense_objects_nets.dataset.transforms")()
            for transform_class_name in dataset_info.get("dataset_transforms", [])
        ]

        # If an hyper-parameters sweep is defined for the current run, generate all the required training config 
        training_config = io_utils.load_config_file(os.path.join(config_base_dir, "training", run_config["training_config"]))

        if "override_config" in run_config:
            for key, value in run_config["override_config"].items():
                training_config = params_utils.override_config_parameter(training_config, param_key=key, param_value=value)

        sweeps = components.get("hparams_sweeps", None)
        if sweeps is not None and "hparams_sweep" in run_config["components"]:
            hparams_sweep = sweeps[run_config["components"]["hparams_sweep"]]
            sweep_combined_parameters, sweep_params_names = params_utils.combine_training_parameters(params_dict=hparams_sweep)
            sweep_training_configs = params_utils.create_combined_training_configs(sweep_combined_parameters, training_config)
        else:
            sweep_training_configs = [training_config]
            sweep_params_names = [""]

        # Load the data needed to create each training run
        for config, hparams_name in zip(sweep_training_configs, sweep_params_names):
            # Update the output directory to include the run name and hyper-parameters values
            run_name = run_config["run_name"] if not hparams_name else f"{run_config['run_name']}_{hparams_name}"
            if "base" not in config: 
                config["base"] = {"outputs_path": "${WS_DIR}/runs"} # Default outputs path
            config["base"]["outputs_path"] = os.path.join(config["base"]["outputs_path"], run_name)
            
            runs_data.append({
                "run_name": run_name,
                "training_config": config,
                "trainer_class": class_references["trainer_class"],
                "evaluator_class": class_references["evaluator_class"],
                "dataset_class": class_references["dataset_class"],
                "network_class": network_class,
                "dataset_scenes_config": dataset_scenes_config,
                "dataset_transforms": dataset_transforms,
                "checkpoint_path": checkpoint_path,
                "starting_epoch": starting_epoch,
            })

    return runs_data

def execute_training_run(run_data: Dict[str, Any]):
    run_training_config = run_data["training_config"]

    # Init trainer and evaluator
    trainer = run_data["trainer_class"](run_training_config["trainer"])
    evaluator = run_data["evaluator_class"](run_training_config["evaluator"])

    # Init datasets
    dataset_config = run_training_config["dataset"]
    dataset_scenes = run_data["dataset_scenes_config"]
    dataset_transforms = run_data["dataset_transforms"]
    
    print("Loading training dataset...")
    train_dataset = run_data["dataset_class"](
        scenes_config=dataset_scenes["train"], dataset_config=dataset_config, transforms=dataset_transforms
    )
    print("Loading validation dataset...")
    val_dataset = run_data["dataset_class"](
        scenes_config=dataset_scenes["validation"], dataset_config=dataset_config, transforms=dataset_transforms
    )
    print("Loading evaluation dataset...")
    eval_dataset = run_data["dataset_class"](
        scenes_config=dataset_scenes["evaluation"], dataset_config=dataset_config 
    )

    # Build model and optimizer
    # If provided, load pretrained state dict from the given paths.
    pretrained_dict = run_data["checkpoint_path"]
    network_class = run_data["network_class"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pretrained_dict is not None and "model" in pretrained_dict and pretrained_dict["model"] is not None:
        model_weights_path = os.path.expandvars(pretrained_dict["model"])
        model = network_class.from_checkpoint(model_weights_path, device=device)
    else:
        model = network_class(network_config=run_training_config["network"])
    model = model.to(device)

    optimizer = model.get_optimizer(optimizer_config=run_training_config["network"]["optimizer"])
    if pretrained_dict is not None and "optimizer" in pretrained_dict and pretrained_dict["optimizer"] is not None:
        optimizer_params_path = os.path.expandvars(pretrained_dict["optimizer"])
        optimizer.load_state_dict(
            torch.load(optimizer_params_path, map_location=device, weights_only=True)
        )

    # Create the training handler and launch the training
    run = DonTrainingRun(
        trainer=trainer,
        evaluator=evaluator,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        evaluation_dataset=eval_dataset,
        model=model,
        optimizer=optimizer,
        training_config=run_training_config,
    )
    run.execute(starting_epoch=int(run_data["starting_epoch"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description =
        "Script used to executes DON training runs given an input config file.\n"
        "Configs are assumed to be stored in `${WS_DIR}/config/training_runs/`.\n"
        "See `_training_runs_example.yml` for a description of the parameters.\n"
        "\n"
        "Example:\n"
        "   python training_script.py --config <path_to_config_file>\n",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True, type=str, help="Name of the input config file. ")
    parser.add_argument("--config_base_dir", type=str, default="${WS_DIR}/config", help="Path to the config directory")
    args = parser.parse_args()

    print("")
    print("Starting DON training runs script...")
    print("")

    # Load the file containing a list of training parameters, one for each training run
    try:
        config_base_dir = os.path.expandvars(args.config_base_dir)
        config_file = os.path.join(config_base_dir, "training_runs", args.config)
        training_runs_config = io_utils.load_config_file(config_file)
        print(f"Using training parameters file: '{config_file}'")
    except Exception as e:
        print(f"Error loading training parameters file {config_file}: {e}")
        exit(1)

    print("Loading training configuration...")
    runs_data = load_runs_data(
        components=training_runs_config["components"], runs_config=training_runs_config["training_runs"], config_base_dir=config_base_dir
    )
    print(f"Loaded a total of {len(runs_data)} training runs to execute.")
    print("")

    # Execute each training run defined in the input
    for i, run_data in enumerate(runs_data):
        print(f"Executing training run ({i+1}/{len(runs_data)}): '{run_data['run_name']}'")
        execute_training_run(run_data)
        print("")
    
    print("")
    print("Training concluded!")
