# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import os
import torch
import random
import importlib
import torchvision

from torchvision.transforms import v2
from typing import Any, Dict, Tuple

from dense_objects_nets.utils import io_utils
from dense_objects_nets.dataset.don_dataset import DonDataset
from dense_objects_nets.evaluation.don_evaluator import DonEvaluator


def load_class(obj_name):
    try:
        module_path, class_name = obj_name.rsplit(".", 1)
                
        module = importlib.import_module(module_path)
        class_ref = getattr(module, class_name)
    except Exception as e:
        print(f"Error loading class '{obj_name}': {e}")
        exit()
    return class_ref

def load_models(eval_config: Dict[str, Any]) -> Tuple[torch.nn.Module, torch.nn.Module]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_path = os.path.expandvars(eval_config["run_path"])
    best_model_path = os.path.join(run_path, eval_config["model_best"])
    last_model_path = os.path.join(run_path, eval_config["model_last"])

    network_class_ref = load_class(eval_config['network_class'])
    best_model = network_class_ref.from_checkpoint(best_model_path, device)
    last_model = network_class_ref.from_checkpoint(last_model_path, device)
    best_model.eval()
    last_model.eval()

    return best_model, last_model

def load_run_configs(eval_config: Dict[str, Any]) -> Dict[str, Any]:
    training_run_data_path = eval_config["run_path"]
    training_params = io_utils.load_config_file(os.path.join(training_run_data_path, "training_config.yml"))
    scenes_config = io_utils.load_config_file(os.path.join(training_run_data_path, "dataset_config.yml"))

    # Retrocompatibility with old evaluation/dataset configs
    if eval_config["evaluation_split"] == "evaluation" and "evaluation" not in scenes_config:
        eval_config["evaluation_split"] = "validation"

    dataset_scenes = scenes_config[eval_config["evaluation_split"]]

    return training_params, dataset_scenes

def load_evaluation_data(eval_config: Dict[str, Any], dataset_filename: str = None, evaluation_split: str = "evaluation", mask_object: bool = False) -> Tuple[DonEvaluator, DonDataset, torch.nn.Module, torch.nn.Module, Dict[str, Any]]:
    eval_config["plot_object_only"] = mask_object
    eval_config["evaluation_split"] = evaluation_split
    training_params, scenes_config = load_run_configs(eval_config)
    
    # If a dataset filename is provided, override the scenes config
    if dataset_filename is not None:
        scenes_config = io_utils.load_config_file(dataset_filename)

    # Load the evaluation dataset.
    dataset_class = load_class(eval_config["dataset_class"])
    evaluation_dataset = dataset_class(scenes_config, dataset_config=training_params["dataset"])
    evaluation_dataset.seed = training_params["base"].get("dataset_seed", None)

    # Load the evaluator class.
    evaluation_class = load_class(eval_config["evaluator_class"])
    evaluator = evaluation_class(training_params["evaluator"])

    # Load the two trained models that we want to evaluate.
    best_model, last_model = load_models(eval_config)

    return evaluator, evaluation_dataset, best_model, last_model, training_params

def randomize_background(imgs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if not hasattr(randomize_background, "_dataset"):
        randomize_background._dataset = torchvision.datasets.Caltech256(root=os.path.expandvars("${WS_DIR}/data/"), download=True)

    def sample_imgs(samples_num=1):
        imgs = []
        for i in random.sample(range(0, len(randomize_background._dataset)), k=samples_num):
            img, label = randomize_background._dataset[i]
            img = v2.ToImage()(img)
            img = v2.RGB()(img)
            img = v2.ToDtype(torch.float32, scale=True)(img)
            imgs.append(img)

        return imgs

    H, W = imgs.size()[-2:]

    rand_bg_images = []
    random_imgs = sample_imgs(samples_num=len(imgs))
    for img, rand_img, mask in zip(imgs, random_imgs, masks):
        background_img = v2.Resize(size=[H, W])(rand_img)
        background_img = background_img.to(img.device)

        mask_pixels = torch.nonzero(mask)

        background_img[:, mask_pixels[:, 0], mask_pixels[:, 1]] = img[:, mask_pixels[:, 0], mask_pixels[:, 1]]

        rand_bg_images.append(background_img)

    rand_bg_images = torch.stack(rand_bg_images) # (num_imgs, 3, H, W)

    return rand_bg_images
