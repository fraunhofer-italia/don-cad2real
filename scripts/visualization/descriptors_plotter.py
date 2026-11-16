# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

# GUI functions must be imported before any 'cv2' import to avoid conflicts with PyQt5
from dense_objects_nets.utils.gui_utils import select_runs_folders, select_files

import matplotlib
matplotlib.use("Qt5Agg")

import os
import torch
import argparse

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors
from typing import Dict, List, Tuple
from collections import defaultdict

from dense_objects_nets.utils import io_utils, evaluation_utils
from dense_objects_nets.utils.don_snapshot import DonSnapshotInfo, DonSnapshot
from dense_objects_nets.network.don_backbone import DonNetwork
from dense_objects_nets.dataset.don_dataset import DonDataset

# If cuml is available, use it for faster descriptors dimensionality reduction
try:
    from cuml.manifold import TSNE
    CUML_AVAILABLE = True
except ImportError:
    from sklearn.manifold import TSNE
    CUML_AVAILABLE = False
    print("cuML not found. Using scikit-learn t-SNE implementations.") 

# Set random seeds for reproducibility
torch.manual_seed(42)

# 30 unique colors for objects visualization
unique_colors = [
    colors.to_rgb(color) 
    for color in list(matplotlib.colormaps['Set3'].colors + matplotlib.colormaps['tab10'].colors + matplotlib.colormaps['Pastel2'].colors)
]

# Create a single figure and axis for plotting
fig, ax = None, None
valid_ids = None


def plot(perplexity_objects_descriptors: Dict[int, Dict[int, np.ndarray]], snapshots: List[DonSnapshot], img_per_row: int = 4, max_visualized_snapshots: int = 20) -> bool:
    global unique_colors

    perplexities = sorted(perplexity_objects_descriptors.keys())

    for ax, perplexity in zip(axes, perplexities):
        ax.cla()
        objects_descriptors = perplexity_objects_descriptors[perplexity]
        ax.set_title(f"t-SNE Descriptors (perplexity={perplexity})")
    
        for obj_id in sorted(objects_descriptors.keys()):
            model_obj_pred = objects_descriptors[obj_id]

            color = unique_colors[obj_id % len(unique_colors)]
            ax.scatter(model_obj_pred[:, 0], model_obj_pred[:, 1], c=[color], s=1, label=f"Object {obj_id}")

        # Set plot title and labels
        handles, labels = ax.get_legend_handles_labels()

        ids = [int(label.split(" ")[-1]) for label in labels]
        sorted_indices = np.argsort(ids)
        sorted_handles = [handles[i] for i in sorted_indices]
        sorted_labels = [labels[i] for i in sorted_indices]                

        ax.legend(sorted_handles, sorted_labels, markerscale=5)

    # Prepare images for visualization
    H, W = snapshots[0].shape
    imgs = []
    for s in snapshots[:max_visualized_snapshots]:
        img = cv.cvtColor(s.rgb.permute(1, 2, 0).cpu().numpy(), cv.COLOR_RGB2BGR)

        objects_overlay = np.zeros((H, W, 3), dtype=np.float32)
        for obj_id in s.objects_ids:
            if valid_ids is not None and obj_id not in valid_ids:
                continue

            obj_mask = (s.mask == obj_id).cpu().numpy()

            color = unique_colors[obj_id % len(unique_colors)]
            objects_overlay[obj_mask] = color

        img = cv.addWeighted(img, 0.5, cv.cvtColor(objects_overlay, cv.COLOR_RGB2BGR), 0.5, 0.0)
        imgs.append(img)


    # Stack images into a single one, with padding if necessary
    if len(imgs) > 1:
        num_padding_imgs = (img_per_row - len(imgs) % img_per_row) % img_per_row
        imgs += [np.zeros((H, W, 3), dtype=np.uint8)] * num_padding_imgs
        img = np.vstack([np.hstack(imgs[i:i + img_per_row]) for i in range(0, len(imgs), img_per_row)])
    else:
        img = imgs[0]

    cv.namedWindow("Snapshot", cv.WINDOW_NORMAL)
    cv.imshow("Snapshot", img)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)  # Allow the plot to update
    
    while True:
        key = cv.waitKey(10)
        if key == ord(' '): 
            print("\nContinuing to next snapshot...")
            return True
        elif key == ord('q'):
            print("Exiting visualization.")
            return False

def get_2d_descriptors(model: DonNetwork, snapshots: List[DonSnapshotInfo], tsne_metric: str, num_iterations: int, perplexities: List[int]) -> Dict[int, Dict[int, np.ndarray]]:
    global valid_ids

    model_objects_descriptors = defaultdict(lambda: [])
    
    with torch.no_grad():
        descriptors_imgs = model(torch.stack([s.rgb for s in snapshots]).to(model.device))  # (N, D, H, W)
        descriptors_imgs = descriptors_imgs.permute(0, 2, 3, 1)  # (N, H, W, D)

    for snapshot, desc_img in zip(snapshots, descriptors_imgs):
        for obj_id in set(snapshot.objects_ids):
            if valid_ids is not None and obj_id not in valid_ids:
                continue

            obj_mask = (snapshot.mask == obj_id)
            model_descriptors = desc_img[obj_mask].cpu().numpy()

            model_objects_descriptors[obj_id].append(model_descriptors)

    model_objects_descriptors = {obj_id: np.concatenate(descriptors, axis=0) for obj_id, descriptors in model_objects_descriptors.items()}

    D = descriptors_imgs.shape[-1]
    if D <= 2:
        print("Descriptors dimension <= 2, skipping dimensionality reduction.")
        return model_objects_descriptors

    objects_ids = list(model_objects_descriptors.keys()) # Ensure objects are processed in the same order for both models
    snapshots_descriptors = np.concatenate([model_objects_descriptors[obj_id] for obj_id in objects_ids], axis=0)

    perplexity_models_descriptors = {}
    for perplexity in perplexities:
        print(f"Running t-SNE dimensionality reduction (perplexity={perplexity}, num_iterations={num_iterations})...")
        if CUML_AVAILABLE:
            tsne = TSNE(
                n_components=2, 
                metric=tsne_metric, 
                init='pca', 
                perplexity=perplexity, 
                n_iter=num_iterations,
                n_neighbors=perplexity * 3,
                random_state=42
            )
        else:
            tsne = TSNE(
                n_components=2, 
                metric=tsne_metric, 
                init='pca', 
                perplexity=perplexity, 
                max_iter=num_iterations,
                random_state=42
            )
        tsne_snapshots_descriptors = tsne.fit_transform(snapshots_descriptors)

        # Repack descriptors to match the original object IDs
        desc_index = 0
        tsne_model_objects_descriptors = {}
        for obj_id in objects_ids:
            num_descriptors = len(model_objects_descriptors[obj_id])
            tsne_model_objects_descriptors[obj_id] = tsne_snapshots_descriptors[desc_index:desc_index + num_descriptors]
            desc_index += num_descriptors

        perplexity_models_descriptors[perplexity] = tsne_model_objects_descriptors

    print("Dimensionality reduction DONE!")
    return perplexity_models_descriptors

def load_snapshots(dataset: DonDataset, num_snapshots: int, tries: int = 10) -> Tuple[List[DonSnapshot]]:
    global valid_ids

    loaded_snapshots = []
    while tries > 0:
        tries -= 1

        snapshots_ids = torch.randperm(len(dataset.snapshots_info_))[:num_snapshots]

        snapshots = [
            dataset.snapshots_[i] for i in snapshots_ids
            if (valid_ids is None or (set(dataset.snapshots_info_[i].objects_ids) & set(valid_ids)))
        ]
        loaded_snapshots.extend(snapshots)

        if len(loaded_snapshots) >= num_snapshots:
            return loaded_snapshots[:num_snapshots]

    raise ValueError(f"Failed to load {num_snapshots} valid snapshots after {tries} tries.")

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cosine", "-cos", required=False, action="store_true", default=False,
                        help="Whether to use cosine distance for t-SNE. If 'use_tsne' is not set, this flag is ignored.")
    parser.add_argument("--use_euclidean", "-l2", required=False, action="store_true", default=False,
                        help="Whether to use Euclidean distance for t-SNE. If 'use_tsne' is not set, this flag is ignored.")
    parser.add_argument("--perplexities", "-p", nargs='+', type=int, default=[20, 30, 50], help="Perplexities for t-SNE dimensionality reduction.")
    parser.add_argument("--num_iterations", "-ni", default=10000, help="Number of iterations for t-SNE dimensionality reduction.")
    parser.add_argument("--use_last_model", "-lm", required=False, action="store_true", default=False,
                        help="Whether to use the last model for visualization. If not set, the best model is used.")
    parser.add_argument("--window_size", "-w", default=1, help="Number of snapshots grouped before clustering/dimensionality reduction.")
    args = parser.parse_args()

    # Check that only one dimensionality reduction method is selected
    if (not args.use_cosine and not args.use_euclidean) or (args.use_cosine and args.use_euclidean):
        print("Please select a t-SNE distance metric: either '--use_cosine' or '--use_euclidean'.")
        exit(0)

    run_folders, _ = select_runs_folders(base_dir="${WS_DIR}/runs", query_file="evaluation_config.yml")
    if len(run_folders) != 1:
        print("Please select a single run folder containing an evaluation config file.")
        exit(0)

    eval_dataset_filenames, _ = select_files(base_dir="${WS_DIR}/config/dataset", query_file="**.yml")
    if len(eval_dataset_filenames) != 1:
        print("Please select a single dataset config file.")
        exit(0)

    evaluation_config = io_utils.load_config_file(os.path.join(run_folders[0], "evaluation_config.yml"))
    training_config = io_utils.load_config_file(os.path.join(run_folders[0], "training_config.yml"))
    dataset_config = io_utils.load_config_file(eval_dataset_filenames[0])

    evaluation_dataset = evaluation_utils.load_class(evaluation_config["dataset_class"])(
        scenes_config=dataset_config,
        dataset_config=training_config["dataset"]
    )

    evaluator = evaluation_utils.load_class(evaluation_config["evaluator_class"])(
        training_config["evaluator"],
    )

    best_model, last_model = evaluation_utils.load_models(evaluation_config)

    # Load valid IDs if available
    if hasattr(evaluation_dataset, "valid_ids"):
        valid_ids = evaluation_dataset.valid_ids

    # Set model and dimensionality reduction parameters
    model = last_model if args.use_last_model else best_model
    tsne_metric = ("euclidean" if args.use_euclidean else "cosine")
    perplexities = [int(p) for p in args.perplexities]
    num_iterations = int(args.num_iterations)

    # Prepare plotting figure
    fig, axes = plt.subplots(1, len(perplexities), figsize=(12, 6))

    visualization_running = True
    print("Press 'space' to continue to the next snapshot, or 'q' to quit.")
    while visualization_running:
        snapshots_group = load_snapshots(evaluation_dataset, int(args.window_size))

        objects_descriptors = get_2d_descriptors(
            model, snapshots_group, tsne_metric=tsne_metric, num_iterations=num_iterations, perplexities=perplexities
        )

        visualization_running = plot(objects_descriptors, snapshots_group)