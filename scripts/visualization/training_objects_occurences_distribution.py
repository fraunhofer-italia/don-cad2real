# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

# GUI functions must be imported before any 'cv2' import to avoid conflicts with PyQt5
from dense_objects_nets.utils.gui_utils import select_runs_folders

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple
from matplotlib import colors, colormaps


# 30 unique colors for objects visualization
unique_colors = [
    colors.to_rgb(color) 
    for color in list(colormaps['Set3'].colors + colormaps['tab10'].colors + colormaps['Pastel2'].colors)
]


def get_classes_occurrences(run_folder: str, query_file: str) -> Tuple[Dict[str, List[int]], Dict[str, int], int]:
    with open(os.path.join(run_folder, query_file), 'r') as f:
        metrics = json.load(f)

    classes_occurrences = metrics["objects_occurrences_distribution"]
    pixels_count = metrics["objects_matched_pixels_count"]

    num_iterations = max(max(occurrences) for occurrences in classes_occurrences.values())
    
    return classes_occurrences, pixels_count, num_iterations

def plot_occurrences_count(classes_occurrences: Dict[str, List[int]], num_iterations: int):
    class_names = sorted(list(classes_occurrences.keys()), key=lambda x: int(x.split("_")[1]))  # Sort by class index (assuming format "class_X")
    frequencies = [len(classes_occurrences[cls]) for cls in class_names]

    plt.figure(figsize=(10, 6))
    # Use the same color cycle as matplotlib's default (used by plt.scatter/plt.plot below)
    plt.bar(class_names, frequencies, color=unique_colors[:len(class_names)])
    plt.xlabel("Class")
    plt.ylabel("Number of appearances")
    plt.title(f"Total appearances of each class over {num_iterations} iterations")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_class_occurrences(classes_occurrences: Dict[str, List[int]]):
    plt.figure(figsize=(12, 6))

    class_names = sorted(list(classes_occurrences.keys()), key=lambda x: int(x.split("_")[1]), reverse=True) # Reverse so that the first class is on top
    for i, cls in enumerate(class_names):
        occurrences = classes_occurrences[cls]
        plt.scatter(occurrences, [i]*len(occurrences), label=cls, s=3, color=unique_colors[len(class_names)-1-i])

    plt.yticks(range(len(classes_occurrences)), class_names)
    plt.xlabel("Iteration")
    plt.ylabel("Class")
    plt.title("Class appearances over training iterations")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_smoothed_class_frequency(classes_occurrences: Dict[str, List[int]], num_iterations: int, window_fraction: float = 0.1):
    class_counts = {}
    for cls, occurrences in classes_occurrences.items():
        counts = np.zeros(num_iterations + 1)  # +1 to include the last iteration
        for idx in occurrences:
            counts[idx] += 1

        # Smooth with rolling window
        window_size = max(1, int(num_iterations * window_fraction))
        smoothed = np.convolve(counts, np.ones(window_size)/window_size, mode='same')
        class_counts[cls] = smoothed

    plt.figure(figsize=(12, 6))
    class_names = sorted(list(class_counts.keys()), key=lambda x: int(x.split("_")[1]))
    for i, cls in enumerate(class_names):
        plt.plot(class_counts[cls], label=cls, color=unique_colors[i])

    plt.xlabel("Iteration")
    plt.ylabel(f"Smoothed Appearance Frequency (window={window_size})")
    plt.title("Class appearance trends over training")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pixels_count(pixels_count: Dict[str, int]):
    class_names = sorted(list(pixels_count.keys()), key=lambda x: int(x.split("_")[1]))
    counts = [pixels_count[cls] for cls in class_names]

    plt.figure(figsize=(10, 6))
    plt.bar(class_names, counts, color=unique_colors[:len(class_names)])
    plt.xlabel("Class")
    plt.ylabel("Number of matched pixels")
    plt.title("Total matched pixels for each class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric_filename", "-m", default="trainer_metrics.json", help="Name of the queried metrics file.")
    args = parser.parse_args()

    runs_folders, _ = select_runs_folders(base_dir="${WS_DIR}/runs", query_file=args.metric_filename)
    if len(runs_folders) != 1:
        print("Please select a single run folder containing an evaluation config file.")
        exit(0)

    classes_occurrences_dict, pixels_count, num_iterations = get_classes_occurrences(runs_folders[0], query_file=args.metric_filename)
    plot_occurrences_count(classes_occurrences_dict, num_iterations)
    plot_class_occurrences(classes_occurrences_dict)
    plot_smoothed_class_frequency(classes_occurrences_dict, num_iterations)
    plot_pixels_count(pixels_count)