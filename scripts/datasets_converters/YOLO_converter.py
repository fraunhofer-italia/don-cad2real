# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

"""
Convert dataset in our custom format to YOLO format.
"""

import os
import torchvision
from tqdm import tqdm

from dense_objects_nets.utils import io_utils
from dense_objects_nets.utils.don_snapshot import DonSnapshot


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-bp", type=str, required=True, help="Path dataset root")
    parser.add_argument("--output_path", "-op", type=str, required=True, help="Path to save the converted dataset")
    parser.add_argument("--split", "-s", default="train", help="Splits to convert (train/test/val)")
    parser.add_argument("--valid_ids", "-v", nargs="+", type=int, default=None)
    args = parser.parse_args()

    valid_ids = args.valid_ids if args.valid_ids is not None else []

    # Create output directories
    output_dir = io_utils.create_directory(args.output_path, "")
    output_imgs_dir = io_utils.create_directory(output_dir, f"images/{args.split}")
    output_labels_dir = io_utils.create_directory(output_dir, f"labels/{args.split}")

    # Load scene snapshots
    imgs_counter = 0
    for scene in tqdm(os.listdir(args.base_path), desc="Processing scenes"):
        scene_base_path = os.path.join(args.base_path, scene)

        if not os.path.isdir(scene_base_path):
            continue

        try:
            scene_snapshots = io_utils.load_config_file(os.path.join(scene_base_path, "snapshots.yml"))
        except FileNotFoundError:
            print(f"Skipping scene {scene} as snapshots.yml is missing.")
            continue

        for id, snapshot_dict in tqdm(scene_snapshots.items(), desc="Processing scene snapshots", leave=False):
            s = DonSnapshot.from_config_dict(scene_base_path, snapshot_dict)
            snapshot_objects = s.objects

            # Filter objects by valid ids (if provided) 
            if len(valid_ids) > 0:
                snapshot_objects = [obj for obj in snapshot_objects if obj.id in valid_ids]

            yolo_labels = []
            for obj in snapshot_objects:
                H, W = s.shape
                
                bbox = obj.bbox
                cx = ((bbox[0] + bbox[2]) / 2.) / W
                cy = ((bbox[1] + bbox[3]) / 2.) / H
                w = (bbox[2] - bbox[0]) / W
                h = (bbox[3] - bbox[1]) / H

                yolo_labels.append(f"{obj.id-1} {cx} {cy} {w} {h}") # YOLO classes start at 0

            # Save img and labels
            torchvision.utils.save_image(s.rgb, os.path.join(output_imgs_dir, f"{imgs_counter:06d}.jpg"))

            with open(os.path.join(output_labels_dir, f"{imgs_counter:06d}.txt"), "w") as f:
                f.write("\n".join(yolo_labels))

            imgs_counter += 1

    print(f"Converted {imgs_counter} images.")