# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import os
import random
import argparse

from tqdm import tqdm
from typing import Any, Dict
from dense_objects_nets.utils import io_utils

def filter_snapshots(snapshots_dict: Dict[str, Any], filter_snapshots_dict: Dict[str, Any] | None, max_skip: int):
    filter_keys = set(filter_snapshots_dict.keys()) if filter_snapshots_dict else set()
    valid_ids = sorted(list(set(snapshots_dict.keys()) - filter_keys))

    filtered_snapshots = {}
    index = random.randint(0, max_skip)
    while index < len(valid_ids) - 1:
        next_id = valid_ids[index]
        filtered_snapshots[next_id] = snapshots_dict[next_id]
        
        index += random.randint(1, max_skip)

    return filtered_snapshots

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter snapshots from one dataset.")
    parser.add_argument("--in_path", "-in", type=str, required=True, help="Path to the original dataset root directory.")
    parser.add_argument("--max_skip", "-ms", type=float, required=True, help="Maximum skip distance for filtering.")
    parser.add_argument("--filter_dataset", "-fd", type=str, required=False, default=None, help="Path to the dataset used as filter.")
    args = parser.parse_args()

    scans_folders = [
        os.path.join(args.in_path, scan) for scan in sorted(os.listdir(args.in_path))
    ]
   
    filter_scans_folders = [
        os.path.join(args.filter_dataset, scan) for scan in sorted(os.listdir(args.filter_dataset))
    ] if args.filter_dataset else [None] * len(scans_folders)

    for scan, filter_scan, in tqdm(zip(scans_folders, filter_scans_folders), desc="Filtering snapshots", total=len(scans_folders)):
        snapshots_dict = io_utils.load_config_file(os.path.join(scan, "snapshots.yml"))
        filter_snapshots_dict = io_utils.load_config_file(os.path.join(filter_scan, "snapshots.yml")) if filter_scan else None

        snapshot_dict_updated = filter_snapshots(snapshots_dict, filter_snapshots_dict, args.max_skip)
        io_utils.save_dict(snapshots_dict, "snapshots_original.yml", scan)
        io_utils.save_dict(snapshot_dict_updated, "snapshots.yml", scan)

        scan_info = io_utils.load_config_file(os.path.join(scan, "scene_info.yml"))
        scene_info_updated = scan_info.copy()
        scene_info_updated["snapshots_count"] = len(snapshot_dict_updated)
        io_utils.save_dict(scan_info, "scene_info_original.yml", scan)
        io_utils.save_dict(scene_info_updated, "scene_info.yml", scan)