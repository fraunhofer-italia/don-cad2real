# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

"""
Scripts to create synthetic snapshots by composing multiple object views from BOP singleton real-object scans.
"""

import os
import json
import random
import argparse
import cv2 as cv
import numpy as np

from glob import glob
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
from scipy.spatial.transform import Rotation as R

from dense_objects_nets.utils import io_utils


def to_pose_matrix(rotation: np.ndarray, translation: np.ndarray, translation_scale: float = 0.001) -> np.ndarray:    
    pose = np.eye(4)
    pose[:3, :3] = np.array(rotation).reshape(3, 3)
    pose[:3, 3] = np.array(translation).reshape(3) * translation_scale # Convert translation to meters. BOP dataset uses mm.

    return pose

def pose_to_dict(pose: np.ndarray) -> Dict[str, Any]:
    assert pose.shape == (4, 4), "Pose must be a 4x4 matrix."
    
    translation = pose[:3, 3].tolist()
    rotation = R.from_matrix(pose[:3, :3]).as_quat().tolist()
    
    return {
        "translation": {
            "x": translation[0],
            "y": translation[1],
            "z": translation[2]
        },
        "rotation": {
            "x": rotation[0],
            "y": rotation[1],
            "z": rotation[2],
            "w": rotation[3]
        }
    }

def create_snapshot(obj_info: Dict[int, Any], sampled_obj_ids: List[int], out_size: List[int]) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    snapshot_dict = {
        "objects_ids": [],
        "objects_names": [],
        "object_to_world": [],
        "camera_matrix": [],
    }

    instance_id = 1
    out_rgb_img, out_depth_img, out_mask_img, out_instance_img = None, None, None, None
    for obj_id in sampled_obj_ids:
        camera_info, scene_info, scan = obj_info[obj_id]

        sampled_snapshot_id = random.choice(list(scene_info.keys()))
        obj_dict = scene_info[sampled_snapshot_id][0]  # Assuming single object per snapshot
        camera_dict = camera_info[sampled_snapshot_id]

        sampled_snapshot_id = int(sampled_snapshot_id)  # Ensure it's an integer for file naming

        # Load images
        rgb_img = cv.imread(os.path.join(scan, "rgb", f"{sampled_snapshot_id:06d}.png"), cv.IMREAD_COLOR)
        mask_img = cv.imread(os.path.join(scan, "mask_visib", f"{sampled_snapshot_id:06d}_000000.png"), cv.IMREAD_GRAYSCALE)
        depth_img = cv.imread(os.path.join(scan, "depth", f"{sampled_snapshot_id:06d}.png"), cv.IMREAD_UNCHANGED)
        depth_img = (depth_img * camera_dict["depth_scale"]).astype(np.uint16)

        if out_rgb_img is None:
            out_rgb_img = np.zeros((out_size[1], out_size[0], 3), dtype=np.uint8)
            out_depth_img = np.zeros((out_size[1], out_size[0]), dtype=np.uint16)
            out_mask_img = np.zeros((out_size[1], out_size[0]), dtype=np.uint8)
            out_instance_img = np.zeros((out_size[1], out_size[0]), dtype=np.uint8)
            
            obj_mask = mask_img > 0

            # Randomly place the object in the output image
            y = random.randint(0, out_size[1] - rgb_img.shape[0])
            x = random.randint(0, out_size[0] - rgb_img.shape[1])

            obj_mask_shifted = np.zeros_like(out_mask_img, dtype=bool)
            obj_mask_shifted[y:y+rgb_img.shape[0], x:x+rgb_img.shape[1]] = obj_mask

            out_rgb_img[obj_mask_shifted] = rgb_img[obj_mask]
            out_depth_img[obj_mask_shifted] = depth_img[obj_mask]
            out_mask_img[obj_mask_shifted] = obj_id
            out_instance_img[obj_mask_shifted] = instance_id

            # Update the camera matrix to account for the shift
            cam_K = np.array(camera_dict["cam_K"]).reshape(3, 3)
            cam_K[0, 2] += x
            cam_K[1, 2] += y

            instance_id += 1
            snapshot_dict["objects_ids"].append(obj_id)
            snapshot_dict["objects_names"].append(f"obj_{obj_id:02d}")
            snapshot_dict["camera_matrix"].append(cam_K.tolist())
            snapshot_dict["camera_to_world"] = pose_to_dict(np.eye(4))  # Here we assume T_o2w = I, so T_o2c = T_o2w
            snapshot_dict["object_to_world"].append(
                pose_to_dict(to_pose_matrix(obj_dict["cam_R_m2c"], obj_dict["cam_t_m2c"]))
            )
            continue

        # Get the current obj mask size
        obj_pixels = np.nonzero(mask_img)
        y_min, y_max = np.min(obj_pixels[0]), np.max(obj_pixels[0])
        x_min, x_max = np.min(obj_pixels[1]), np.max(obj_pixels[1])
        obj_height = y_max - y_min + 1
        obj_width = x_max - x_min + 1

        # Combine images ensuring a minimum object visibility
        successful_placement = False
        free_pixels = np.nonzero(out_mask_img == 0)
        anchor_pixels = random.choices(list(zip(free_pixels[0], free_pixels[1])), k=len(free_pixels[0]))
        for y, x in anchor_pixels:
            shifted_obj_height = min(obj_height, out_mask_img.shape[0] - y)
            shifted_obj_width = min(obj_width, out_mask_img.shape[1] - x)

            # Create pixels ranges for the shifted object and original mask
            y1, y2 = y, y + shifted_obj_height
            x1, x2 = x, x + shifted_obj_width

            oy1, oy2 = y_min, y_min + shifted_obj_height
            ox1, ox2 = x_min, x_min + shifted_obj_width

            shifted_mask = np.zeros_like(out_mask_img, dtype=bool)
            shifted_mask[y1:y2, x1:x2] = mask_img[oy1:oy2, ox1:ox2] > 0
            shifted_mask = shifted_mask & (out_mask_img == 0)

            if not (np.sum(shifted_mask) / np.sum(mask_img > 0) >= 0.3):  # Ensures at least 30% visibility
                continue

            # Update output images
            out_rgb_img[shifted_mask] = rgb_img[oy1:oy2, ox1:ox2][shifted_mask[y1:y2, x1:x2]]
            out_depth_img[shifted_mask] = depth_img[oy1:oy2, ox1:ox2][shifted_mask[y1:y2, x1:x2]]
            out_mask_img[shifted_mask] = obj_id
            out_instance_img[shifted_mask] = instance_id
            successful_placement = True
            instance_id += 1
            break
            
        if not successful_placement:
            print(f"Warning: Could not place object {obj_id} in the snapshot without excessive occlusion.")
            continue

        # Update the camera matrix to account for the shift
        cam_K = np.array(camera_dict["cam_K"]).reshape(3, 3)
        cam_K[0, 2] += x1 - x_min
        cam_K[1, 2] += y1 - y_min

        snapshot_dict["objects_ids"].append(obj_id)
        snapshot_dict["objects_names"].append(f"obj_{obj_id:02d}")
        snapshot_dict["camera_matrix"].append(cam_K.tolist())
        snapshot_dict["object_to_world"].append(
            pose_to_dict(to_pose_matrix(obj_dict["cam_R_m2c"], obj_dict["cam_t_m2c"])) # T_o2c = T_o2w
        )

    return snapshot_dict, out_rgb_img, out_depth_img, out_mask_img, out_instance_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BOP dataset to our custom format.")
    parser.add_argument("--in_path", "-in", type=str, required=True, help="Path to the original BOP dataset root directory.")
    parser.add_argument("--out_path", "-out", type=str, required=True, help="Path to save the converted dataset.")
    parser.add_argument("--num_object_snapshots", "-ns", type=int, required=True, help="Number of snapshots to generate per object.")
    parser.add_argument("--out_size", "-os", type=list, required=False, default=[720, 540], help="Output image size as [width, height].")
    parser.add_argument("--max_objects_per_snapshot", "-mo", type=int, default=10, help="Maximum number of objects per generated snapshot.")
    args = parser.parse_args()

    dataset_path = os.path.join(os.path.expandvars(args.in_path), "*")
    scans_folders = [scan for scan in sorted(glob(dataset_path)) if os.path.isdir(scan)]

    # Save info from each scan
    obj_info = {}
    for scan in scans_folders:
        with open(os.path.join(scan, "scene_camera.json"), "r") as f:
            camera_info = json.load(f)

        with open(os.path.join(scan, "scene_gt.json"), "r") as f:
            scene_info = json.load(f)

        obj_id = scene_info["0"][0]["obj_id"]  # Get object ID from the first snapshot

        obj_info[obj_id] = (camera_info, scene_info, scan)


    dataset_objects = list(obj_info.keys())
    for obj_id in tqdm(dataset_objects, desc="Processing objects"):
        # Create a scan folder for the object
        scan_out_path = io_utils.create_directory(args.out_path, f"{obj_id:06d}")

        # Create folders for images and masks
        images_dir = io_utils.create_directory(scan_out_path, "images")
        masks_dir = io_utils.create_directory(scan_out_path, "images_masks")

        snapshots = {}
        for sample_id in tqdm(range(args.num_object_snapshots), desc=f"Generating snapshots for 'obj_{obj_id:02d}'", leave=False):
            num_sample_objects = random.randint(1, args.max_objects_per_snapshot)
            sampled_obj_ids = random.choices(dataset_objects, k=num_sample_objects)

            sampled_obj_ids += [obj_id]  # Ensure the main object is included

            snapshot_dict, rgb_img, depth_img, mask_img, instance_img = create_snapshot(obj_info, sampled_obj_ids, args.out_size)

            # Save snapshot data
            snapshot_dict["rgb_img_filename"] = f"{sample_id:06d}_rgb.png"
            snapshot_dict["depth_img_filename"] = f"{sample_id:06d}_depth.png"
            snapshot_dict["mask_img_filename"] = f"{sample_id:06d}_mask.png"
            snapshot_dict["instances_mask_img_filename"] = f"{sample_id:06d}_instances_mask.png"
            cv.imwrite(os.path.join(images_dir, snapshot_dict["rgb_img_filename"]), rgb_img)
            cv.imwrite(os.path.join(images_dir, snapshot_dict["depth_img_filename"]), depth_img)
            cv.imwrite(os.path.join(masks_dir, snapshot_dict["mask_img_filename"]), mask_img)
            cv.imwrite(os.path.join(masks_dir, snapshot_dict["instances_mask_img_filename"]), instance_img)
            snapshots[sample_id] = snapshot_dict

        # Save scan info
        scan_objects = []
        for snapshot_dict in snapshots.values(): scan_objects += snapshot_dict["objects_names"]
        info_dict = {
            "snapshots_count": len(snapshots),
            "object_names": sorted(list(set(scan_objects))),
        }
        io_utils.save_dict(snapshots, "snapshots.yml", scan_out_path)
        io_utils.save_dict(info_dict, "scene_info.yml", scan_out_path)