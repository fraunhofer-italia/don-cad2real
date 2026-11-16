# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

"""
Convert BOP dataset to our custom format.

Input dataset format is expected to be consistent with: https://huggingface.co/bop-benchmark 
"""

import os
import json
import random
import argparse
import cv2 as cv
import numpy as np

from glob import glob
from tqdm import tqdm
from typing import Any, Dict, List
from scipy.spatial.transform import Rotation as R

from dense_objects_nets.utils import io_utils


def to_pose_matrix(rotation: np.ndarray, translation: np.ndarray, translation_scale: float = 0.001) -> np.ndarray:    
    pose = np.eye(4)
    pose[:3, :3] = np.array(rotation).reshape(3, 3)
    pose[:3, 3] = np.array(translation).reshape(3) * translation_scale # Convert translation to meters. BOP dataset uses mm.

    return pose

def sum_masks_no_overlaps(masks: List[np.ndarray]) -> np.ndarray:
    out_mask = np.zeros_like(masks[0], dtype=np.uint8)
    overlap_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        object_pixels = mask != 0

        # Check for overlapping pixels and update the overlap mask
        overlapping_pixels = object_pixels & (out_mask != 0)
        overlap_mask += overlapping_pixels
        
        # Update the output mask with the new mask, only fill empty pixels
        out_mask[(out_mask == 0) & object_pixels] = mask[(out_mask == 0) & object_pixels]

    # Remove overlapping pixels from the final mask
    out_mask[overlap_mask != 0] = 0

    return out_mask

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

def convert_scene_info(scene_info: Dict[str, Any], scene_ids: List[str]) -> Dict[str, Any]:
    scene_info_converted = {}
    for snapshot_id in scene_ids:
        snapshot_objects = scene_info[snapshot_id]
        object_ids = [
            obj_dict["obj_id"] for obj_dict in snapshot_objects
        ]

        object_poses = [
            to_pose_matrix(obj_dict["cam_R_m2c"], obj_dict["cam_t_m2c"]) #  T_o2c
            for obj_dict in snapshot_objects
        ]

        scene_info_converted[snapshot_id] = {
            "objects_ids": object_ids,
            "objects_poses": object_poses,
        }

    return scene_info_converted

def convert_camera_info(camera_info: Dict[str, Any], scene_ids: List[str]) -> Dict[str, Any]:
    camera_info_converted = {}
    for snapshot_id in scene_ids:
        scan_dict = camera_info[snapshot_id]
        K = scan_dict["cam_K"]

        if "cam_R_w2c" in scan_dict and "cam_t_w2c" in scan_dict:
            T_w2c = to_pose_matrix(scan_dict["cam_R_w2c"], scan_dict["cam_t_w2c"])
            T_c2w = np.linalg.inv(T_w2c)
        else:
            T_c2w = np.eye(4) # Default to origin if no camera pose is provided

        camera_info_converted[snapshot_id] = {
            "K": K,
            "T_c2w": T_c2w,
        }

    return camera_info_converted

def save_scan_info(scan: str, camera_info: Dict[str, Any], scene_info: Dict[str, Any], out_path: str) -> None:
    snapshots_dict = {}
    snapshot_ids = list(camera_info.keys())
    for snapshot_id in snapshot_ids:
        camera_dict = camera_info[snapshot_id]
        scene_dict = scene_info[snapshot_id]

        snapshot_id = int(snapshot_id)  # Ensure snapshot_id is an integer
        snapshots_dict[snapshot_id] = {
            "rgb_img_filename": f"{snapshot_id:06d}_rgb.png",
            "mask_img_filename": f"{snapshot_id:06d}_mask.png",
            "instances_mask_img_filename": f"{snapshot_id:06d}_instances_mask.png",
            "depth_img_filename": f"{snapshot_id:06d}_depth.png",
            
            "objects_ids": scene_dict["objects_ids"],
            "objects_names": [f"obj_{obj_id:02d}" for obj_id in scene_dict["objects_ids"]],
            
            "camera_matrix": camera_dict["K"],
            "camera_to_world": pose_to_dict(camera_dict["T_c2w"]),
            "object_to_world": [pose_to_dict(camera_dict["T_c2w"] @ T_o2c) for T_o2c in scene_dict["objects_poses"]],
        }

    scan_objects = []
    for snapshot_dict in snapshots_dict.values(): scan_objects += snapshot_dict["objects_names"]
    info_dict = {
        "snapshots_count": len(snapshots_dict),
        "object_names": sorted(list(set(scan_objects))),
    }
    
    scan_name = scan.rsplit("/", 1)[-1]
    out_dir = io_utils.create_directory(out_path, scan_name)
    io_utils.save_dict(snapshots_dict, "snapshots.yml", out_dir)
    io_utils.save_dict(info_dict, "scene_info.yml", out_dir)

    return snapshots_dict

def save_scan_images(scan: str, snapshots_info: Dict[str, Any], out_path: str, depth_scale: float = 0.1) -> None:
    scan_name = scan.rsplit("/", 1)[-1]
    out_dir = os.path.join(out_path, scan_name)

    images_dir = io_utils.create_directory(out_dir, "images")
    masks_dir = io_utils.create_directory(out_dir, "images_masks")
    for snapshot_num, snapshot_info in tqdm(snapshots_info.items(), desc=f"Converting images for {scan_name}", leave=False):
        rgb_img_filename = glob(os.path.join(scan, "rgb", f"{snapshot_num:06d}.*"))[0] # Wildcard to match any extension (.png, .jpg, ...)
        depth_img_filename = glob(os.path.join(scan, "depth", f"{snapshot_num:06d}.*"))[0]
        rgb_img = cv.imread(rgb_img_filename, cv.IMREAD_COLOR)
        depth_img = cv.imread(depth_img_filename, cv.IMREAD_UNCHANGED)
        depth_img = (depth_img * depth_scale).astype(np.uint16)

        rgb_img_path = os.path.join(images_dir, snapshot_info["rgb_img_filename"])
        depth_img_path = os.path.join(images_dir, snapshot_info["depth_img_filename"])
        cv.imwrite(rgb_img_path, rgb_img)
        cv.imwrite(depth_img_path, depth_img)
    
        # Save masks images merging them in a single image
        masks_imgs = []
        instances_masks_imgs = []
        for instance_id, obj_id in enumerate(snapshot_info["objects_ids"]):
            mask_img_filename = glob(os.path.join(scan, "mask_visib", f"{snapshot_num:06d}_{instance_id:06d}.*"))[0]
            mask_img = cv.imread(mask_img_filename, cv.IMREAD_GRAYSCALE)
            
            mask_img[mask_img > 0] = (obj_id)
            masks_imgs.append(mask_img)

            instance_mask_img = mask_img.copy()
            instance_mask_img[instance_mask_img > 0] = (instance_id + 1)
            instances_masks_imgs.append(instance_mask_img)

        # Merge all mask images by summing them and removing overlaps (if any)
        mask = sum_masks_no_overlaps(masks_imgs)
        instances_mask = sum_masks_no_overlaps(instances_masks_imgs)

        mask_img_path = os.path.join(masks_dir, snapshot_info["mask_img_filename"])
        cv.imwrite(mask_img_path, mask)
        instances_mask_img_path = os.path.join(masks_dir, snapshot_info["instances_mask_img_filename"])
        cv.imwrite(instances_mask_img_path, instances_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BOP dataset to our custom format.")
    parser.add_argument("--in_path", "-in", type=str, required=True, help="Path to the original BOP dataset root directory.")
    parser.add_argument("--out_path", "-out", type=str, required=True, help="Path to save the converted dataset.")
    parser.add_argument("--depth_scale", "-ds", type=float, required=True, help="Scale factor for depth images.")
    parser.add_argument("--num_samples", "-ns", type=int, required=False, help="Max number of samples per scene to process.")
    args = parser.parse_args()

    dataset_path = os.path.join(os.path.expandvars(args.in_path), "*")
    scans_folders = [scan for scan in sorted(glob(dataset_path)) if os.path.isdir(scan)]

    for scan in tqdm(scans_folders, desc=f"Processing scans in {args.in_path}"):
        with open(os.path.join(scan, "scene_camera.json"), "r") as f:
            camera_info = json.load(f)

        with open(os.path.join(scan, "scene_gt.json"), "r") as f:
            scene_info = json.load(f)

        scene_ids = list(scene_info.keys())
        if args.num_samples and args.num_samples < len(scene_ids):
            scene_ids = random.sample(scene_ids, args.num_samples)

        scene_info = convert_scene_info(scene_info, scene_ids)
        camera_info = convert_camera_info(camera_info, scene_ids)

        snapshots_info = save_scan_info(scan, camera_info, scene_info, args.out_path)
        save_scan_images(scan, snapshots_info, args.out_path, args.depth_scale)