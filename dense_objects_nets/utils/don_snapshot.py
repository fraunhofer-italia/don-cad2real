# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import os
import torch
import torchvision

import cv2 as cv
import numpy as np
import torch.nn.functional as F
from scipy.spatial import cKDTree

from dataclasses import dataclass
from torchvision.transforms import v2
from typing import Tuple, Dict, List, Any
from scipy.spatial.transform import Rotation as R

from dense_objects_nets.utils import geometry_utils


def load_pose(pose_dict : Dict[str, Any], device : torch.DeviceObjType = torch.device("cpu")) -> torch.Tensor:
    tvec = pose_dict["translation"]
    tvec = [tvec["x"], tvec["y"], tvec["z"]]
    rquat = pose_dict["rotation"]
    rquat = [rquat["x"], rquat["y"], rquat["z"], rquat["w"]]
    return geometry_utils.rotation_translation_to_transform(rquat, tvec, device)

def encode_strings_to_tensor(str_list):
    # Encode each string to a byte tensor
    encoded = [torch.tensor(list(s.encode('utf-8')), dtype=torch.uint8) for s in str_list]
    max_len = max(len(e) for e in encoded)
    padded = [F.pad(e, (0, max_len - len(e))) for e in encoded]  # pad to max length
    return torch.stack(padded)  # shape: (N, max_len)

def decode_tensor_to_strings(tensor):
    return [bytes(row.tolist()).decode('utf-8').rstrip('\x00') for row in tensor]

@dataclass
class DonSnapshotInfo():
    """
    Wrapper class to hold information about a snapshot without loading the actual data. Used to index the dataset.
    """
    objects_ids: List[int]
    scene_path: str
    dict: Dict[str, Any]

@dataclass
class SnapshotObject():
    instance_id: int
    id: int
    name: str
    T_o2w: torch.Tensor
    points: torch.Tensor  # (N, 3) tensor of 3D points in object coordinates
    pixels: torch.Tensor  # (N, 2) tensor of pixel coordinates in the snapshot image, in (x, y) format
    shape: Tuple[int, int]  # (H, W) shape of the object mask in the snapshot

    def match_points(self, other: 'SnapshotObject') -> Tuple[torch.Tensor, torch.Tensor]:
        if self.id != other.id:
            raise ValueError("Cannot match points between objects from different classes.")
        
        if len(self.points) == 0 or len(other.points) == 0:
            return torch.empty((0, 2), device=self.points.device), torch.empty((0,), device=self.points.device)
        
        kdtree = cKDTree(other.points.cpu().numpy())
        distances, associations = kdtree.query(self.points.cpu().numpy())
        pixels_matches = other.pixels[associations]

        return pixels_matches, torch.tensor(distances, device=self.points.device)

    def to(self, device: torch.device, non_blocking: bool = True) -> 'SnapshotObject':
        self.T_o2w = self.T_o2w.to(device, non_blocking=non_blocking)
        if self.points is not None: self.points = self.points.to(device, non_blocking=non_blocking)
        if self.pixels is not None: self.pixels = self.pixels.to(device, non_blocking=non_blocking)
        return self
    
    @property
    def mask(self) -> torch.Tensor:
        H, W = self.shape
        mask = torch.zeros((H, W), dtype=torch.bool, device=self.pixels.device)
        mask[self.pixels[:, 1], self.pixels[:, 0]] = True
        return mask
    
    @property
    def bbox(self) -> torch.Tensor:
        x_min, y_min = torch.min(self.pixels, dim=0)[0]
        x_max, y_max = torch.max(self.pixels, dim=0)[0]
        return torch.tensor([x_min, y_min, x_max, y_max], device=self.pixels.device)


@dataclass
class DonSnapshot():
    rgb: torch.Tensor
    depth: torch.Tensor
    mask: torch.Tensor
    instances_mask: torch.Tensor
    K: torch.Tensor
    T_c2w: torch.Tensor
    objects: List[SnapshotObject]

    def to(self, device: torch.device, non_blocking: bool = True) -> 'DonSnapshot':
        self.rgb = self.rgb.to(device, non_blocking=non_blocking)
        self.depth = self.depth.to(device, non_blocking=non_blocking)
        self.mask = self.mask.to(device, non_blocking=non_blocking)
        self.instances_mask = self.instances_mask.to(device, non_blocking=non_blocking)
        self.K = self.K.to(device, non_blocking=non_blocking) if self.K is not None else None
        self.T_c2w = self.T_c2w.to(device, non_blocking=non_blocking) if self.T_c2w is not None else None
        self.objects = [obj.to(device, non_blocking=non_blocking) for obj in self.objects]
        return self

    def get_object_by_instance_id(self, instance_id: int) -> SnapshotObject:
        obj_index = self.objects_instances_ids.index(instance_id)
        return self.objects[obj_index]

    def get_object_by_id(self, id: int) -> SnapshotObject:
        obj_index = self.objects_ids.index(id)
        return self.objects[obj_index]

    def get_objects_by_ids(self, ids: List[int]) -> List[SnapshotObject]:
        objects = [obj for obj in self.objects if obj.id in ids]
        objects.sort(key=lambda x: x.id)  # Sort by id for consistency

        return objects

    @staticmethod
    def from_config_dict(scene_base_path : str, snapshot_dict : Dict[str, Any], device : torch.DeviceObjType = torch.device("cpu")) -> 'DonSnapshot':
        scene_base_path = os.path.expandvars(scene_base_path)

        rgb_path = os.path.join(scene_base_path, "images", snapshot_dict['rgb_img_filename'])
        depth_path = os.path.join(scene_base_path, "images", snapshot_dict['depth_img_filename'])
        mask_path = os.path.join(scene_base_path, "images_masks", snapshot_dict['mask_img_filename'])
        instances_mask_path = os.path.join(scene_base_path, "images_masks", snapshot_dict['instances_mask_img_filename'])

        rgb = torchvision.io.read_image(rgb_path, mode=torchvision.io.ImageReadMode.RGB).to(device)
        rgb = v2.ToDtype(torch.float32, scale=True)(rgb)

        depth = cv.imread(depth_path, cv.IMREAD_UNCHANGED)
        depth = torch.from_numpy(depth).to(device).int()

        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        mask = torch.from_numpy(mask).to(device).int()

        instances_mask = cv.imread(instances_mask_path, cv.IMREAD_GRAYSCALE)
        instances_mask = torch.from_numpy(instances_mask).to(device).int()

        T_c2w = load_pose(snapshot_dict["camera_to_world"], device)

        # Load camera matrix (supports per-object camera matrix for dataset generated from snapshots stitching)
        camera_matrix = torch.tensor(snapshot_dict["camera_matrix"], dtype=torch.float32, device=device)
        if camera_matrix.ndim > 1:
            camera_matrix = camera_matrix.reshape(camera_matrix.shape[0], 3, 3)
        else:
            camera_matrix = camera_matrix.reshape(3, 3)
        
        # Load objects from the snapshot
        objects_ids = [int(id) for id in snapshot_dict["objects_ids"]]
        objects_names = snapshot_dict["objects_names"]
        object_poses = [load_pose(object_pose, device) for object_pose in snapshot_dict["object_to_world"]]

        objects = []
        for object_num, (id, name, T_o2w) in enumerate(zip(objects_ids, objects_names, object_poses)):
            instance_id = object_num + 1  # Instance ids are 1-indexed in the mask
            object_mask = (instances_mask == instance_id).bool()

            if not object_mask.any():
                continue # Skip objects that are not in the snapshot

            object_pixels = torch.nonzero(object_mask)
            object_pixels = torch.index_select(object_pixels, 1, torch.tensor([1, 0], device=object_pixels.device)) # (N, 2) tensor of pixels in (x, y) format

            # Select the correct camera matrix (supports per-object camera matrices)
            if camera_matrix.ndim == 3:
                K = camera_matrix[object_num]
            else:
                K = camera_matrix

            T_c2o = torch.linalg.inv(T_o2w) @ T_c2w
            pixels_depth = depth[object_pixels[:, 1], object_pixels[:, 0]] * 0.001 # Convert depth from mm to meters
            object_points = geometry_utils.to_3D(object_pixels, pixels_depth, K, T_c2o)[:3, :].T # (N, 3)

            objects.append(
                SnapshotObject(
                    instance_id=instance_id,
                    id=id,
                    name=name,
                    T_o2w=T_o2w,
                    points=object_points,
                    pixels=object_pixels,
                    shape=object_mask.shape[-2:]  # (H, W)
                )
            )
       
        return (
            DonSnapshot(
                rgb = rgb,
                depth = depth,
                mask = mask,
                instances_mask = instances_mask,
                K = camera_matrix,
                T_c2w = T_c2w,
                objects = objects,
            )
        )
    
    @staticmethod
    def from_dict(data: Dict[str, torch.Tensor]) -> "DonSnapshot":
        points_index = 0
        objects_points = []
        object_pixels = []
        for n_points in data["points_per_object"]:
            objects_points.append(data["objects_points"][points_index: points_index+n_points])
            object_pixels.append(data["objects_pixels"][points_index: points_index+n_points])
            points_index += n_points
        
        instances_ids = data["instance_ids"]
        objects_ids = data["objects_ids"]
        objects_poses = data["objects_poses"]
        objects_names = decode_tensor_to_strings(data["objects_names"])
        objects = [
            SnapshotObject(
                instance_id=instance_id,
                id=id,
                name=name,
                T_o2w=objects_poses[i],
                points=objects_points[i],
                pixels=object_pixels[i],
                shape=data["mask"].shape[-2:]
            )
            for i, (instance_id, id, name) in enumerate(zip(instances_ids, objects_ids, objects_names))
        ]

        return DonSnapshot(
            rgb = data["rgb"],
            depth = data["depth"],
            mask = data["mask"],
            instances_mask = data["instances_mask"],
            K = data["K"],
            T_c2w = data["T_c2w"],
            objects = objects
        )

    def to_dict(self) -> Dict[str, torch.Tensor]:        
        return {
            "rgb": self.rgb,
            "depth": self.depth,
            "mask": self.mask,
            "instances_mask": self.instances_mask,
            "K": self.K,
            "T_c2w": self.T_c2w,
            "instance_ids": torch.tensor([obj.instance_id for obj in self.objects]), # (N,) tensor of instance ids
            "objects_ids": torch.tensor([obj.id for obj in self.objects]), # (N,) tensor of object ids
            "objects_names": encode_strings_to_tensor([obj.name for obj in self.objects]),
            "objects_points": torch.cat([obj.points for obj in self.objects], dim=0), # (tot_num_points, 3)
            "objects_pixels": torch.cat([obj.pixels for obj in self.objects], dim=0), # (tot_num_points, 2)
            "objects_poses": torch.stack([obj.T_o2w for obj in self.objects]), # (N, 4, 4) tensor of object poses
            "points_per_object": torch.tensor([obj.points.shape[0] for obj in self.objects]) # (N,) used to index the two above tensors
        }
    
    @property
    def objects_instances_ids(self) -> List[int]:
        return [obj.instance_id for obj in self.objects]

    @property
    def objects_ids(self) -> List[int]:
        return [obj.id for obj in self.objects]

    @property
    def objects_names(self) -> List[str]:
        return [obj.name for obj in self.objects]

    @property
    def device(self) -> torch.DeviceObjType:
        return self.rgb.device
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.rgb.shape[-2:]
    
    @property
    def H(self) -> int:
        return self.shape[0]
    
    @property
    def W(self) -> int:
        return self.shape[1]
    