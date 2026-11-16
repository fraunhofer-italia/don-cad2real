# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import os
import yaml
import pytz
import torch
import open3d as o3d

from datetime import datetime
from typing import Tuple, Dict, List, Any

from dense_objects_nets.utils.don_snapshot import DonSnapshot, DonSnapshotInfo


def create_directory(base_path : str, dir_name : str) -> str:
    """
    Create a directory with the given name and returns its path.
    
    Args:
    - base_path (str): base path where the directory will be created.
    - dir_name (str): name of the directory.

    Returns:
    - (str): path to the created directory.  
    """
    dir_path = os.path.expandvars(os.path.join(base_path, dir_name))
    os.makedirs(dir_path, exist_ok=True)
    
    return dir_path

def create_timestamped_directory(base_path : str) -> str:
    """
    Create a directory named after the current timestamp and returns its path.
    
    Args:
    - base_path (str): base path where the directory will be created.

    Returns:
    - (str): path to the created directory.  
    """
    time_stamp = datetime.now(pytz.timezone("Europe/Rome"))
    dir_name = time_stamp.strftime("%Y-%m-%d-%H-%M-%S")

    return create_directory(base_path, dir_name)

def load_config_file(filename : str) -> Dict[Any, Any]:
    """
    Load a config dict from the input yaml file.

    Args:
    - filename (str): path to the yaml file containing the config parameters.

    Returns:
    - (dict): dict containing the content of the loaded yaml file.
    """
    with open(os.path.expandvars(filename)) as file:
        config = yaml.safe_load(file)
    return config

def load_triangle_mesh(name : str, path : str) -> o3d.t.geometry.TriangleMesh:
    """
    Load the Open3D triangle mesh with the given name at the given path.
   
    Args:
    - name (str): name of the input file.
    - path (str): path to the directory containing the input file.
    """
    filename = os.path.expandvars(os.path.join(path, name))
    return o3d.io.read_triangle_mesh(filename)

def load_points_cloud(name : str, path : str) -> o3d.t.geometry.PointCloud:
    """
    Load the Open3D points cloud with the given name at the given path.
   
    Args:
    - name (str): name of the input file.
    - path (str): path to the directory containing the input file.
    """
    filename = os.path.expandvars(os.path.join(path, name))
    return o3d.t.io.read_point_cloud(filename)

def load_camera_info(scene_base_path : str, device : torch.DeviceObjType = torch.device("cpu")) -> Tuple[torch.Tensor, int, int]:
    """
    Load the camera parameters from the `camera_info.yml` file inside the scene base directory.

    Args:
    - scene_base_path (str): base path to the scene directory.
    - device (torch.device, optional): device where the output tensor will be loaded. Defaults to `torch.device("cpu")`.

    Returns:
    - `torch.tensor`: K the 3x3 camera matrix of the camera.
    - `int`: the width of the images taken by the camera.
    - `int`: the height of the images taken by the camera.
    """
    camera_info = load_config_file(os.path.join(scene_base_path, "camera_info.yml"))
            
    camera_matrix = torch.tensor(camera_info["camera_info"]["K"], dtype=torch.float32, device=device).reshape(3, 3)

    width = camera_info["camera_info"]["width"]
    height = camera_info["camera_info"]["height"]

    return camera_matrix, width, height

def load_snapshots_info(scene_base_path : str, snapshots_filename : str = "snapshots.yml") -> List['DonSnapshotInfo']:
    """
    Load the snapshots info from the `snapshots.yml` file inside the scene base directory.

    Args:
    - scene_base_path (str): base path to the scene directory.
    - snapshots_filename (str, optional): name of the file containing the snapshots. Defaults to `snapshots.yml`.

    Returns:
    - (list): list containing the loaded snapshots info.
    """
    snapshots_data = load_config_file(os.path.join(scene_base_path, snapshots_filename))

    snapshots_info = [
        DonSnapshotInfo(scene_path=scene_base_path, dict=snapshot_dict, objects_ids=snapshot_dict["objects_ids"])
        for snapshot_id, snapshot_dict in snapshots_data.items()
    ]
    
    return snapshots_info

def load_snapshots(scene_base_path : str, snapshots_filename : str = "snapshots.yml", device : torch.DeviceObjType = torch.device("cpu")) -> List['DonSnapshot']:
    """
    Load all the snapshots in the scene base directory.

    Args:
    - scene_base_path (str): base path to the scene directory.
    - snapshots_filename (str, optional): name of the file containing the snapshots. Defaults to `snapshots.yml`.
    - device (torch.device, optional): device where the output tensors will be loaded. Defaults to `torch.device("cpu")`.

    Returns:
    - (dict): dict containing the loaded snapshots indexed by the corresponding ids.
    """
    snapshots_data = load_config_file(os.path.join(scene_base_path, snapshots_filename))

    snapshots = [
        DonSnapshot.from_config_dict(scene_base_path, snapshot_dict, device=device)
        for snapshot_id, snapshot_dict in snapshots_data.items()
    ]
    
    return snapshots
    
def save_dict(dict : Dict[Any, Any], name : str, path : str):
    """
    Save the input dict at the given path.

    Args:
    - dict (dict): dictionary that we want to save.
    - name (str): name of the output file.
    - path (str): path to the destination directory.
    """
    with open(os.path.expandvars(os.path.join(path, f"{name}")), "w") as file:
            yaml.dump(dict, file, default_flow_style=False)

def save_triangle_mesh(mesh : o3d.t.geometry.TriangleMesh, name : str, path : str):
    """
    Save the input Open3D triangle mesh at the given directory with the given name.
   
    Args:
    - mesh (open3d.t.geometry.TriangleMesh): the Open3D triangle mesh that we want to save.
    - name (str): name of the output file.
    - path (str): path to the destination directory.
    """
    filename = os.path.expandvars(os.path.join(path, name))
    o3d.io.write_triangle_mesh(filename, mesh=mesh, write_vertex_normals=False, print_progress=True)

def save_points_cloud(pcd : o3d.t.geometry.PointCloud, name : str, path : str):
    """
    Save the input Open3D points cloud at the given directory with the given name.

    Args:
    - pcd (open3d.t.geometry.PointCloud): the Open3D points cloud that we want to save.
    - name (str): name of the output file.
    - path (str): path to the destination directory.
    """
    filename = os.path.expandvars(os.path.join(path, name))
    o3d.io.write_point_cloud(filename, pointcloud=pcd, print_progress=True)