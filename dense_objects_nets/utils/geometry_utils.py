# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import torch
import numpy as np

from scipy.spatial.transform import Rotation
from typing import Dict, List

def add_batch_dimension(data : Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    for key in data: data[key] = data[key].unsqueeze(0)
    return data

def rotation_translation_to_transform(rquat : List[float], tvec : List[float], device : torch.DeviceObjType = torch.device("cpu")) -> torch.Tensor:
    T = np.zeros((4, 4))
    T[:3, :3] = Rotation.from_quat(rquat).as_matrix()
    T[:3, 3] = tvec
    T[3, 3] = 1

    T = torch.from_numpy(T).float()
    T = T.to(device)

    return T

def to_3D(pixels_2d  : torch.Tensor, depth : torch.Tensor, K : torch.Tensor, T_cam2world : torch.Tensor) -> torch.Tensor:
    """
    Transform the input tensor of pixel 2D coordinates in image frame to the corresponding 3D coordinates in world frame.

    Args:
    - pixels_2d (`torch.tensor`): tensor containing the coordinates of the 2D pixels. Expected shape is 'Nx2' where 'N'
    is the number of pixels (Each row is a pixel expressed as [u, v]).
    - depth (`torch.tensor`): tensor containing the camera-pixels distances expressed in meters.
    - K (`torch.tensor`): 3x3 camera matrix.
    - T_cam2world (`torch.tensor`): 4x4 transformation matrix defining the transformation from camera to world frame.

    Returns:
    - `torch.tensor`: tensor of shape '4xN' containing the 3D world position w.r.t the fixed world frame. (Each column
    is a point expressed in homogeneous coordinates [x, y, z, 1])
    """
    fx = K[0, 0]
    fy = K[1, 1]

    cx = K[0, 2]
    cy = K[1, 2]
    
    # Get two tensors containing the width and height coordinates of the input pixels
    u = pixels_2d[:, 0]
    v = pixels_2d[:, 1]
    
    # Get three tensors containing the [x, y, z] positions of all points with respect to the camera frame
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    # Convert them to world (fixed) frame
    points_3d = torch.stack([x, y, z, torch.ones(x.size(), device=x.device)], dim=1)
    points_3d = points_3d.transpose(0, 1) # from Nx4 to 4xN
    points_3d = torch.matmul(T_cam2world, points_3d) # 4x4 @ 4xN = 4xN
    
    return points_3d

def to_2D(points_3d : torch.Tensor, K : torch.Tensor, T_world2cam : torch.Tensor) -> torch.Tensor:
    """
    Transform the input tensor of points 3D coordinates in world frame to the corresponding pixel 2D coordinates in image frame.
    
    Args:
    - points_3d (`torch.tensor`): tensor containing the coordinates of the 3D points w.r.t the world fixed frame. Expected 
    shape is '4xN' where 'N' is the number of points (Each column is a point expressed in homogeneous coordinates [x, y, z, 1])
    - K (`torch.tensor`): 3x3 camera matrix.
    - T_world2cam (`torch.tensor`): 4x4 transformation matrix defining the transformation from world to camera frame.

    Returns:
    - `torch.tensor`: tensor of shape 'Nx2' containing the 2D image frame coordinates [u, v] of the input pixel 3D points.
    """
    # 3D to 2D homogeneous coordinates projection
    pixel_2d = torch.matmul(K, torch.matmul(T_world2cam[:3, :], points_3d)) # 3x3 @ (3x4 @ 4xN) = 3x3 @ 3xN = 3xN
    
    # Convert to cartesian coordinates and round to int 
    pixel_2d = pixel_2d[:2, :] / pixel_2d[2, :]
    pixel_2d = torch.round(pixel_2d).int()
        
    # Transpose to 2xN to Nx2 for convenience
    pixel_2d = pixel_2d.transpose(0, 1)
    
    return pixel_2d

def transform_pts_Rt(pts : torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Applies a rigid transformation to 3D points.
    """
    assert (pts.shape[1] == 3)

    pts_t = R @ pts.T + t.reshape((3, 1))
    return pts_t.T

def project_points(pts: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Projects 3D points onto a 2D plane using the camera matrix K.
    Points must be in camera coordinates.
    """
    if pts.ndim == 1:
        pts = pts.reshape((1, -1))

    assert (pts.shape[1] == 3)

    p_proj = (K @ pts.T).T  # (n_points, 3)
    p_proj = p_proj[:, :2] / p_proj[:, 2:3]  # Normalize by the z coordinate

    if p_proj.shape[0] == 1:
        p_proj = p_proj.squeeze(0)

    return p_proj