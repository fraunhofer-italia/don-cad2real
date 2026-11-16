# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

import dense_objects_nets.utils.geometry_utils as geometry_utils


def compute_descriptors_distances_L2(d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
    """
    Computes the L2 descriptors distances for each of the N input matches pairs. 

    Args:
    - d1 (`torch.tensor`): tensor of shape DN containing the first set of model descriptors.
    - d2 (`torch.tensor`): tensor of shape DN containing the second set of model descriptors.

    Returns:
    - `torch.tensor`: (N, ) tensor containing the descriptors distances for each of the N input matches pairs.
    """
    descriptors_distances = torch.linalg.vector_norm(d1 - d2, dim=0)
        
    return descriptors_distances

def compute_descriptors_distances_cosine(d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine descriptors distances for each of the N input matches pairs. 

    Args:
    - d1 (`torch.tensor`): tensor of shape DN containing the first set of model descriptors.
    - d2 (`torch.tensor`): tensor of shape DN containing the second set of model descriptors.

    Returns:
    - `torch.tensor`: (N, ) tensor containing the descriptors distances for each of the N input matches pairs.
    """
    # Normalize the descriptors
    d1_norm = F.normalize(d1, p=2, dim=0)
    d2_norm = F.normalize(d2, p=2, dim=0)

    # Compute cosine similarity
    descriptors_distances = F.cosine_similarity(d1_norm, d2_norm, dim=0)

    return descriptors_distances


def compute_points_distances(depth1: torch.Tensor, depth2: torch.Tensor, T1: torch.Tensor, T2: torch.Tensor, K: torch.Tensor, d1_matches: torch.Tensor, d2_matches: torch.Tensor) -> torch.Tensor:
    """
    Computes the 3D points distances (L2 distance) for the points associated with each of the N input matches pairs.

    Distance value for matches whose pixels are associated with an invalid depth value (depth == 0) are set to `torch.inf`.
    Returns a tensor containing the valid distances indexes.

    Args:
    - depth1 (`torch.tensor`): depth image associated with the first set of matches.
    - depth2 (`torch.tensor`): depth image associated with the second set of matches.
    - T1 (`torch.tensor`): 4x4 transformation matrix defining the transformation from camera to world frame for the first set of matches.
    - T2 (`torch.tensor`): 4x4 transformation matrix defining the transformation from camera to world frame for the second set of matches.
    - K (`torch.tensor`): 3x3 camera matrix for the scene.
    - d1_matches (`torch.tensor`): tensor of shape Nx2 containing the first set of pixels matches.
    - d2_matches (`torch.tensor`): tensor of shape Nx2 containing the second set of pixels matches.

    Returns:
    - `tuple`: tuple containing two `torch.tensor`. The fist containing the 3D points distances for each of the N input
    matches pairs. The second the containing the indexes of the valid distance values in the first tensor.
    """
    # Depth values associated with the input matches (in meters).
    d1 = depth1[d1_matches[:, 1], d1_matches[:, 0]]*0.001
    d2 = depth2[d2_matches[:, 1], d2_matches[:, 0]]*0.001

    # Filter out matches with invalid depth values.
    valid_depth_indexes = (d1 > 0) * (d2 > 0)
    valid_d1 = d1[valid_depth_indexes]
    valid_d2 = d2[valid_depth_indexes]
    valid_d1_matches = d1_matches[valid_depth_indexes]
    valid_d2_matches = d2_matches[valid_depth_indexes]

    # Get 3D points associated with the matches with valid depth values.
    points1 = geometry_utils.to_3D(valid_d1_matches, valid_d1, K=K, T_cam2world=T1)
    points2 = geometry_utils.to_3D(valid_d2_matches, valid_d2, K=K, T_cam2world=T2)

    # 3D points distance between the above points.
    valid_matches_points_distances = torch.linalg.vector_norm((points1 - points2).float(), dim=0)

    # Output vector, matches with invalid depth values have distance set to infinity.
    matches_points_distances = torch.full((len(d1),), fill_value=torch.inf, device=d1.device)
    matches_points_distances[valid_depth_indexes] = valid_matches_points_distances

    return matches_points_distances, valid_depth_indexes


def compute_pixels_distances(d1_matches: torch.Tensor, d2_matches: torch.Tensor) -> torch.Tensor:
    """
    Computes the pixels distances for each of the N input matches pairs. 

    Args:
    - d1_matches (`torch.tensor`): tensor of shape Nx2 containing the first set of pixels matches.
    - d2_matches (`torch.tensor`): tensor of shape Nx2 containing the second set of pixels matches.

    Returns:
    - `torch.tensor`: tensor containing the pixels distances for each of the N input matches pairs.
    """
    matches_pixels_distances = torch.linalg.vector_norm((d1_matches - d2_matches).float(), dim=1)

    return matches_pixels_distances


def compute_pck(d1_matches: torch.Tensor, d2_matches: torch.Tensor, max_k: int = 100) -> Tuple[float, List[float]]:
    """
    Computes the area under the curve (AUC) for the PCK@k metric for values of 'k' ranging from [1, `max_k`] where 'k'
    is the maximum allowed error, measured in pixels, for a prediction to still be counted as correct.

    Args:
    - d1_matches (`torch.tensor`): tensor of shape Nx2 containing the first set of pixels matches.
    - d2_matches (`torch.tensor`): tensor of shape Nx2 containing the second set of pixels matches.
    - max_k (`int`, optional): max value of 'k' used to evaluate the PCK metric. Defaults to 100.

    Returns:
    - Tuple[float, List[float]]: A tuple containing the AUC for the PCK@k metric and a list of PCK values for each k.
    """
    # Computes matches pixels distance.
    matches_pixels_distances = compute_pixels_distances(d1_matches, d2_matches)

    # Computes PCK@k  for k in [1, 'max_k']
    pck_values = []
    for k in range(1, max_k + 1):
        valid_matches_distances = matches_pixels_distances[matches_pixels_distances <= k]
        pck = valid_matches_distances.size(0) / matches_pixels_distances.size(0)

        pck_values.append(pck)

    auc = np.trapz(y=pck_values, dx=1)

    return auc, pck_values
