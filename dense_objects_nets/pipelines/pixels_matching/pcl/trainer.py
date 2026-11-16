# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import torch
import torch.nn.functional as F

from typing import Any, Dict, List
from itertools import combinations
from dense_objects_nets.pipelines.pixels_matching.don_matches_trainer import DonMatchesTrainer


class PclTrainer(DonMatchesTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.loss_margin = config["loss_margin"]
        self.scale_by_hard_negatives = config["scale_by_hard_negatives"]

    def get_pixels_matches_loss(self, descriptors: torch.Tensor, pairs_positions: torch.Tensor):
        # Compute the distance for the descriptors
        desc_dist = torch.cdist(descriptors, descriptors, p=2) # (M, M)

        # Matches distances
        matches_dist = desc_dist[pairs_positions[:, 0], pairs_positions[:, 1]]  # (2*num_sampled_points)

        # Non distances ignoring the positive similarities, the diagonal (self-similarity) and the lower part of the matrix (similarities w.r.t. the background points)
        desc_dist = desc_dist.masked_fill(torch.eye(desc_dist.size(0), device=desc_dist.device).bool(), 0)
        desc_dist[pairs_positions[:, 0], pairs_positions[:, 1]] = 0
        non_matches_dist = desc_dist[:len(pairs_positions),]  # (2*num_sampled_points, M)            

        # Matches loss
        matches_loss = matches_dist.pow(2).sum() / len(matches_dist)
        
        # Non-matches loss
        non_matches_dist = torch.clamp(self.loss_margin - non_matches_dist, min=0).pow(2)
        
        if self.scale_by_hard_negatives:
            num_non_matches = torch.count_nonzero(non_matches_dist) # Total number of non-matches that are not hard negatives (i.e. distances that are less than the margin)
        else:
            num_non_matches = non_matches_dist.numel() - (2 * non_matches_dist.size(1)) # Total number of non-matches. Each row has 2 cell thats are NOT non-matches (self-similarity and match itself)
            
        non_matches_loss = non_matches_dist.sum()
        if num_non_matches > 0: non_matches_loss /= num_non_matches

        return matches_loss + non_matches_loss
        
    def get_centers_loss(self, object_instances_centers: Dict[int, List[torch.Tensor]]) -> torch.Tensor:
        # Concatenate all centroids and compute the pairs positions
        current_index = 0
        centers_descriptors = []
        centers_pairs_positions = []
        for obj_id, centers in object_instances_centers.items():
            centers_descriptors.extend(centers)

            # Get all combinations of center pairs for the current object
            num_centers = len(centers)
            for i, j in combinations(range(num_centers), 2):
                centers_pairs_positions.append([current_index + i, current_index + j])
                centers_pairs_positions.append([current_index + j, current_index + i]) # Consider both (i, j) and (j, i)

            current_index += num_centers

        centers_descriptors = torch.stack(centers_descriptors, dim=0)  # (total_num_centers, D)
        centers_pairs_positions = torch.tensor(centers_pairs_positions, dtype=torch.int, device=centers_descriptors.device)  # (2*num_center_pairs, 2)

        # Compute the L2 distance for each center pair
        centers_dist = torch.cdist(centers_descriptors, centers_descriptors, p=2) # (M, M)
        pos_dist = centers_dist[centers_pairs_positions[:, 0], centers_pairs_positions[:, 1]]  # (2*num_center_pairs)

        centers_loss = pos_dist.pow(2).mean()

        return centers_loss