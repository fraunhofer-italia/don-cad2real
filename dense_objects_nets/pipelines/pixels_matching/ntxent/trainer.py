# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import torch
import torch.nn.functional as F

from typing import Any, Dict, List
from itertools import combinations
from dense_objects_nets.pipelines.pixels_matching.don_matches_trainer import DonMatchesTrainer


class NTXentTrainer(DonMatchesTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tau = config.get("temperature", 0.1)
        self.center_loss_tau = config.get("center_loss_temperature", self.tau)

    def get_pixels_matches_loss(self, descriptors: torch.Tensor, pairs_positions: torch.Tensor):
        return self.get_ntxent_loss(descriptors, pairs_positions, tau=self.tau)
    
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

        # Compute the NT-Xent loss for the centers
        centers_loss = self.get_ntxent_loss(centers_descriptors, centers_pairs_positions, tau=self.center_loss_tau)

        return centers_loss

    def get_ntxent_loss(self, descriptors: torch.Tensor, pairs_positions: torch.Tensor, tau) -> torch.Tensor:
        # Normalize the descriptors
        descriptors = F.normalize(descriptors, p=2, dim=1)  # (M, D)

        # Compute the similarity matrix for the descriptors
        desc_sim = torch.mm(descriptors, descriptors.T) # (M, M)
        desc_sim_exp = torch.exp(desc_sim / tau) # (M, M)
        
        # Remove self-similarities from the similarity matrix
        desc_sim_exp = desc_sim_exp.masked_fill(torch.eye(desc_sim_exp.size(0), device=desc_sim_exp.device).bool(), 0)
        
        # Positive similarities
        pos_sim = desc_sim_exp[pairs_positions[:, 0], pairs_positions[:, 1]]  # (2*num_pairs)

        # Negative similarities (sum over rows).
        neg_sim = desc_sim_exp[pairs_positions[:, 0], :].sum(dim=1)  # (2*num_pairs)

        # Compute the loss for the current object
        loss_per_pair = -torch.log(pos_sim / neg_sim)  # (2*num_pairs)
        loss = loss_per_pair.mean()
        
        return loss