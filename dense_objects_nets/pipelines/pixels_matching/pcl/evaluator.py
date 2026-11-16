# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

from torch import Tensor

from dense_objects_nets.evaluation import metrics
from dense_objects_nets.pipelines.pixels_matching.don_matches_evaluator import DonMatchesEvaluator
from dense_objects_nets.utils.descriptors_matching_utils import find_descriptors_matches_L2

class PclEvaluator(DonMatchesEvaluator):

    def find_descriptors_matches(self, img1_pred: Tensor, img2_pred: Tensor, img1_pixels: Tensor) -> Tensor:
        """
        Find descriptors matches between the two input predictions images.
        Pairs are returned as a tensor of predicted matches of shape Nx4.  
        
        Args:
        - img1_pred (`torch.tensor`): tensor of shape DHW containing the model descriptors prediction on the first image.
        - img2_pred (`torch.tensor`): tensor of shape DHW containing the model descriptors prediction on the second image.
        - img1_pixels (`torch.tensor`): tensor of shape Nx2 containing the pixels positions of the img1 descriptors that we want to match.

        Returns:
        - `torch.tensor`: tensor containing the model predicted pixels matches for the current pair of images.
        """
        return find_descriptors_matches_L2(img1_pred, img2_pred, img1_pixels)
    
    def compute_descriptors_distances(self, d1: Tensor, d2: Tensor) -> Tensor:
        """
        Computes the distances between each pair of descriptors in d1 and d2. 
        Overrides depending on the specific distance metric used and the descriptors type.

        Args:
        - d1 (`torch.tensor`): tensor of shape DxN containing N descriptors of dimension D.
        - d2 (`torch.tensor`): tensor of shape DxN containing N descriptors of dimension D.

        Returns:
        - `torch.tensor`: tensor of shape N containing the L2 distances between each pair of descriptors in d1 and d2.
        """
        return metrics.compute_descriptors_distances_L2(d1, d2)