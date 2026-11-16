# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import torch
import cv2 as cv
import numpy as np

from torch import Tensor
from abc import abstractmethod
from collections import defaultdict
from typing import List, Dict, Any

from dense_objects_nets.evaluation.don_evaluator import DonEvaluator, Metric, EvaluationResults
from dense_objects_nets.network.don_backbone import DonNetwork
from dense_objects_nets.utils.don_snapshot import DonSnapshot
from dense_objects_nets.utils import visualization_utils
from dense_objects_nets.evaluation import metrics


class DonMatchesEvaluator(DonEvaluator):
    """
    Abstract class for a Dense Object Nets evaluator object.
    """

    def __init__(self, config: Dict[str, Any]):
        self.plot_object_only_ = config["plot_object_only"]
        self.num_evaluation_matches_ = config["num_evaluation_matches"]    
        self.num_visualized_matches_ = config["num_evaluation_visualized_matches"]

    def evaluate(self, model: DonNetwork, samples: List[Dict[str, Any]]) -> Dict[EvaluationResults, Any]:
        """
        Evaluate the given input model on the given evaluation data samples returning some metrics
        """
        out_images = []
        metrics_values = defaultdict(lambda: [])

        for data in samples:
            s1 = DonSnapshot.from_dict(data["s1"])
            s2 = DonSnapshot.from_dict(data["s2"])
            shared_objects_ids = data["shared_objects"] # (objects_num, )
            objects_pixels_matches = data["objects_pixels_matches"] # (objects_num, num_sampled_points, 4)
            objects_dimensions = data["objects_dimensions"]  # (objects_num, 3)

            # Move data to the model device.
            s1.to(model.device, non_blocking=True)
            s2.to(model.device, non_blocking=True)
            objects_pixels_matches = objects_pixels_matches.to(model.device, non_blocking=True)

            # Get the model prediction on the two (augmented) images.
            with torch.no_grad():
                img1_pred, img2_pred = model.forward(torch.stack([s1.rgb, s2.rgb], dim=0)) 

            real_matches, predicted_matches = [], []
            for obj_id, pixels_matches, obj_dimensions in zip(shared_objects_ids, objects_pixels_matches, objects_dimensions):
                # Remove padding from objects pixels (if any) and skip if empty.
                pixels_matches = pixels_matches[pixels_matches[:, 0] >= 0]
                if len(pixels_matches) == 0:
                    continue

                # Clip the object pixels matches to the first 'num_evaluation_matches' matches.
                pixels_matches = pixels_matches[: self.num_evaluation_matches_]

                # Compute the closest descriptor in img2 for each match descriptor in img1.
                descriptor_matches = self.find_descriptors_matches(
                    img1_pred, img2_pred, img1_pixels=pixels_matches[:, :2]
                )

                # Compute the evaluation metrics
                metrics_dict = self.compute_metrics(
                    obj_id,
                    s1,
                    s2,
                    img1_pred,
                    img2_pred,
                    real_matches=pixels_matches,
                    predicted_matches=descriptor_matches,
                    obj_dimensions=obj_dimensions,
                )

                # Save the metrics, both for the current object and overall.
                obj_name = s1.get_object_by_id(obj_id).name
                for metric, metric_value in metrics_dict.items():
                    metrics_values[metric].append(metric_value)

                    if metric.log_per_object:
                        m =  metric.rename(f"[{obj_name}]/{metric.name}]")
                        metrics_values[m].append(metric_value)

                # Save real and predicted matches for the current object.
                predicted_matches.append(descriptor_matches)
                real_matches.append(pixels_matches)

            # Create an image to visualize output descriptors and matches.
            # Use original RGB and masks images (pre color augmentations transforms) for the outputs visualization.
            out_img = self.visualize_predictions(
                s1,
                s2,
                img1_pred,
                img2_pred,
                shared_ids=shared_objects_ids,
                real_matches=real_matches,
                predicted_matches=predicted_matches,
            )
            out_images.append(out_img)

        return {
            EvaluationResults("Metrics"): metrics_values,
            EvaluationResults("Evaluation Images", EvaluationResults.Types.IMAGES): out_images
        }

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def compute_metrics(self, obj_id: int, s1: DonSnapshot, s2: DonSnapshot, img1_pred: Tensor, img2_pred: Tensor, real_matches: Tensor, predicted_matches: Tensor, obj_dimensions: Tensor) -> Dict[Metric, float]:
        """
        Using the model predictions on the input images computes a set of evaluation metrics using the real matches as reference.

        Computed metrics are:
         1. Average descriptors distance between pairs of img1-img2 descriptors for each predicted match pixel positions.
         2. Average descriptors distance between pairs of img1-img2 descriptors for each real match pixel positions.
         3. Average descriptors distance between img2 descriptors for real and predicted img2 matches pixel positions.
         
         4. Average 3D points distance between points associated with real and predicted img2 matches pixel positions.
         5. Average pixels distance between real and predicted img2 matches pixel positions.
         6. Average 3D points distance between points associated with real and predicted img2 matches pixel positions. Normalized by the objects dimension (in meters).
         7. Average pixels distance between real and predicted img2 matches pixel positions. Normalized by the objects mask dimension in img2 (in pixels).
         
         8. PCK@k metrics for values of k in [1, 100] for each pairs of img1-img2 predicted match pixel positions.
        """
        d1_real_matches, d2_real_matches = real_matches[:, :2], real_matches[:, 2:]
        d1_real = img1_pred[:, d1_real_matches[:, 1], d1_real_matches[:, 0]]
        d2_real = img2_pred[:, d2_real_matches[:, 1], d2_real_matches[:, 0]]

        d2_pred_matches = predicted_matches[:, 2:]
        d2_pred = img2_pred[:, d2_pred_matches[:, 1], d2_pred_matches[:, 0]]

        # 1. Compute the descriptors distance between each img1 match and its associated predicted match in img2.
        pred_matches_descriptors_distances = self.compute_descriptors_distances(
            d1=d1_real,
            d2=d2_pred,
        )

        # 2. Compute the descriptors distance between each img1 match and its associated real match in img2.
        real_matches_descriptors_distances = self.compute_descriptors_distances(
            d1=d1_real,
            d2=d2_real,
        )

        # 3. Compute the descriptors distance between each real img2 match and the corresponding predicted img2 match.
        img2_matches_descriptors_distances = self.compute_descriptors_distances(
            d1=d2_real,
            d2=d2_pred,
        )

        # 4. Compute the 3D points distance between 3D points associated with real img2 matches the corresponding predicted img2 matches.
        # Matches with an invalid distance values (matches composed by pixels with depth==0) are filtered out.
        matches_points_distances, valid_distances = metrics.compute_points_distances(
            depth1=s2.depth,
            depth2=s2.depth,
            T1=s2.T_c2w,
            T2=s2.T_c2w,
            K=s2.K,
            d1_matches=d2_real_matches,
            d2_matches=d2_pred_matches,
        )
        matches_points_distances = matches_points_distances[valid_distances]


        # 5. Compute the pixels distance between each real img2 match and the corresponding predicted img2 match.
        matches_pixels_distances = metrics.compute_pixels_distances(
            d1_matches=d2_real_matches, 
            d2_matches=d2_pred_matches
        )

        # 6. Compute the normalized points distances.
        object_dimension = obj_dimensions.max().cpu().item()
        normalized_matches_points_distances = matches_points_distances / object_dimension

        # 7. Compute the normalized pixels distances.
        object_mask = s2.get_object_by_id(obj_id).mask
        ys, xs = torch.where(object_mask)
        object_mask_height = (ys.max() - ys.min()).cpu().item()
        object_mask_width = (xs.max() - xs.min()).cpu().item()
        object_mask_dimension = max(object_mask_height, object_mask_width)
        normalized_matches_pixels_distances = matches_pixels_distances / object_mask_dimension

        # 8. Computes the area under the curve for the PCK@k metric evaluated on the pixels distances between each real
        # img2 match and the corresponding predicted img2 match.
        auc_pck, pck_values = metrics.compute_pck(
            d1_matches=d2_real_matches,
            d2_matches=d2_pred_matches
        )

        avg_pred_match_desc_dist = torch.mean(pred_matches_descriptors_distances).cpu().item()
        avg_real_match_desc_dist = torch.mean(real_matches_descriptors_distances).cpu().item()
        avg_img2_match_desc_dist = torch.mean(img2_matches_descriptors_distances).cpu().item()
        avg_match_points_dist = torch.mean(matches_points_distances).cpu().item()
        avg_match_pixel_dist = torch.mean(matches_pixels_distances).cpu().item()
        avg_match_points_dist_normalized = torch.mean(normalized_matches_points_distances).cpu().item()
        avg_match_pixel_dist_normalized = torch.mean(normalized_matches_pixels_distances).cpu().item()

        # Return the metrics dict.
        metrics_values = {
            Metric("Descriptors Metrics/Avg Pred Match Descriptors Distance cosine"): avg_pred_match_desc_dist,
            Metric("Descriptors Metrics/Avg Real Match Descriptors Distance cosine"): avg_real_match_desc_dist,
            Metric("Descriptors Metrics/Avg Img2 Match Descriptors Distance cosine"): avg_img2_match_desc_dist,
            Metric("Pixels Metrics/Avg Match Points Distance"): avg_match_points_dist,
            Metric("Pixels Metrics/Avg Match Pixels Distance"): avg_match_pixel_dist,
            Metric("Pixels Metrics/Avg Match Points Distance (Normalized)"): avg_match_points_dist_normalized,
            Metric("Pixels Metrics/Avg Match Pixels Distance (Normalized)"): avg_match_pixel_dist_normalized,
            Metric("Pixels Metrics/AUC PCK@k for k in [1, 100]"): auc_pck
        }

        for k, val in enumerate(pck_values):
            metrics_values[Metric(f"Pixels Metrics/PCK@{k+1}", log_per_object=False)] = val
        
        return metrics_values

    def visualize_predictions(self, s1: DonSnapshot, s2: DonSnapshot, img1_pred: Tensor, img2_pred: Tensor, shared_ids: List[int], real_matches: List[Tensor], predicted_matches: List[Tensor]) -> np.ndarray:
        """
        Renders an 'evaluation image' used to visually asses the quality of the predictions.
        Real matches, predicted matches and output descriptors are plotted together for both input images.
        """
        rgb_images = [s1.rgb, s2.rgb]
        predictions_images = [img1_pred, img2_pred]
        images_masks = [s1.mask, s2.mask] if self.plot_object_only_ else None
        out_img = visualization_utils.plot_predictions(
            rgb_images, predictions_images, images_masks,
        )

        for obj_id, r_matches, p_matches in zip(shared_ids, real_matches, predicted_matches):
            # Sample 'num_matches' random matches and distinct colors for plotting.
            r_matches = r_matches[: self.num_visualized_matches_].cpu().numpy()
            p_matches = p_matches[: self.num_visualized_matches_].cpu().numpy()
            matches_colors = visualization_utils.colors[
                : min(len(r_matches), self.num_visualized_matches_)
            ]

            # Draw the sampled matches over the "out_img" obtained by the vertical concatenation of "original", "descriptors" and "overlay" images.
            # Matches are shifted both vertically and horizontally to compensate for the concatenation.
            height_offset, width_offset = s1.shape
            for i in range(0, 3):  # First iteration draws matches for the "original" img, then "descriptors" and finally for the "overlay" one.
                shifted_r_matches = r_matches + [
                    [0, i * height_offset, width_offset, i * height_offset]
                ]
                shifted_p_matches = p_matches + [
                    [0, i * height_offset, width_offset, i * height_offset]
                ]

                # Draw the real matches.
                out_img = visualization_utils.draw_matches(
                    out_img,
                    img1_pixels=shifted_r_matches[:, :2],
                    img2_pixels=shifted_r_matches[:, 2:],
                    matches_colors=matches_colors,
                    marker=cv.MARKER_CROSS,
                )
                # Draw the predicted_matches.
                out_img = visualization_utils.draw_matches(
                    out_img,
                    img1_pixels=shifted_p_matches[:, :2],
                    img2_pixels=shifted_p_matches[:, 2:],
                    matches_colors=matches_colors,
                    marker=cv.MARKER_STAR,
                )

        return out_img
