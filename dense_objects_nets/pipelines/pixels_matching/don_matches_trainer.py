# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import os
import json
import torch

from abc import abstractmethod
from typing import Dict, Any, List
from collections import defaultdict
from dense_objects_nets.training.don_trainer import DonTrainer
from dense_objects_nets.network.don_backbone import DonNetwork

from .utils import debug


class DonMatchesTrainer(DonTrainer):
    def __init__(self, config: Dict[str, Any]):
        # Whether to use the centers loss or not
        self.compute_centers_loss = config.get("compute_centers_loss", True)

        # Whether to use the DANN loss or not
        self.compute_dann_loss = config.get("compute_dann_loss", False)

        # Loss scales values
        self.pixels_matches_loss_scale = config.get("pixels_matches_loss_scale", 1.0)
        self.center_loss_scale = config.get("center_loss_scale", 1.0)
        self.dann_loss_scale = config.get("dann_loss_scale", 1.0)
        self.dann_target_warmup_iterations = config.get("dann_warmup_iterations", 1)

        self.debug = config.get("debug", False)

        # Dicts to save some trainer metrics
        self.iteration = 0
        self.objects_occurrences_distribution = defaultdict(lambda: [])
        self.objects_matched_pixels_count = defaultdict(int)

    @abstractmethod
    def get_pixels_matches_loss(self, descriptors: torch.Tensor, pairs_positions: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_centers_loss(self, object_instances_centers: Dict[int, List[torch.Tensor]]) -> torch.Tensor:
        pass
    
    def get_dann_loss(self, model: DonNetwork, batch_dann_descriptors: List[torch.Tensor], batch_dann_domain_labels: List[torch.Tensor]) -> torch.Tensor:
        dann_descriptors = torch.stack(batch_dann_descriptors)
        dann_domain_labels = torch.stack(batch_dann_domain_labels)

        # Shuffle the descriptors and labels
        perm = torch.randperm(dann_descriptors.size(0))
        dann_descriptors = dann_descriptors[perm]
        dann_domain_labels = dann_domain_labels[perm]
           
        # Compute the training progress p with respect to the target warmup iterations
        p = self.iteration / self.dann_target_warmup_iterations
        lambd = model.get_lambda(p)

        # First update the domain classifier, without affecting the DON backbone
        model.dann_update(dann_descriptors, dann_domain_labels, lambd=lambd)

        # Then compute the DANN loss to backpropagate through the DON backbone
        dann_loss = model.get_dann_loss(dann_descriptors, dann_domain_labels, lambd=lambd)
        
        return dann_loss
    
    def run(self, model: DonNetwork, data: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        self.iteration += 1

        # Process each sample in the batch separately to avoid memory issues and problems with data indexing
        batch_pixels_loss = 0.0
        batch_objects_instances_centers = defaultdict(list)
        batch_domain_classifier_descriptors, batch_domain_classifier_labels = [], []
        for sample in data:
            # Move all tensors to the device in use
            for key in sample:
                sample[key] = sample[key].to(model.device, non_blocking=True)

            train_images = sample["train_images"] # (3, 3, H, W)
            objects_pixels_matches = sample["objects_pixels_matches"] # (objects_num, num_sampled_points, 4)
            background_pixels =  sample["background_pixels"] # (objects_num, 3, num_sampled_background_points // 3, 2)
            shared_objects = sample["shared_objects"] # (objects_num, )

            train_predictions = model.forward(train_images) # (3, D, H, W)
            
            # Use flattened descriptors for the three images to avoid memory issues with autograd and temporary tensors created for 2D indexing
            D, H, W = train_predictions.shape[1:]
            d1 = train_predictions[0].permute(1, 2, 0).reshape(-1, D)  # (H*W, D)
            d2 = train_predictions[1].permute(1, 2, 0).reshape(-1, D)  # (H*W, D)
            d3 = train_predictions[2].permute(1, 2, 0).reshape(-1, D)  # (H*W, D)

            # 1 - Compute the pixels matches/points pairs loss for each object in the scene
            total_loss, total_objects = 0.0, 0
            for i, pixels_matches in enumerate(objects_pixels_matches):
                # Remove padding from objects pixels (if any) and skip if empty
                pixels_matches = pixels_matches[pixels_matches[:, 0] >= 0]
                if len(pixels_matches) == 0:
                    continue

                # Log valid object views count
                obj_id = shared_objects[i].item()
                self.objects_occurrences_distribution[f"Obj_{obj_id}"].append(self.iteration)
                self.objects_matched_pixels_count[f"Obj_{obj_id}"] += len(pixels_matches)

                # Get the object points pairs and background points descriptors from the three images. Convert 2D to 1D indices
                img1_indices = pixels_matches[:, 1] * W + pixels_matches[:, 0]
                img2_indices = pixels_matches[:, 3] * W + pixels_matches[:, 2]
                img1_descriptors = d1[img1_indices] # (num_sampled_points, D)
                img2_descriptors = d2[img2_indices] # (num_sampled_points, D)

                img1_bg_pixels = background_pixels[i][0]
                img2_bg_pixels = background_pixels[i][1]
                img3_bg_pixels = background_pixels[i][2]
                img1_bg_indices = img1_bg_pixels[:, 1] * W + img1_bg_pixels[:, 0]
                img2_bg_indices = img2_bg_pixels[:, 1] * W + img2_bg_pixels[:, 0]
                img3_bg_indices = img3_bg_pixels[:, 1] * W + img3_bg_pixels[:, 0]
                img1_bg_descriptors = d1[img1_bg_indices] # (num_sampled_background_points // 3, D)
                img2_bg_descriptors = d2[img2_bg_indices] # (num_sampled_background_points // 3, D)
                img3_bg_descriptors = d3[img3_bg_indices] # (num_sampled_background_points // 3, D)
                bg_descriptors = torch.cat((img1_bg_descriptors, img2_bg_descriptors, img3_bg_descriptors), dim=0) # (num_sampled_background_points, D)

                # Concatenate object points descriptors and background points descriptors
                #   Let M = (2*num_sampled_points + num_sampled_background_points)
                descriptors = torch.cat([img1_descriptors, img2_descriptors, bg_descriptors], dim=0) # (M, D)

                # Prepare the positions of the object points pairs in the descriptors similarity matrix (just need to offset indexes by len(pixels_matches) to account for concatenated descriptors)
                indexes = torch.arange(0, len(pixels_matches), dtype=torch.int, device=descriptors.device)
                pairs_positions_img1 = torch.stack([indexes, indexes + len(pixels_matches)], dim=1)
                pairs_positions_img2 = torch.stack([indexes + len(pixels_matches), indexes], dim=1)
                centers_pairs_positions = torch.cat([pairs_positions_img1, pairs_positions_img2], dim=0) # (2*num_sampled_points, 2)

                # Compute the NT-Xent loss for the current object
                loss_per_object = self.get_pixels_matches_loss(descriptors, centers_pairs_positions)
                
                total_loss += loss_per_object
                total_objects += 1

                if self.debug:
                    debug.check_pixels_matches(train_images.clone().detach(), pixels_matches.clone().detach(), background_pixels[i].clone().detach())

            # Average the loss over all points
            assert total_objects > 0, "No objects to compute loss on"
            total_loss = total_loss / total_objects
            batch_pixels_loss += total_loss

            # Prepare data for the centers loss and DANN loss computation (all are computed per batch)
            descriptors = torch.cat([d1, d2, d3], dim=0) # (3*H*W, D)
            if self.compute_centers_loss:
                iteration_objects_masks = sample["iteration_objects_masks"] # (num_iteration_objects, 3*H*W)
                iteration_objects_masks_ids = sample["iteration_objects_masks_ids"] # (num_iteration_objects, )

                # Compute the number of pixels per mask (used to compute the mean centers)
                iteration_objects_masks_sums = iteration_objects_masks.sum(dim=1, keepdim=True) # (num_iteration_objects, 1)

                instances_centers = (iteration_objects_masks.float() @ descriptors) / iteration_objects_masks_sums  # (num_iteration_objects, D)
                for obj_id, center in zip(iteration_objects_masks_ids, instances_centers):
                    batch_objects_instances_centers[obj_id.item()].append(center)
            
            if self.compute_dann_loss:
                dann_descriptors_mask = sample["dann_descriptors_mask"] # (num_dann_points, )
                dann_descriptors_labels = sample["dann_domain_labels"] # (num_dann_points, )

                dann_descriptors = descriptors[dann_descriptors_mask.view(-1)] # (num_dann_points, D)
                batch_domain_classifier_descriptors.append(dann_descriptors)
                batch_domain_classifier_labels.append(dann_descriptors_labels)

            if self.debug:
                if self.compute_centers_loss:
                    debug.check_centers_loss_masks(train_images.clone(), iteration_objects_masks.clone(), iteration_objects_masks_ids.clone())
                if self.compute_dann_loss:
                    debug.check_dann_loss_descriptors(train_images.clone(), dann_descriptors_mask.clone(), dann_descriptors_labels.clone())

        # 2 - Compute the Centers Loss using shared and unique objects
        if self.compute_centers_loss:
            centers_loss = self.get_centers_loss(batch_objects_instances_centers)

        # 3 - Compute the DANN loss using the sampled descriptors from the whole batch
        if self.compute_dann_loss and len(dann_descriptors) > 0 and hasattr(model, "get_dann_loss"):
            dann_loss = self.get_dann_loss(model, batch_domain_classifier_descriptors, batch_domain_classifier_labels)


        # Return the total points loss, centers loss and DANN loss (if computed)
        losses = {}
        losses["Pixels loss"] = (batch_pixels_loss / len(data)) * self.pixels_matches_loss_scale
        if self.compute_centers_loss: losses["Centers loss"] = (centers_loss / len(data)) * self.center_loss_scale
        if self.compute_dann_loss: losses["Domain Classifier loss"] = dann_loss * self.dann_loss_scale
        return losses

    def log_training_metrics(self, outputs_dir: str, out_filename: str = "trainer_metrics.json"):
        metrics = {
            "objects_occurrences_distribution": self.objects_occurrences_distribution,
            "objects_matched_pixels_count": self.objects_matched_pixels_count,
        }

        file_path = os.path.join(os.path.expandvars(outputs_dir), out_filename)
        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=2)