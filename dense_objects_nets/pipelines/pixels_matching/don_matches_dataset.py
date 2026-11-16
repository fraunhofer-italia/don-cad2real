# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import torch

from copy import deepcopy
from collections import defaultdict
from typing import Dict, Tuple, Any, List

from dense_objects_nets.utils import io_utils
from dense_objects_nets.utils.don_snapshot import DonSnapshot, SnapshotObject
from dense_objects_nets.dataset.don_dataset import DonDataset
from dense_objects_nets.dataset.transforms import GeometricTransform
from dense_objects_nets.pipelines.pose_estimation.utils import models_utils # TODO move this from 'pose_estimation.utils'

class DonPixelsMatchesDataset(DonDataset):

    def init_dataset(self, scenes_config: Dict[str, Any]):
        # Load the object dimensions and symmetries (if available) for the dataset
        dataset_models = io_utils.load_config_file(scenes_config["models_names"])
        self.obj_dimensions = models_utils.get_obj_dimensions(dataset_models)

        dataset_obj_models_info = io_utils.load_config_file(scenes_config["models_info"]) if "models_info" in scenes_config else {}
        self.obj_symmetries = models_utils.get_models_symmetries(dataset_obj_models_info, self.dataset_objects_.keys())

    def setup_dataset_parameters(self, dataset_config: Dict[str, Any]):
        self.points_per_object = int(dataset_config["points_per_object"])
        self.background_points_per_iteration = int(dataset_config["background_points_per_iteration"])
        self.match_max_distance = float(dataset_config["match_max_distance"])
        self.min_num_valid_matches = int(dataset_config["min_num_valid_matches"])
        self.min_centers_object_pixels = int(dataset_config.get("min_centers_object_pixels", 300))
        self.dann_domain_objects_points = int(dataset_config.get("dann_domain_objects_points", 1000))
        self.iteration_max_shared_objects = dataset_config.get("iteration_max_shared_objects", 5)
        self.iteration_max_objects = dataset_config.get("iteration_max_objects", 15)

        # If a list of valid ids is provided, use it to filter the dataset objects
        self.valid_ids = list(self.dataset_objects_.keys()) # By default all dataset objects are valid
        if "valid_ids" in dataset_config and dataset_config["valid_ids"] is not None:
            valid_ids = set(dataset_config["valid_ids"])
            self.valid_ids = list(set(self.valid_ids) & valid_ids)

        # Check if only certain dataset scenes should be used for pixels matching
        self.valid_pixels_matching_snapshots_ids = defaultdict(list)
        if "pixels_matching_scenes_query" in dataset_config and dataset_config["pixels_matching_scenes_query"] is not None:
            valid_scenes_query = dataset_config["pixels_matching_scenes_query"]
            pixels_matching_valid_scenes_ids = [
                scene_id for scene_id, scene_name in enumerate(self.scenes_config_["scenes"]) if valid_scenes_query in scene_name
            ]

            for obj_id, obj_scenes_ids in self.dataset_objects_scenes_.items():
                for scene_id, scene_snapshots_ids in obj_scenes_ids.items():
                    if scene_id in pixels_matching_valid_scenes_ids:
                        self.valid_pixels_matching_snapshots_ids[obj_id] += scene_snapshots_ids

            if not self.valid_pixels_matching_snapshots_ids:
                print(f"WARN! - No valid snapshots found for the given pixels_matching_scenes_query '{valid_scenes_query}'. Using all dataset snapshots.")
                self.valid_pixels_matching_snapshots_ids = self.dataset_objects_
        else:
            self.valid_pixels_matching_snapshots_ids = self.dataset_objects_

        # Check if some scenes should be used only for self matching (i.e., two augmented views of the same snapshot)
        self.self_matching_snapshots_ids = defaultdict(list)
        if "self_matching_scenes_query" in dataset_config and dataset_config["self_matching_scenes_query"] is not None:
            self_matching_scenes_query = dataset_config["self_matching_scenes_query"]
            self_matching_scenes_ids = [
                scene_id for scene_id, scene_name in enumerate(self.scenes_config_["scenes"]) if self_matching_scenes_query in scene_name
            ]

            for obj_id, obj_scenes_ids in self.dataset_objects_scenes_.items():
                for scene_id, scene_snapshots_ids in obj_scenes_ids.items():
                    if scene_id in self_matching_scenes_ids:
                        self.self_matching_snapshots_ids[obj_id] += scene_snapshots_ids

            # Setup geometry transformations to be applied to self matching snapshots
            self.geometry_transform = GeometricTransform(apply_probability=1.0)

        # Similarly check if only certain dataset scenes should be used for s3 sampling
        self.valid_s3_snapshots_ids = []
        if "s3_scenes_query" in dataset_config and dataset_config["s3_scenes_query"] is not None:
            s3_scenes_query = dataset_config["s3_scenes_query"]
            valid_s3_scenes = [
                scene_id for scene_id, scene_name in enumerate(self.scenes_config_["scenes"]) if s3_scenes_query in scene_name
            ]
            
            for scene_id in valid_s3_scenes:
                self.valid_s3_snapshots_ids += self.dataset_scenes_[scene_id]

            if not self.valid_s3_snapshots_ids:
                print(f"WARN! - No valid snapshots found for the given s3_scenes_query '{dataset_config['s3_scenes_query']}'. Using all dataset snapshots for s3.")
                self.valid_s3_snapshots_ids = list(range(len(self.snapshots_)))
        else:
            self.valid_s3_snapshots_ids = list(range(len(self.snapshots_)))

        # Setup snapshot domains for DANN training
        self.snapshots_domains = [0] * len(self.snapshots_info_)  # By default all snapshots are from real domain (0)
        if "dann_synth_scenes_query" in dataset_config and dataset_config["dann_synth_scenes_query"] is not None:
            dann_synth_scenes_query = dataset_config["dann_synth_scenes_query"]
            for i, snapshot_info in enumerate(self.snapshots_info_):
                if dann_synth_scenes_query.lower() in snapshot_info.scene_path.lower():
                    self.snapshots_domains[i] = 1  # Mark snapshot as synthetic domain (1)

    def sample_snapshots_matching_pair(self, tries: int = 50, force_single_instance: bool = False) -> Tuple[DonSnapshot, DonSnapshot, int, int, List[SnapshotObject], List[torch.Tensor]]:
        while tries > 0:
            tries -= 1

            # Sample an object id from the dataset objects
            obj_id = self.rand_generator_.choice(self.valid_ids)
            if obj_id not in self.valid_pixels_matching_snapshots_ids:
                continue # This object is not present in any of the valid pixels matching snapshots

            # Sample a first snapshot containing the object
            s1_id = self.rand_generator_.choice(self.valid_pixels_matching_snapshots_ids[obj_id])
            s1 = self.snapshots_[s1_id]

            # Sample a second snapshot ensuring that the two share at least one object
            # If the first snapshots can only be used for self matching, force geometry augmentation on the second snapshot
            if s1_id in self.self_matching_snapshots_ids[obj_id]:
                s2_id = s1_id
                s2 = self.snapshots_[s2_id]
                s2 = self.geometry_transform(s2)

                # Requires objects to be matched by instance ids since snapshots are the same and we assume no pose information is available
                match_by_instance = True
            else:
                valid_s2_ids = list(set(self.valid_pixels_matching_snapshots_ids[obj_id]) - set(self.self_matching_snapshots_ids[obj_id]) - {s1_id})
                if not valid_s2_ids: 
                    continue
                s2_id = self.rand_generator_.choice(valid_s2_ids)
                s2 = self.snapshots_[s2_id]

                match_by_instance = False

            # If required, ensure that both snapshots contain only single instances of each object
            if force_single_instance and (len(s1.objects_ids) != len(set(s1.objects_ids)) or len(s2.objects_ids) != len(set(s2.objects_ids))):
                continue

            # Snapshots instances grouped by object id
            s1_unique_ids = set(s1.objects_ids) & set(self.valid_ids)
            s2_unique_ids = set(s2.objects_ids) & set(self.valid_ids)
            s1_objects = {obj_id: [obj.instance_id for obj in s1.get_objects_by_ids([obj_id])] for obj_id in s1_unique_ids}
            s2_objects = {obj_id: [obj.instance_id for obj in s2.get_objects_by_ids([obj_id])] for obj_id in s2_unique_ids}

            shared_objects_ids = list(set(s1_objects.keys()) & set(s2_objects.keys()))
            if len(shared_objects_ids) == 0:
                continue

            s1_shared_objects = []
            s2_shared_objects = []
            for obj_id in shared_objects_ids:
                # If snapshots contains multiple instances of the same object, use only one random instance for matching
                if match_by_instance:
                    shared_instances = list(set(s1_objects[obj_id]) & set(s2_objects[obj_id]))
                    if len(shared_instances) == 0:
                        continue
                    instance_id = self.rand_generator_.choice(shared_instances)
                    s1_shared_objects.append(s1.get_object_by_instance_id(instance_id))
                    s2_shared_objects.append(s2.get_object_by_instance_id(instance_id))

                else:
                    s1_instance = self.rand_generator_.choice(s1_objects[obj_id])
                    s2_instance = self.rand_generator_.choice(s2_objects[obj_id])
                    s1_shared_objects.append(s1.get_object_by_instance_id(s1_instance))
                    s2_shared_objects.append(s2.get_object_by_instance_id(s2_instance))

            # Cap the number of shared objects to avoid too large batches
            shared_objects = list(zip(s1_shared_objects, s2_shared_objects))
            if len(shared_objects) > self.iteration_max_shared_objects:
                shared_objects = self.rand_generator_.choice(shared_objects, self.iteration_max_shared_objects, replace=False)

            # Compute matches for each shared object and filter out objects with not enough matches
            valid_objects, valid_matches = [], []
            for s1_obj, s2_obj in shared_objects:
                s2_pixels_matches, distances = s1_obj.match_points(s2_obj) # (num_points, 2), (num_points, )
                if len(s2_pixels_matches) == 0:
                    continue

                obj_pixels_matches = torch.cat((s1_obj.pixels, s2_pixels_matches), dim=1) # (num_points, 4)

                valid_mask = distances < self.match_max_distance
                obj_pixels_matches = obj_pixels_matches[valid_mask] # (num_valid_points, 4)

                if len(obj_pixels_matches) < self.min_num_valid_matches:
                    continue

                valid_objects.append((s1_obj, s2_obj))
                valid_matches.append(obj_pixels_matches)

            # If at least one object has enough valid matches, return the snapshots and the valid matches
            if len(valid_matches) > 0:
                return s1, s2, s1_id, s2_id, valid_objects, valid_matches

        raise ValueError("Failed to sample a valid snapshots pair after multiple tries.")

    def get_pixels_matches_data(self, data: Dict[str, Any], s1: DonSnapshot, s2: DonSnapshot, s3: DonSnapshot, objects_matches: List[torch.Tensor], shared_objects_ids: List[int]):
        background_indexes = []
        objects_pixel_matches = []
        for obj_id, obj_pixels_matches in zip(shared_objects_ids, objects_matches):
            # Sample a random subset of visible model points to use as reference descriptors for the current iteration
            rand_indexes = torch.randperm(len(obj_pixels_matches), generator=self.torch_rand_generator_)
            rand_indexes = rand_indexes[:self.points_per_object]
            obj_pixels_matches = obj_pixels_matches[rand_indexes] # (points_per_object, 4)

            # Ensure all object points tensors have the same size
            if len(obj_pixels_matches) < self.points_per_object:
                padding = torch.full((self.points_per_object - len(obj_pixels_matches), 4), -1)
                obj_pixels_matches = torch.cat([obj_pixels_matches, padding], dim=0)

            # Sample a random subset of background points from the two snapshots
            s1_bg_pixels = torch.nonzero(s1.mask != obj_id)
            s2_bg_pixels = torch.nonzero(s2.mask != obj_id)
            s3_bg_pixels = torch.nonzero(s3.mask != obj_id if obj_id in s3.objects_ids else torch.ones_like(s3.mask)) # If the object is not present in s3, sample from the whole image

            rand_bg_pixels = []
            for bg_pixels in [s1_bg_pixels, s2_bg_pixels, s3_bg_pixels]:
                rand_indexes = torch.randperm(len(bg_pixels), generator=self.torch_rand_generator_)

                if len(rand_indexes) < self.background_points_per_iteration // 3:
                    # Duplicate some indexes to reach the required number of background points
                    num_replications = (self.background_points_per_iteration // 3) // len(rand_indexes) + 1
                    rand_indexes = rand_indexes.repeat(num_replications)

                rand_indexes = rand_indexes[:self.background_points_per_iteration // 3]
                bg_pixels = bg_pixels[rand_indexes]

                # Flip the background points to match the object points format (v, u) -> (u, v)
                bg_pixels = torch.index_select(bg_pixels, 1, torch.tensor([1,0], device=bg_pixels.device))
                rand_bg_pixels.append(bg_pixels)
            rand_bg_pixels = torch.stack(rand_bg_pixels, dim=0)  # (3, num_bg_points // 3, 2)

            background_indexes.append(rand_bg_pixels)
            objects_pixel_matches.append(obj_pixels_matches)

        data.update({
            "objects_pixels_matches": torch.stack(objects_pixel_matches).int(),  # (num_objects, points_per_object, 4)
            "background_pixels": torch.stack(background_indexes).int(),  # (num_objects, 3, num_bg_points // 3, 2)
        })

    def get_centers_data(self, data: Dict[str, Any], s1: DonSnapshot, s2: DonSnapshot, s3: DonSnapshot, tries: int = 50):
        remaining_tries = tries
        min_object_pixels = self.min_centers_object_pixels
        
        while remaining_tries > 0:
            remaining_tries -= 1

            # Relax the minimum number of pixels requirement to increase the chances of sampling valid objects
            if remaining_tries <= 0:
                min_object_pixels = 1 # Last try, accept any object
            elif remaining_tries < tries // 2:
                min_object_pixels = int(min_object_pixels * 0.5)

            # Collect all object masks for the current iteration
            iteration_objects_masks = []
            iteration_objects_masks_ids = []
            iteration_objects_masks_snapshot_ids = []
            for snapshot_objects, snapshot_id in [(s1.objects, 0), (s2.objects, 1), (s3.objects, 2)]:
                for obj in snapshot_objects:
                    if obj.id in self.valid_ids and obj.mask.sum() >= min_object_pixels:
                        iteration_objects_masks.append(obj.mask)
                        iteration_objects_masks_ids.append(obj.id)
                        iteration_objects_masks_snapshot_ids.append(snapshot_id)

            # Sample the required number of instances for the current iteration
            sampled_indexes = torch.randperm(len(iteration_objects_masks), generator=self.torch_rand_generator_)
            sampled_indexes = sampled_indexes[:self.iteration_max_objects]
            
            iteration_objects_masks = torch.stack(iteration_objects_masks)[sampled_indexes]  # (num_sampled_objects, H, W)
            iteration_objects_masks_ids = torch.tensor(iteration_objects_masks_ids)[sampled_indexes]
            iteration_objects_masks_snapshot_ids = torch.tensor(iteration_objects_masks_snapshot_ids)[sampled_indexes]

            # Check that at least two different instances of the same object are present in the sampled masks
            if len(set(iteration_objects_masks_ids.tolist())) != len(iteration_objects_masks_ids):
                # Reshape masks to ease indexing during training
                masks = torch.zeros((len(iteration_objects_masks), 3, s1.H, s1.W), device=s1.rgb.device, dtype=torch.uint8)
                for i, (mask, snapshot_id) in enumerate(zip(iteration_objects_masks, iteration_objects_masks_snapshot_ids)):
                    masks[i][snapshot_id] = mask
                masks = masks.view(len(iteration_objects_masks), -1).bool()  # (num_sampled_objects, 3*H*W)

                data.update({
                    "iteration_objects_masks": masks,  # (num_sampled_objects, 3*H*W)
                    "iteration_objects_masks_ids": iteration_objects_masks_ids,  # (num_sampled_objects, )
                })
                return

        raise ValueError("Failed to sample valid objects masks after multiple tries. This should not happen.")

    def get_dann_data(self, data: Dict[str, Any], s1: DonSnapshot, s2: DonSnapshot, s3: DonSnapshot, s1_domain: int, s2_domain: int, s3_domain: int):
        # Sample a random subset of object pixels from the three snapshots trying to balance real and synthetic domains
        num_synth_domains = sum([1 for d in [s1_domain, s2_domain, s3_domain] if d == 1]) # 0 = real, 1 = synth
        num_real_domains = 3 - num_synth_domains

        if num_real_domains == 0 or num_synth_domains == 0:
            real_split, synth_split = 1/3, 1/3
        else:
            real_split = (1/2) / num_real_domains # Either 1/2 or 1/4
            synth_split = (1/2) / num_synth_domains

        H, W = s1.shape
        snapshots_objects_pixels = []
        snapshots_domains_labels = []
        for i, (mask, domain) in enumerate([(s1.mask, s1_domain), (s2.mask, s2_domain), (s3.mask, s3_domain)]):
            objects_pixels = torch.nonzero(mask.flatten() > 0)
            rand_indexes = torch.randperm(len(objects_pixels), generator=self.torch_rand_generator_)
            
            split = real_split if domain == 0 else synth_split
            rand_indexes = rand_indexes[:int(self.dann_domain_objects_points * split)]

            objects_pixels_mask = objects_pixels[rand_indexes] # (num_dann_objs_points, 1)

            # Offset the pixel indexes to account for the snapshot position in the batch
            offset = i * H * W
            objects_pixels_mask += offset

            domain_labels = torch.full((len(objects_pixels_mask),), domain, device=s1.rgb.device, dtype=torch.float32)

            snapshots_objects_pixels.append(objects_pixels_mask)
            snapshots_domains_labels.append(domain_labels)

        dann_descriptors_mask = torch.cat(snapshots_objects_pixels, dim=0)
        dann_domain_labels = torch.cat(snapshots_domains_labels, dim=0)

        # Ensure enough DANN pixels have been sampled, if not duplicate some (required for batching)
        if len(dann_descriptors_mask) < self.dann_domain_objects_points:
            num_needed = self.dann_domain_objects_points - len(dann_descriptors_mask)
            replication_indexes = torch.randperm(len(dann_descriptors_mask), generator=self.torch_rand_generator_)
            replication_indexes = replication_indexes.repeat((num_needed // len(dann_descriptors_mask)) + 1)
            replication_indexes = replication_indexes[:num_needed]

            dann_descriptors_mask = torch.cat([dann_descriptors_mask, dann_descriptors_mask[replication_indexes]], dim=0)
            dann_domain_labels = torch.cat([dann_domain_labels, dann_domain_labels[replication_indexes]], dim=0)

        data.update({
            "dann_descriptors_mask": dann_descriptors_mask,  # (num_dann_objs_points, 1)
            "dann_domain_labels": dann_domain_labels.unsqueeze(1),  # (num_dann_objs_points, 1)
        })

    def __getitem__(self, idx: int):
        # Sample two random snapshots sharing at least one object and compute matches for each shared object
        s1, s2, s1_id, s2_id, shared_objects, objects_matches = self.sample_snapshots_matching_pair()
        shared_objects_ids = [s1_obj.id for s1_obj, s2_obj in shared_objects]

        # Sample a third random snapshot used to generate negative matches (this ensures all objects are compared to each other during training)
        s3_id = self.rand_generator_.choice(self.valid_s3_snapshots_ids)
        s3: DonSnapshot = self.snapshots_[s3_id]

        # Base data sample
        data = {
            "train_images": torch.stack([s1.rgb, s2.rgb, s3.rgb]), # (3, 3, H, W)
            "images_masks": torch.stack([s1.mask, s2.mask, s3.mask]), # (3, H, W)
            "shared_objects": torch.tensor(shared_objects_ids), # (num_objects, )
        }
        
        # Add data needed for pixels matching training
        self.get_pixels_matches_data(data, s1, s2, s3, objects_matches, shared_objects_ids)

        # Add data needed for centers loss computation
        self.get_centers_data(data, s1, s2, s3)

        # Add data needed for DANN loss computation
        s1_domain = self.snapshots_domains[s1_id]
        s2_domain = self.snapshots_domains[s2_id]
        s3_domain = self.snapshots_domains[s3_id]
        self.get_dann_data(data, s1, s2, s3, s1_domain, s2_domain, s3_domain)

        return data
    
    def __collate_fn__(self, batch: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        return batch # Do not collate, return a list of samples, allowing for variable size tensors

    def load_evaluation_samples(self, num_samples: int, max_tries_per_sample: int = 50) -> List[Dict[str, torch.Tensor]]:
        eval_samples = []
        tries_per_sample = max_tries_per_sample
        while len(eval_samples) < num_samples and tries_per_sample > 0:
            tries_per_sample -= 1

            # Sample an object id from the dataset objects
            obj_id = self.rand_generator_.choice(self.valid_ids)

            # Sample a first snapshot containing the object
            s1_id = self.rand_generator_.choice(self.dataset_objects_[obj_id])
            s1 = self.snapshots_[s1_id]

            # Sample a second snapshot ensuring that the two share at least one object
            valid_s2_ids = list(set(self.dataset_objects_[obj_id]) - {s1_id})
            s2_id = self.rand_generator_.choice(valid_s2_ids)
            s2 = self.snapshots_[s2_id]

            # Get shared objects between the two snapshots
            shared_objects_ids = list(set(s1.objects_ids) & set(s2.objects_ids))

            if len(shared_objects_ids) == 0:
                continue

            # Check if both snapshots contain multiple instances of the same object, if so, keep only one and mask out the others
            for obj_id in shared_objects_ids:
                for snapshot in [s1, s2]:
                    obj_instances = snapshot.get_objects_by_ids([obj_id])
                    if len(obj_instances) == 1:
                        continue
                    
                    # Pick a random instance to keep
                    obj_instances_ids = [inst.instance_id for inst in obj_instances]
                    chosen_instance_id = self.rand_generator_.choice(obj_instances_ids)
                    
                    # Remove all other instances of the same object from the snapshot and mask them out in the rgb/mask/instances_mask
                    snapshot.objects = [obj for obj in snapshot.objects if obj.instance_id == chosen_instance_id or obj.id != obj_id]
                    for removed_instance_id in set(obj_instances_ids) - {chosen_instance_id}:
                        mask = (snapshot.instances_mask == removed_instance_id)
                        snapshot.rgb[:, mask] = 0
                        snapshot.mask[mask] = 0
                        snapshot.instances_mask[mask] = 0

            # Compute matches for each shared object and filter out objects with not enough matches
            valid_objects_ids, valid_matches = [], []
            for obj_id in shared_objects_ids:
                s1_obj = s1.get_object_by_id(obj_id) # Only one instance per object id is present after the previous step
                s2_obj = s2.get_object_by_id(obj_id)

                s2_pixels_matches, distances = s1_obj.match_points(s2_obj) # (num_points, 2), (num_points, )
                if len(s2_pixels_matches) == 0:
                    continue

                obj_pixels_matches = torch.cat((s1_obj.pixels, s2_pixels_matches), dim=1) # (num_points, 4)

                valid_mask = distances < self.match_max_distance
                obj_pixels_matches = obj_pixels_matches[valid_mask] # (num_valid_points, 4)

                if len(obj_pixels_matches) < self.min_num_valid_matches:
                    continue

                valid_objects_ids.append(obj_id)
                valid_matches.append(obj_pixels_matches)
        
            # Sample a random subset of matches and ensure all tensors have the same size
            for i, obj_pixels_matches in enumerate(valid_matches):
                rand_indexes = torch.randperm(len(obj_pixels_matches), generator=self.torch_rand_generator_)
                rand_indexes = rand_indexes[:self.points_per_object]
                obj_pixels_matches = obj_pixels_matches[rand_indexes] # (points_per_object, 4)

                if len(obj_pixels_matches) < self.points_per_object:
                    padding = torch.full((self.points_per_object - len(obj_pixels_matches), 4), -1)
                    obj_pixels_matches = torch.cat([obj_pixels_matches, padding], dim=0)

                valid_matches[i] = obj_pixels_matches

            if len(valid_matches) > 0:
                data = {
                    "s1": deepcopy(s1.to_dict()),
                    "s2": deepcopy(s2.to_dict()),
                    "shared_objects": torch.tensor(valid_objects_ids),  # (num_objects, )
                    "objects_pixels_matches": torch.stack(valid_matches).int(),  # (num_objects, points_per_object, 4)
                }
                
                # Add information about object dimensions and symmetries (if available)
                obj_dimensions = [self.obj_dimensions[s2.get_object_by_id(obj_id).name] for obj_id in valid_objects_ids]
                data["objects_dimensions"] = torch.tensor(obj_dimensions)  # (num_objects, 3)
                if self.obj_symmetries:
                    obj_symmetries = [self.obj_symmetries[obj_id] for obj_id in valid_objects_ids]
                    data["objects_symmetries"] = obj_symmetries  # List of (num_symmetries, 3, 3) tensors   
                
                eval_samples.append(data)
                tries_per_sample = max_tries_per_sample  # Reset tries for the next sample

        if len(eval_samples) < num_samples:
            raise ValueError("Failed to sample the required number of evaluation samples after multiple tries.")

        return eval_samples
