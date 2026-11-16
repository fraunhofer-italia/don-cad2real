# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import os
import torch
import random
import numpy as np

from enum import Enum
from tqdm import tqdm
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Dict, Tuple, Any, List

from torch.utils.data import Dataset
from dense_objects_nets.utils import io_utils
from dense_objects_nets.utils.don_snapshot import DonSnapshot, DonSnapshotInfo


class DonDatasetType(Enum):
    SINGLE_OBJECT_SCENE_ONE_ID = "SINGLE_OBJECT_SCENE_ONE_ID"
    SINGLE_OBJECT_SCENE_MULTIPLE_ID = "SINGLE_OBJECT_SCENE_MULTIPLE_ID"
    MULTI_OBJECTS_SCENE = "MULTI_OBJECTS_SCENE"
    EVALUATION = "EVALUATION"

class SnapshotLoader():
    def __init__(self, snapshots_info: List[DonSnapshotInfo], transforms: List[Callable] = []):
        self.snapshots_info_ = snapshots_info
        self.transforms_ = transforms

    def __getitem__(self, id: int) -> DonSnapshot:
        snapshot = DonSnapshot.from_config_dict(
            scene_base_path=self.snapshots_info_[id].scene_path,
            snapshot_dict=self.snapshots_info_[id].dict,
        )

        # Apply the transforms to the loaded snapshot (if any)
        for transform in self.transforms_:
            snapshot = transform(snapshot)

        return snapshot
    
    def __len__(self):
        return len(self.snapshots_info_)

class DonDataset(Dataset, ABC):
    
    def __init__(self, scenes_config: Dict[str, Any], dataset_config: Dict[str, Any], transforms: List[Callable] = []):
        # Save the given input config files
        self.config_ = dataset_config
        self.scenes_config_ = scenes_config
        
        # Load info associated to each snapshot in the dataset
        #
        # 'dataset_scenes_' is a dictionary that maps each scene id to a list of snapshot indexes belonging to that scene
        # 'dataset_objects_' is a dictionary that maps each object id to a list of snapshot indexes where the object is present
        # 'dataset_objects_scenes_' is a dictionary that maps each object id to a list of snapshot indexes grouping them by scene they belong to
        self.snapshots_info_, self.dataset_scenes_, self.dataset_objects_, self.dataset_objects_scenes_ = self.load_dataset_snapshots_info(scenes_config)

        # Loader to load the snapshots on demand using the same indexing as 'self.snapshots_info_'
        self.snapshots_ = SnapshotLoader(self.snapshots_info_, transforms=transforms)

        # Initialize the dataset scenes
        self.init_dataset(scenes_config)

        # Setup the parameters needed by the dataset to generate each training item
        self.setup_dataset_parameters(dataset_config)

        # Initialize the dataset type
        # Can be used to determine the behavior of the '__getitem__' function
        self.dataset_type_ = DonDatasetType(scenes_config["dataset_type"])
        self.setup_data_load_type_probabilities(self.dataset_type_)

        # Initialize the random generators
        self.seed = dataset_config.get("seed", None)

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        pass

    def init_dataset(self, scenes_config: Dict[str, Any]):
        pass

    def setup_dataset_parameters(self, dataset_config: Dict[str, Any]):
        pass

    def setup_data_load_type_probabilities(self, dataset_type: DonDatasetType):
        pass

    def reset_generators(self, seed: int = None):
        self.rand_generator_ = np.random.default_rng()
        self.torch_rand_generator_ = torch.Generator()
        
        if seed is not None:
            seed = int(seed)

            random.seed(seed)

            np.random.seed(seed)
            self.rand_generator_ = np.random.default_rng(seed=seed)
            
            torch.manual_seed(int(seed))
            self.torch_rand_generator_ = self.torch_rand_generator_.manual_seed(int(seed))

    def load_dataset_snapshots_info(self, scenes_config: Dict[str, Any]) -> Tuple[List[DonSnapshotInfo], Dict[int, List[int]], Dict[int, List[int]], Dict[int, Dict[int, List[int]]]]:
        snapshots_info = []
        snapshot_scenes_ids = []
        for scene_id, scene in enumerate(tqdm(scenes_config["scenes"], desc="Loading dataset scenes", leave=False)):
            dataset_base_path = os.path.expandvars(scenes_config["dataset_base_path"])
            scene_path = os.path.join(dataset_base_path, scene)
            
            snapshot_info = io_utils.load_snapshots_info(scene_path)
            
            snapshots_info += snapshot_info
            snapshot_scenes_ids += ([scene_id] * len(snapshot_info))

        dataset_scenes = defaultdict(lambda: [])
        dataset_objects = defaultdict(lambda: [])
        dataset_objects_scenes = defaultdict(lambda: defaultdict(lambda: []))
        for snapshot_id, (snapshot_info, snapshot_scene_id), in enumerate(zip(snapshots_info, snapshot_scenes_ids)):
            dataset_scenes[snapshot_scene_id].append(snapshot_id)

            for object_id in snapshot_info.objects_ids:
                dataset_objects[object_id].append(snapshot_id)
                dataset_objects_scenes[object_id][snapshot_scene_id].append(snapshot_id)

        return snapshots_info, dataset_scenes, dataset_objects, dataset_objects_scenes

    def load_validation_samples(self, num_samples: int, augment: bool = False, batch: bool = True) -> Dict[str, torch.Tensor] | List[Dict[str, torch.Tensor]]:
        # Enable/disable data augmentation
        transforms = self.transforms
        self.transforms = [] if not augment else transforms

        indexes = self.rand_generator_.choice(list(range(0, len(self))), size=min(num_samples, len(self)), replace=False)
        samples = [self[i] for i in indexes]
        if batch: samples = self.__collate_fn__(samples)

        # Restore transforms
        self.transforms = transforms
        return samples

    def load_evaluation_samples(self, num_samples: int) -> List[Dict[str, torch.Tensor]]:
        # By default, evaluation samples are not batched and not augmented
        return self.load_validation_samples(num_samples, augment=False, batch=False)

    def __len__(self):
        return len(self.snapshots_info_)
    
    def __collate_fn__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Default collate, same as torch.DataLoader
        collated_batch = {}
        for key in batch[0].keys():
            collated_batch[key] = torch.stack([item[key] for item in batch])
        return collated_batch
    
    def __worker_init_fn__(self, worker_id: int):
        worker_info = torch.utils.data.get_worker_info()
        base_seed = torch.initial_seed() % (2**32)
        worker_info.dataset.reset_generators(seed=base_seed)

    def get_dataloader(self, batch_size: int = 1, num_workers: int = 0, pin_memory: bool = True, prefetch_factor: int = 3) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None, # Avoid setting prefetch_factor > 1 when num_workers=0
            collate_fn=self.__collate_fn__, # Allow dataset to define its own collate function (default is torch.stack)
            worker_init_fn=self.__worker_init_fn__, # Used to re-seed the random generators for each worker
            generator=self.torch_rand_generator_, # Use the dataset torch generator to have reproducible results
        )

    @property
    def scenes_config(self) -> Dict[str, Any]:
        return self.scenes_config_

    @property
    def dataset_config(self) -> Dict[str, Any]:
        return self.config_

    @dataset_config.setter
    def dataset_config(self, config: Dict[str, Any]):
        self.setup_dataset_parameters(config)
        self.seed = config.get("seed", None)
        self.config_ = config

    @property
    def objects_ids(self) -> List[int]:
        return list(self.dataset_objects_.keys())

    @property
    def transforms(self) -> List[Callable]:
        return self.snapshots_.transforms_

    @transforms.setter
    def transforms(self, transforms: List[Callable]):
        self.snapshots_.transforms_ = transforms

    @property
    def type(self) -> 'DonDatasetType':
        return self.dataset_type_

    @type.setter
    def type(self, t: str):
        self.dataset_type_ = DonDatasetType(t)
        self.setup_data_load_type_probabilities(self.dataset_type_)

    @property
    def seed(self) -> int:
        return self.seed_
    
    @seed.setter
    def seed(self, seed: int):
        self.seed_ = seed
        self.reset_generators(seed=seed)