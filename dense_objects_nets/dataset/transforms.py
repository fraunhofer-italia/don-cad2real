# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import os
import torch
import random
import torchvision

import numpy as np

from typing import Tuple
from torchvision.transforms import v2
from torchvision.datasets.utils import download_and_extract_archive

from dense_objects_nets.utils.don_snapshot import DonSnapshot


class RandomizeBackground(object):
    """
    Randomize the background of the input images data using random images from the 'Caltech256' dataset. 
    """

    def __init__(self, apply_probability: float = 0.8):
        root = os.path.expandvars("${WS_DIR}/data/")
        caltech_root = os.path.join(root, "caltech256")
        dataset_dir = os.path.join(caltech_root, "256_ObjectCategories")

        if not os.path.isdir(dataset_dir):
            os.makedirs(caltech_root, exist_ok=True)

            download_and_extract_archive(
                url="https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar",
                download_root=caltech_root,
                filename="256_ObjectCategories.tar",
                md5="67b4f42ca05d46448c6bb8ecd2220f6d",
            )
        
        self.dataset = torchvision.datasets.Caltech256(root=root, download=False)
        self.apply_probability = apply_probability

    def __call__(self, snapshot: DonSnapshot) -> DonSnapshot:
        if random.random() > self.apply_probability:
            return snapshot # Do nothing

        # Sample a random image from the dataset
        H, W = snapshot.shape
        rand_img = self.sample_img(out_size=(H, W))

        # Paste the object pixels onto the random image
        mask = snapshot.mask.bool()
        rand_img[:, mask] = snapshot.rgb[:, mask]

        snapshot.rgb = rand_img
        return snapshot

    def sample_img(self, out_size: Tuple[int, int] = (224, 224)):
        img, label = random.choice(self.dataset)
        img = v2.ToImage()(img)
        img = v2.RGB()(img)
        img = v2.ToDtype(torch.float32, scale=True)(img)
        img = v2.Resize(size=out_size)(img)
        return img



class ColorAugmentation(object):
    """
    Apply a random subset of color augmentations to each image in the input images data.
    """

    def __init__(self, apply_probability: float = 0.8, augment_probabilities: dict = None):
        self.apply_probability = apply_probability

        # Probabilities for each transform
        if augment_probabilities is not None:
            self.probs = augment_probabilities
        else:
            self.probs = {
                "color_jitter": 0.7,
                "gamma": 0.5,
                "noise": 0.3,
                "blur": 0.3,
                "posterize": 0.2,
                "solarize": 0.05,
                "equalize": 0.0, # Disable equalize by default
            }

        # Define transforms
        self.transforms = {
            "color_jitter": v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            "gamma": self._gamma,
            "noise": self._noise,
            "blur": v2.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
            "solarize": v2.RandomSolarize(threshold=0.8, p=1.0),
            "posterize": v2.RandomPosterize(bits=4, p=1.0),
            "equalize": v2.RandomEqualize(p=1.0),
        }

        assert all(name in self.transforms for name in self.probs.keys()), "All probabilities must have a corresponding transform"

    def _gamma(self, img: torch.Tensor):
        g = random.uniform(0.5, 2)
        return v2.functional.adjust_gamma(img, gamma=g)

    def _noise(self, img: torch.Tensor):
        noise = torch.randn_like(img) * 0.02
        return torch.clamp(img + noise, 0, 1)

    def __call__(self, snapshot: DonSnapshot) -> DonSnapshot:
        if random.random() > self.apply_probability:
            return snapshot # Do nothing

        for name, prob in self.probs.items():
            if random.random() < prob:
                snapshot.rgb = self.transforms[name](snapshot.rgb)

        return snapshot



class GeometricTransform(object):
    """
    Apply a random subset of geometric transformations to each image in the input images data.

    NOTE: THIS TRANSFORM INVALIDATES THE SNAPSHOT T_c2w AND K FIELDS! OBJECTS 2D/3D PIXELS/POINTS ARE STILL VALID.
    """

    def __init__(self, apply_probability: float = 0.5, augment_probabilities: dict = None):
        self.apply_probability = apply_probability
        self.interpolation = torchvision.transforms.InterpolationMode.NEAREST

        # Probabilities for each transform
        if augment_probabilities is not None:
            self.probs = augment_probabilities
        else:
            self.probs = {
                "random_affine": 0.5,
                "random_rotation": 0.3,
                "random_horizontal_flip": 0.3,
                "random_vertical_flip": 0.3,
                "random_perspective": 0.05, # Perspective is quite extreme, use with caution
            }

        # Define transforms
        self.transforms = {
            "random_affine": self._random_affine,
            "random_perspective": self._random_perspective,
            "random_rotation": self._random_rotation,
            "random_horizontal_flip": self._random_horizontal_flip,
            "random_vertical_flip": self._random_vertical_flip,
        }

        assert all(name in self.transforms for name in self.probs.keys()), "All probabilities must have a corresponding transform"

    def __call__(self, snapshot: DonSnapshot) -> DonSnapshot:
        if random.random() > self.apply_probability:
            return snapshot # Do nothing

        # Stack snapshot images in a single tensor to apply the same transforms to all channels
        snapshot_images = torch.vstack([
            snapshot.rgb, # 3 channels
            snapshot.mask[None, :, :], snapshot.instances_mask[None, :, :], # 2 more channels
            snapshot.depth[None, :, :] # Final channel
        ]) # (6, H, W)

        # # Similarly for each object create a mask of objects pixels/points indexes later used to map them to the transformed image
        objects_masks = []
        for obj in snapshot.objects:
            obj_points_indexes = torch.arange(1, len(obj.points) + 1, dtype=torch.int64)

            obj_indexes_mask = torch.zeros(obj.shape, dtype=torch.int64) # (H, W)
            obj_indexes_mask[obj.mask] = obj_points_indexes
        
            objects_masks.append(obj_indexes_mask)
        objects_masks = torch.stack(objects_masks) # (num_objects, H, W)
        
        stacked_images = torch.vstack([snapshot_images, objects_masks]) # (6 + num_objects, H, W)


        # Apply the transforms
        for name, prob in self.probs.items():
            if random.random() < prob:
                stacked_images = self.transforms[name](stacked_images)


        # Update snapshot' objects
        # - Removes snapshot objects that have no valid pixels after transformation
        # - Updates pixels/points indexes for objects that are still valid
        valid_objects = []
        for i, obj in enumerate(snapshot.objects):
            indexes = stacked_images[6 + i, :, :].long() - 1 # (H, W) with -1 for background pixels
            object_mask = indexes >= 0

            if not object_mask.any():
                continue # This object has no valid pixels after transformation, remove it

            # Update points to keep only valid ones
            obj.points = obj.points[indexes[object_mask]]

            # Recompute the object's pixels
            object_pixels = torch.nonzero(object_mask)
            object_pixels = torch.index_select(object_pixels, 1, torch.tensor([1, 0], device=object_pixels.device)) # (N, 2) tensor of pixels in (x, y) format
            obj.pixels = object_pixels

            valid_objects.append(obj)

        if len(valid_objects) == 0:
            # If no objects are valid after the transform, return the original snapshot
            return snapshot

        # Unstack the images back to the snapshot
        snapshot.rgb = stacked_images[0:3, :, :]
        snapshot.mask = stacked_images[3, :, :].int()
        snapshot.instances_mask = stacked_images[4, :, :].int()
        snapshot.depth = stacked_images[5, :, :]
        snapshot.objects = valid_objects
        snapshot.T_c2w = None # Invalidate T_c2w and K since they are not valid anymore
        snapshot.K = None

        return snapshot

    def _random_affine(self, img: torch.Tensor):
        H, W = img.size()[-2:]
        angle, translations, scale, shear = v2.RandomAffine.get_params(
            degrees=[-45, 45],
            translate=[0.3, 0.3],
            scale_ranges=[0.7, 1.5],
            shears=[0, 1, 0, 1],
            img_size=(H, W)
        )
        return v2.functional.affine(img, angle, translations, scale, shear, interpolation=self.interpolation)

    def _random_perspective(self, img: torch.Tensor):
        H, W = img.size()[-2:]
        distortion_scale = np.random.uniform(0.2, 0.5)
        start_points, end_points = v2.RandomPerspective.get_params(
            width=W,
            height=H,
            distortion_scale=distortion_scale
        )
        return v2.functional.perspective(img, start_points, end_points, interpolation=self.interpolation)

    def _random_rotation(self, img: torch.Tensor):
        H, W = img.size()[-2:]
        angle, translations, scale, shear = v2.RandomAffine.get_params(
            degrees=[-45, 45],
            translate=[0, 0],
            scale_ranges=[1.0, 1.0],
            shears=[0, 0, 0, 0],
            img_size=(H, W)
        )
        return v2.functional.affine(img, angle, translations, scale, shear, interpolation=self.interpolation)

    def _random_horizontal_flip(self, img: torch.Tensor):
        return v2.functional.horizontal_flip(img)

    def _random_vertical_flip(self, img: torch.Tensor):
        return v2.functional.vertical_flip(img)