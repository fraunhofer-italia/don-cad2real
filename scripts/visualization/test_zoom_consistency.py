# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

# GUI functions must be imported before any 'cv2' import to avoid conflicts with PyQt5
from dense_objects_nets.utils.gui_utils import select_runs_folders

import os
import torch
import cv2 as cv
import numpy as np

from dense_objects_nets.evaluation.don_evaluator import DonEvaluator
from dense_objects_nets.utils import io_utils, visualization_utils, evaluation_utils
from dense_objects_nets.pipelines.pose_estimation.utils import dynamic_zoom_in as dzi

def visualize_descriptors(img1: torch.Tensor, img2: torch.Tensor, pred1: torch.Tensor, pred2: torch.Tensor, real_matches: torch.Tensor, pred_matches: torch.Tensor) -> np.ndarray:
    img1 = cv.cvtColor(img1.permute(1, 2, 0).cpu().numpy(), cv.COLOR_RGB2BGR)
    img2 = cv.cvtColor(img2.permute(1, 2, 0).cpu().numpy(), cv.COLOR_RGB2BGR)
    imgs = np.hstack([img1, img2])
    
    pred1_img = visualization_utils.render_descriptors_image([pred1.permute(1, 2, 0)]).cpu().numpy()
    pred2_img = visualization_utils.render_descriptors_image([pred2.permute(1, 2, 0)]).cpu().numpy()
    pred_imgs = np.hstack([pred1_img, pred2_img])

    H, W, D = img1.shape
    real_matches = real_matches.cpu().numpy()
    pred_matches = pred_matches.cpu().numpy() + np.array([0, 0, W, 0])
    colors = visualization_utils.colors[:len(real_matches)]

    imgs = visualization_utils.draw_matches(imgs, real_matches, real_matches + [[W, 0]], colors)
    imgs = visualization_utils.draw_matches(imgs, pred_matches[:, :2], pred_matches[:, 2:], colors, marker=cv.MARKER_STAR)

    pred_imgs = visualization_utils.draw_matches(pred_imgs, real_matches, real_matches + [[W, 0]], colors)
    pred_imgs = visualization_utils.draw_matches(pred_imgs, pred_matches[:, :2], pred_matches[:, 2:], colors, marker=cv.MARKER_STAR)

    return np.hstack([imgs, pred_imgs])

def check_matches_consistency(imgs: torch.Tensor, masks: torch.Tensor, model: torch.nn.Module, evaluator: DonEvaluator) -> np.ndarray:
    # Sample two random backgrounds for the images
    imgs_rbg1 = evaluation_utils.randomize_background(imgs, masks)
    imgs_rbg2 = evaluation_utils.randomize_background(imgs, masks)

    matches_imgs = []
    for img_rbg1, img_rbg2, mask in zip(imgs_rbg1, imgs_rbg2, masks):
        # Sample random pixels from the mask
        mask_indices = torch.nonzero(mask)
        num_samples = min(5, len(mask_indices))
        sampled_indices = mask_indices[torch.randperm(len(mask_indices))[:num_samples]]
        sampled_indices = sampled_indices.flip(1)  # (y, x) -> (x, y)

        # Move tensors to the model's device
        img_rbg1 = img_rbg1.to(model.device)
        img_rbg2 = img_rbg2.to(model.device)
        sampled_indices = sampled_indices.to(model.device)

        # Forward pass on both images. Manually upsample the output to the original image size to work with all images sizes
        with torch.no_grad():
            pred1 = model(img_rbg1.unsqueeze(0), upsample=False)
            pred2 = model(img_rbg2.unsqueeze(0), upsample=False)
            pred1 = torch.nn.functional.interpolate(pred1, size=img_rbg1.shape[-2:], mode="bilinear", align_corners=True)
            pred2 = torch.nn.functional.interpolate(pred2, size=img_rbg2.shape[-2:], mode="bilinear", align_corners=True)
        pred1 = pred1.squeeze(0)
        pred2 = pred2.squeeze(0)

        # Find descriptor matches
        matches = evaluator.find_descriptors_matches(pred1, pred2, sampled_indices)

        img = visualize_descriptors(img_rbg1, img_rbg2, pred1, pred2, sampled_indices, matches)
        matches_imgs.append(img)

    matches_imgs = np.vstack(matches_imgs)

    return matches_imgs


if __name__ == "__main__":
    run_folders, _ = select_runs_folders(base_dir="${WS_DIR}/runs", query_file="evaluation_config.yml")
    if len(run_folders) != 1:
        print("Please select a single run folder containing an evaluation config file.")
        exit(0)

    evaluation_config_filename = os.path.join(run_folders[0], "evaluation_config.yml")
    evaluation_config = io_utils.load_config_file(filename=evaluation_config_filename)

    evaluator, evaluation_dataset, best_model, last_model, training_params = (
        evaluation_utils.load_evaluation_data(evaluation_config)
    )

    snapshots = evaluation_dataset.snapshots_
    snapshots_ids = list(range(len(snapshots)))
    np.random.shuffle(snapshots_ids)

    for i, snapshot_id in enumerate(snapshots_ids):
        snapshot = snapshots[snapshot_id]
        
        # Original images and masks
        obj_imgs = torch.stack(
            [
            snapshot.rgb * snapshot.get_object_by_id(obj_id).mask
            for obj_id in snapshot.objects_ids
            ]
        )
        obj_masks = torch.stack([
            snapshot.get_object_by_id(obj_id).mask
            for obj_id in snapshot.objects_ids
        ])
        
        # Zoomed-in images and masks
        zoomed_obj_imgs = torch.stack(
            [
                dzi.crop_and_zoom_in(snapshot.rgb, snapshot.get_object_by_id(obj_id).bbox)
                for obj_id in snapshot.objects_ids
            ]
        )
        zoomed_obj_masks = torch.stack([
                dzi.crop_and_zoom_in(snapshot.get_object_by_id(obj_id).mask, snapshot.get_object_by_id(obj_id).bbox)
                for obj_id in snapshot.objects_ids
        ])

        while True:
            obj_matches_imgs = check_matches_consistency(obj_imgs, obj_masks, best_model, evaluator)
            zoomed_obj_matches_imgs = check_matches_consistency(zoomed_obj_imgs, zoomed_obj_masks, best_model, evaluator)

            cv.namedWindow("Obj matches", cv.WINDOW_NORMAL)
            cv.namedWindow("Zoomed-in Obj matches", cv.WINDOW_NORMAL)
            cv.imshow("Obj matches", obj_matches_imgs)
            cv.imshow("Zoomed-in Obj matches", zoomed_obj_matches_imgs)
            key = cv.waitKey(0)
            if key == ord('q'):
                exit()
            elif key == ord(' '):
                break