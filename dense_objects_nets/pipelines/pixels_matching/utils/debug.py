# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import torch

import cv2 as cv
import numpy as np
import matplotlib

from dense_objects_nets.utils.visualization_utils import draw_matches, colors

# 30 unique colors for objects visualization
unique_colors = [
    matplotlib.colors.to_rgb(color) 
    for color in list(matplotlib.colormaps['Set3'].colors + matplotlib.colormaps['tab10'].colors + matplotlib.colormaps['Pastel2'].colors)
]

WAIT_KEY = 1  # 1 to step, 0 to pause


@torch.no_grad()
def check_pixels_matches(train_images, pixels_matches, bg_pixels, num_points=5):
    global WAIT_KEY

    img1 = train_images[0].permute(1, 2, 0).cpu().numpy()
    img2 = train_images[1].permute(1, 2, 0).cpu().numpy()
    img3 = train_images[2].permute(1, 2, 0).cpu().numpy()
    img1 = cv.cvtColor(img1, cv.COLOR_RGB2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_RGB2BGR)
    img3 = cv.cvtColor(img3, cv.COLOR_RGB2BGR)

    H, W = img1.shape[:2]

    # Visualize sampled object and background points
    samples_img = np.hstack([img1, img2, img3])
    img1_pixels = pixels_matches[:, :2].cpu().numpy()
    img2_pixels = pixels_matches[:, 2:].cpu().numpy()
    for (u1, v1), (u2, v2) in zip(img1_pixels, img2_pixels):
        samples_img[v1, u1] = (0, 255, 0)
        samples_img[v2, u2 + W] = (0, 255, 0)

    bg_pixels_img1 = bg_pixels[0].cpu().numpy()
    bg_pixels_img2 = bg_pixels[1].cpu().numpy()
    bg_pixels_img3 = bg_pixels[2].cpu().numpy()
    for (u1, v1), (u2, v2), (u3, v3) in zip(bg_pixels_img1, bg_pixels_img2, bg_pixels_img3):
        samples_img[v1, u1] = (0, 0, 255)
        samples_img[v2, u2 + W] = (0, 0, 255)
        samples_img[v3, u3 + 2 * W] = (0, 0, 255)
    
    # Visualize some random matches
    matches_img = np.hstack([img1, img2])
    
    sampled_indices = np.random.choice(len(img1_pixels), size=min(num_points, len(img1_pixels)), replace=False)
    img1_pixels = img1_pixels[sampled_indices]
    img2_pixels = img2_pixels[sampled_indices]
    matches_img = draw_matches(
        matches_img, img1_pixels, (img2_pixels + np.array([W, 0])), matches_colors=colors[: len(sampled_indices)],
    )

    print()
    print(f"Num matches: {len(pixels_matches)}")
    print(f"Num background: {len(bg_pixels_img1) + len(bg_pixels_img2) + len(bg_pixels_img3)}")
    cv.namedWindow("Matches", cv.WINDOW_NORMAL)
    cv.imshow("Matches", np.vstack([samples_img, np.hstack([matches_img, np.zeros_like(img3)])]))
    if cv.waitKey(WAIT_KEY) == ord(' '):
        WAIT_KEY = int(not WAIT_KEY)  # Toggle wait key from 1 to 0 and vice versa


@torch.no_grad()
def check_centers_loss_masks(train_images, objects_masks, objects_ids):
    global WAIT_KEY

    img1 = train_images[0].permute(1, 2, 0)
    img2 = train_images[1].permute(1, 2, 0)
    img3 = train_images[2].permute(1, 2, 0)

    # Reshape masks from flat to HxW
    H, W = img1.shape[:2]
    objects_masks = objects_masks.view(objects_masks.size(0), 3, H, W)  # (num_objects, 3, H, W)
    objects_masks = objects_masks.cpu().numpy()

    masks_overlay = np.zeros((H, 3*W, 3), dtype=np.float32)
    for obj_id, (m1, m2, m3) in zip(objects_ids, objects_masks):
        mask = np.hstack([m1, m2, m3])
        color = unique_colors[obj_id % len(unique_colors)]
        masks_overlay[mask] = color

    img = torch.hstack([img1, img2, img3]).cpu().numpy()
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    img_overlay = cv.addWeighted(img, 0.5, masks_overlay, 0.5, 0)

    cv.namedWindow("Objects Masks", cv.WINDOW_NORMAL)
    cv.imshow("Objects Masks", np.vstack([img_overlay, masks_overlay]))
    if cv.waitKey(WAIT_KEY) == ord(' '):
        WAIT_KEY = int(not WAIT_KEY)  # Toggle wait key from 1 to 0 and vice versa


@torch.no_grad()
def check_dann_loss_descriptors(train_images, dann_descriptors_mask, dann_labels):
    global WAIT_KEY

    img1 = train_images[0].permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
    img2 = train_images[1].permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
    img3 = train_images[2].permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
    imgs = torch.cat([img1, img2, img3], dim=0)  # (3*H*W, 3)

    real_labels_mask = dann_descriptors_mask[dann_labels == 0]
    synth_labels_mask = dann_descriptors_mask[dann_labels == 1]
    imgs[real_labels_mask] = torch.tensor([1.0, 0.0, 0.0], device=imgs.device)  # Red for real
    imgs[synth_labels_mask] = torch.tensor([0.0, 0.0, 1.0], device=imgs.device)  # Blue for sim

    H, W = train_images.shape[2:4]
    imgs = imgs.reshape(3, H, W, 3).cpu().numpy()  # (3, H, W, 3)
    imgs = np.hstack(imgs) # (H, 3*W, 3)
    imgs = cv.cvtColor(imgs, cv.COLOR_RGB2BGR)
    
    print(f"Num DANN real pixels: {len(real_labels_mask)} / {len(dann_labels)}")
    print(f"Num DANN sim pixels: {len(synth_labels_mask)} / {len(dann_labels)}")
    cv.namedWindow("DANN Domain Labels (Real=Red, Sim=Blue)", cv.WINDOW_NORMAL)
    cv.imshow("DANN Domain Labels (Real=Red, Sim=Blue)", imgs)
    if cv.waitKey(WAIT_KEY) == ord(' '):
        WAIT_KEY = int(not WAIT_KEY)  # Toggle wait key from 1 to 0 and vice versa