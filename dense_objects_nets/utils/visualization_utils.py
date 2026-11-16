# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

from matplotlib.colors import to_rgb
from typing import Any, Dict, List

import open3d as o3d
import numpy as np
import torch
import cv2 as cv

# List of 55 visually distinctive colors.
colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#000000", "#800000", "#008000", "#000080", "#808000",
        "#800080", "#008080", "#808080", "#C00000", "#00C000", "#0000C0", "#C0C000", "#C000C0", "#00C0C0", "#C0C0C0", "#400000",
        "#004000", "#000040", "#404000", "#400040", "#004040", "#404040", "#200000", "#002000", "#000020", "#202000", "#200020",
        "#002020", "#202020", "#600000", "#006000", "#000060", "#606000", "#600060", "#006060", "#606060", "#A00000", "#00A000",
        "#0000A0", "#A0A000", "#A000A0", "#00A0A0", "#A0A0A0", "#E00000", "#00E000", "#0000E0", "#E0E000", "#E000E0", "#00E0E0",
]


def plot_predictions(
    rgb_images: List[np.ndarray],
    predictions_images: List[np.ndarray],
    images_masks: List[np.ndarray] = None,
) -> np.ndarray:
    """
    Create an "predictions image" by concatenating vertically three images:
    + the "original img": obtained by the horizontal concatenation of the input rgb images.
    + the "descriptors img": obtained by the horizontal concatenation of the descriptors images obtained by rendering the input predictions images to RGB.
    + the "overlay img": obtained by the weighted overlay of the "descriptors img" on top of the "original img".

    Assuming the input images to be N `torch.tensors` of shape HW, the produced image hence is an RGB image of shape (3HxNW).

    Args:
    - rgb_images (`list`): list of one or more `torch.tensor` of shape HWD containing the input RGB images.
    - predictions_images (`list`): list of one or more `torch.tensor` of shape HWD containing the model descriptors predictions
    for the input images that we want to render.
    - images_masks (`list`, optional): list of 'binary masks' (HW 0-1 `torch.tensor`) delimiting the objects in the input images.
    If provided, for each image, only descriptors within this masks are plotted. Must have the same length as `rgb_images` and `predictions_images`.

    Returns:
    - `numpy.ndarray`: the produced "predictions image" returned in OpenCv format.
    """
    # Permute images from (D, H, W) to (H, W, D) to help with descriptors visualization. (We use OpenCV)
    for i, (img, img_pred) in enumerate(zip(rgb_images, predictions_images)):
        rgb_images[i] = img.permute(1, 2, 0)
        predictions_images[i] = img_pred.permute(1, 2, 0)

    # Create and image from the model descriptors of the two input images.
    descriptors_img = render_descriptors_image(
        predictions_images=predictions_images, images_masks=images_masks
    )

    # Move all data to CPU and NumPy and convert imgs from RGB to BGR to work with OpenCV.
    descriptors_img = descriptors_img.cpu().numpy()
    for i, img in enumerate(rgb_images):
        img = img.cpu().numpy()
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        rgb_images[i] = img

    # Concatenate the original rgb images horizontally.
    original_img = cv.hconcat(rgb_images)

    # Overlay the original rgb image over the descriptors image.
    overlay_img = cv.addWeighted(descriptors_img, 1, original_img, 0.3, 0)

    # Concatenate the three images vertically.
    out_img = cv.vconcat([original_img, descriptors_img, overlay_img])

    return out_img


def render_descriptors_image(
        predictions_images: List[torch.Tensor],
        images_masks: List[torch.Tensor] = None
    ) -> torch.Tensor:
    """
    Converts the input model descriptors tensors to 3 channels RGB visible images then normalize them and concatenate them together.  

    Args:
    - predictions_images (`list`): list of one or more `torch.tensor` of shape HWD containing the model descriptors predictions
    for the input images that we want to render.
    - images_masks (`list`, optional): list of 'binary masks' (HW 0-1 `torch.tensor`) delimiting the objects in the input images.
    If provided, for each image, only descriptors within this masks are plotted. Must have the same length as `predictions_images`.
    
    Returns:
    - `torch.tensor`: torch tensor image obtained concatenating horizontally the computed descriptors images.
    """
    H, W, D = predictions_images[-1].shape

    # If the dimensionality of output images descriptors is grater than 3 reduce it using PCA.
    if D > 3:
        for i, img_pred in enumerate(predictions_images):
            img_pred_flattened = img_pred.reshape((H * W, D))

            # 'torch.pca_lowrank' requires an input matrix A of shape NxD where N is the total number of samples.
            U, S, V = torch.pca_lowrank(img_pred_flattened)

            # Project the outputs using the first 3 principal components.
            img_pred_lowrank = torch.matmul(img_pred_flattened, V[:, :3]).reshape((H, W, 3))

            # Save the reduced image.
            predictions_images[i] = img_pred_lowrank
        
    # If the dimensionality of the output images descriptors is less than 3, we need to pad them to 3 channels.
    if D < 3:
        for i, img_pred in enumerate(predictions_images):
            img_pred = torch.nn.functional.pad(img_pred, (0, 3 - D), mode='constant', value=0)
            predictions_images[i] = img_pred

    # If objects masks are provided plot only the objects descriptors otherwise plot the whole images.
    if images_masks != None:
        # Extract only object descriptors from each image.
        object_pixels = []
        object_descriptors = []
        for img_pred, mask in zip(predictions_images, images_masks):
            # Extract object predicted descriptors.
            img_object_pixels = torch.nonzero(mask)
            img_object_descriptors = img_pred[img_object_pixels[:, 0], img_object_pixels[:, 1]]

            # Save all the object pixels and object descriptors for the input images.
            object_pixels.append(img_object_pixels) 
            object_descriptors.append(img_object_descriptors) 

        # Compute min and max value for each of the 3 descriptors channels.
        #
        # 'descriptors_stack' is a tensor of shape Mx3 where M is the total number of object pixels descriptors from the input images.
        # 'torch.amin/amax(descriptors_stack, dim=0)' extracts the min/max value for each of the 3 channels of the 'descriptors stack' tensor.
        #
        # 'object_descriptors_min/max' are hence tensor of shape 1x3 containing the min/max values for each channel.
        descriptors_stack = torch.vstack(object_descriptors)
        object_descriptors_min = torch.amin(descriptors_stack, dim=0)
        object_descriptors_max = torch.amax(descriptors_stack, dim=0)

        # Min-Max normalize descriptors channels, convert them to rgb values and concatenate all output images horizontally.
        # Values are then rescaled in [0.2, 0.9] to distinguish them from the black background.
        descriptors_images = []
        for img_object_pixels, img_object_descriptors in zip(object_pixels, object_descriptors):
            img_object_descriptors = img_object_descriptors - object_descriptors_min
            img_object_descriptors = img_object_descriptors / (object_descriptors_max - object_descriptors_min)
            img_object_descriptors = (img_object_descriptors * (0.9 - 0.2))  + 0.2

            # Paste normalized descriptors over a black background (img2 pixels are shifted to account for the concatenation).
            descriptors_img = torch.zeros((H, W, 3), device=img_object_descriptors.device)
            descriptors_img[img_object_pixels[:, 0], img_object_pixels[:, 1]] = img_object_descriptors

            descriptors_images.append(descriptors_img)

        # Concatenate images horizontally.
        descriptors_images = torch.cat(descriptors_images, dim=1)

    # Plot the whole images.
    else:
        # Concatenate all flattened images pixels descriptors.
        descriptors_stack = torch.cat([img_pred for img_pred in predictions_images], dim=1)

        # Compute min and max value for each descriptors channel.
        img_descriptors_min = torch.amin(descriptors_stack, dim=(1, 0))
        img_descriptors_max = torch.amax(descriptors_stack, dim=(1, 0))

        # Min-Max normalize descriptors channels for the whole images.
        # Values are then rescaled in [0.2, 0.9] to avoid solid white/black colors.
        descriptors_images = torch.cat(predictions_images, dim=1)
        descriptors_images = descriptors_images - img_descriptors_min
        descriptors_images = descriptors_images / (img_descriptors_max - img_descriptors_min)

    return descriptors_images.detach()


def draw_matches(
    img,
    img1_pixels: np.ndarray,
    img2_pixels: np.ndarray,
    matches_colors: List[str],
    marker: int = cv.MARKER_CROSS,
):
    """
    Draw a "match" connecting the corresponding pairs of input Img1-Img2 pixels position.

    The given input image is expected to be the horizontal concatenation of the two images associated with the input matches.
    Given pixels positions are already expected to be shifted to compensate for the concatenation of the two images:

    "Matches" are represented by a line connecting the pairs of Img1-Img2 pixels.
    Pixels on Img2 are also marked using the given OpenCv marker type.

    Args:
    - img (`numpy.ndarray`): input image where matches will be drawn.
    - img1_pixels (`numpy.ndarray`): array of shape Nx2 containing the img1 pixels position that we want to draw as matches.
    - img2_pixels (`numpy.ndarray`): array of shape Nx2 containing the img2 pixels position that we want to pair with the above.
    - matches_colors (`list`): list of N RGB Hex codes defining the colors used to plot each match.
    - marker (`cv2.MarkerTypes`): OpenCv marker used to mark the img2 pixels positions.

    Returns:
    - `numpy.ndarray`: output image with the matches drawn on top.
    """
    for (u1, v1), (u2, v2), hex_color in zip(img1_pixels, img2_pixels, matches_colors):
        color = to_rgb(hex_color)

        # Draw a marker and a line for the real match and the predicted match
        img = cv.drawMarker(img, (u2, v2), color=color, markerType=marker, thickness=2)
        img = cv.line(img, (u1, v1), (u2, v2), color=color, thickness=1)

    return img


def print_params(
    img : np.ndarray,
    training_params : Dict[str, Any],
    text_margin : int = 10,
    box_width : int = 210,
    box_height : int = 65,
    box_margin : int = 10,
    lines_offset : int = 25,
):
    """
    Print the main training parameters information over the input image.

    Args:
    - img (`numpy.ndarray`): input image where text will be overlaid.
    - training_params (`dict`): dict containing the values for the training that will be printed over the image.
    - text_margin (`int`, optional): margin of the text lines from the borders of the background white box. Defaults to 10 px.
    - box_width (`int`, optional): white background box width. Defaults to 275 px.
    - box_height (`int`, optional): white background box height. Defaults to 70 px.
    - box_margin (`int`, optional): white background box margin from the borders of the image. Defaults to 10 px.
    - lines_offset (`int`, optional): height pixels offset between two text lines. Defaults to 25 px.

    Returns:
    - `numpy.ndarray`: output image with the text overlay printed on top.
    """
    img_height, img_width, _ = img.shape
    box_top_left = (box_margin, img_height - box_height - box_margin)
    box_bottom_right= (box_margin + box_width, img_height - box_margin)

    # Draw white background box.
    img = cv.rectangle(img, pt1=box_top_left, pt2=box_bottom_right, color=(255, 255, 255), thickness=cv.FILLED)

    # Log main training parameters over the image.
    overlay_text = [
        f"D: {training_params['network']['descriptors_dim']}",
        f"LR: {training_params['optimizer']['learning_rate']}",
    ]

    for i, line in enumerate(reversed(overlay_text)):
        text_orig = (
            box_margin + text_margin,
            img_height - box_margin - text_margin - (lines_offset * i),
        )
        img = cv.putText(
            img,
            text=line,
            org=text_orig,
            fontFace=cv.FONT_HERSHEY_PLAIN,
            fontScale=1.8,
            thickness=2,
            color=(0, 0, 0),
        )
    return img


def project_points_onto_image(
    points: np.ndarray,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    camera_matrix: np.ndarray,
    image: np.ndarray,
) -> np.ndarray:
    """
    Projects an array of 3D points onto a given 2D image.

    Args:
    - points (`np.ndarray`): The 3D point of the object.
    - rotation_matrix (`np.ndarray`): 3x3 rotation matrix of the object w.r.t. the camera.
    - translation_vector (`np.ndarray`): 3x1 translation vector of the object w.r.t. the camera.
    - camera_matrix (`np.ndarray`): 3x3 intrinsic camera matrix.
    - image (`np.ndarray`): The image on which to project the points.

    Returns:
    - `np.ndarray`: The image with the projected point cloud overlaid.
    """

    # Transform the points into the camera frame
    points_camera_frame = (rotation_matrix @ points.T).T + translation_vector.T

    # Extract only the points in front of the camera
    points_camera_frame = points_camera_frame[points_camera_frame[:, 2] > 0]

    # Project the 3D points into the 2D image plane
    image_points, _ = cv.projectPoints(
        points_camera_frame,
        rvec=np.zeros(3),  # No additional rotation
        tvec=np.zeros(3),  # No additional translation
        cameraMatrix=camera_matrix,
        distCoeffs=np.zeros(5),  # No distortion
    )
    image_points = np.squeeze(image_points).astype(int)

    # Overlay the projected points on the provided image
    H, W = image.shape[:2]
    points_image = np.zeros_like(image)
    valid_points = (image_points[:, 0] >= 0) & (image_points[:, 0] < W) & (image_points[:, 1] >= 0) & (image_points[:, 1] < H)
    points_image[image_points[valid_points, 1], image_points[valid_points, 0]] = (0, 255, 0)

    out_img = cv.addWeighted(image, 1, points_image, 0.2, 0)

    return out_img


def draw_axis(
    img: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    length: float = 0.05,
    thickness: int = 5
) -> np.ndarray:
    """
    Draws a 3D coordinate axis on a 2D image.

    Args:
    - img (`np.ndarray`): The input image on which the axis will be drawn.
    - rvec (`np.ndarray`): Rotation vector (Rodrigues) that describes the orientation of the object.
    - tvec (`np.ndarray`): Translation vector that describes the position of the object.
    - K (`np.ndarray`): Camera matrix (intrinsic parameters).
    - length (`float`, optional): Length of the axis lines. Default is 0.05.

    Returns:
    - `np.ndarray`: The output image with the drawn axis.
    """
    # Draw the coordinate frame on the image
    axis_points = np.float32(
        [[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]]
    ).reshape(-1, 3)

    # Project the 3D axis points to 2D image points
    image_points, _ = cv.projectPoints(axis_points, rvec, tvec, K, distCoeffs=None)

    # Draw the axis lines on the image
    image_points = np.int32(image_points).reshape(-1, 2)
    output_image = img.copy()
    cv.line(
        output_image,
        tuple(image_points[0]),
        tuple(image_points[1]),
        (0, 0, 255),
        thickness,
    )  # X-axis in red
    cv.line(
        output_image,
        tuple(image_points[0]),
        tuple(image_points[2]),
        (0, 255, 0),
        thickness,
    )  # Y-axis in green
    cv.line(
        output_image,
        tuple(image_points[0]),
        tuple(image_points[3]),
        (255, 0, 0),
        thickness,
    )  # Z-axis in blue

    return output_image


def draw_bbox(
    img: np.ndarray,
    obj_dims: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    thickness: int = 1
) -> np.ndarray:
    """
    Draws a 3D bounding box on the image.

    Args:
    - img (`np.ndarray`): The input image on which the bounding box will be drawn.
    - rvec (`np.ndarray`): Rotation vector (Rodrigues) that describes the orientation of the object.
    - tvec (`np.ndarray`): Translation vector that describes the position of the object.
    - K (`np.ndarray`): Camera matrix (intrinsic parameters).
    - length (`float`, optional): Length of the axis lines. Default is 0.05.

    Returns:
    - `np.ndarray`: The output image with the drawn axis.
    """
    # Draw the coordinate frame on the image
    x_length, y_length, z_length = obj_dims
    bbox_points = np.float32([
        [0, 0, 0], [x_length, 0, 0], [x_length, y_length, 0], [0, y_length, 0], 
        [0, 0, z_length], [x_length, 0, z_length], [x_length, y_length, z_length], [0, y_length, z_length]
    ]).reshape(-1, 3)

    # Shift all the points from the center of the bbox to the bottom left corner
    bbox_points -= np.array([x_length / 2, y_length / 2, z_length / 2])

    # Project the 3D bounding box points to 2D image points
    image_points, _ = cv.projectPoints(bbox_points, rvec, tvec, K, distCoeffs=None)

    # Draw the bounding box lines on the image
    image_points = np.int32(image_points).reshape(-1, 2)
    output_image = img.copy()
    cv.line(
        output_image,
        tuple(image_points[0]),
        tuple(image_points[1]),
        (0, 0, 255),
        thickness,
    )  # X-axis in red
    cv.line(
        output_image,
        tuple(image_points[0]),
        tuple(image_points[3]),
        (0, 255, 0),
        thickness,
    )  # Y-axis in green
    cv.line(
        output_image,
        tuple(image_points[0]),
        tuple(image_points[4]),
        (255, 0, 0),
        thickness,
    )  # Z-axis in blue

    # Draw all the other edges of the bounding box in gray
    edges = [
        (1, 2), (2, 3), (1, 5), (2, 6), (3, 7),
        (4, 5), (5, 6), (6, 7), (4, 7)
    ]

    for edge in edges:
        cv.line(
            output_image,
            tuple(image_points[edge[0]]),
            tuple(image_points[edge[1]]),
            (125, 125, 125) if output_image.dtype == np.uint8 else (0.4, 0.4, 0.4),
            thickness,
        )

    return output_image


def draw_pixels(
    image: np.ndarray, 
    pixels: np.ndarray, 
    color=(0, 0, 255), radius=3
) -> np.ndarray:
    """
    Draws circles on the given image at the specified pixel locations.

    Parameters:
    - image (`np.ndarray`): The image on which to draw the pixels.
    - pixels (`np.ndarray`): An array of pixel coordinates where circles will be drawn.
    - color (`tuple`, optional): The color of the circles in BGR format. Default is (0, 0, 255) which is red.
    - radius (`int`, optional): The radius of the circles to be drawn. Default is 3.

    Returns:
    - `np.ndarray`: The image with the drawn circles.
    """
    for pixel in pixels:
        cv.circle(
            image,
            (int(pixel[0]), int(pixel[1])),
            radius=radius,
            color=color,
            thickness=-1,
        )
    return image
