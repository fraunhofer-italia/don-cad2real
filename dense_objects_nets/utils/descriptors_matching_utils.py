# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import torch
import torch.nn.functional as F


def find_descriptors_matches_L2(img1_pred: torch.Tensor, img2_pred: torch.Tensor, img1_pixels: torch.Tensor) -> torch.Tensor:
    """
    Find descriptors matches between the two input predictions images using the L2 distance.
    
    Pairs are returned as a tensor of predicted matches of shape Nx4.  
    
    Args:
    - img1_pred (`torch.tensor`): tensor of shape DHW containing the model descriptors prediction on the first image.
    - img2_pred (`torch.tensor`): tensor of shape DHW containing the model descriptors prediction on the second image.
    - img1_pixels (`torch.tensor`): tensor of shape Nx2 containing the pixels positions of the img1 descriptors that we want to match.

    Returns:
    - `torch.tensor`: (N, 4) tensor containing the model predicted pixels matches for the current pair of images.
    """
    D, H, W = img1_pred.size()

    # Select descriptors associated with the input img1 pixels.
    img1_descriptors = img1_pred[:, img1_pixels[:, 1], img1_pixels[:, 0]]
    
    # Permute from (D, N) to (N, D).
    img1_descriptors = img1_descriptors.permute(1, 0)

    # Reshape img2 descriptors from (D, H, W) to (H*W, D).
    img2_descriptors = img2_pred.view(D, -1).permute(1, 0)

    # Compute the L2 distance between each img1 descriptor and all the img2 descriptors.
    descriptors_distances = torch.cdist(img1_descriptors, img2_descriptors, p=2) # Shape: (N, H*W)
    
    # Find the indices of the MINIMUM values along each row (i.e., the closest descriptors to each img1_descriptor).
    min_indices = torch.argmin(descriptors_distances, dim=1)

    # Convert the flat indices to 2D coordinates.
    d_v, d_u = torch.unravel_index(min_indices, (H, W)) # Two tensors of shape (N, ).

    predicted_matches = torch.stack([img1_pixels[:, 0], img1_pixels[:, 1], d_u, d_v], dim=1)

    return predicted_matches

def find_descriptors_matches_cosine(img1_pred: torch.Tensor, img2_pred: torch.Tensor, img1_pixels: torch.Tensor) -> torch.Tensor:
    """
    Find descriptors matches between the two input predictions images using the cosine similarity.
    
    Pairs are returned as a tensor of predicted matches of shape Nx4.  
    
    Args:
    - img1_pred (`torch.tensor`): tensor of shape DHW containing the model descriptors prediction on the first image.
    - img2_pred (`torch.tensor`): tensor of shape DHW containing the model descriptors prediction on the second image.
    - img1_pixels (`torch.tensor`): tensor of shape Nx2 containing the pixels positions of the img1 descriptors that we want to match.

    Returns:
    - `torch.tensor`: (N, 4) tensor containing the model predicted pixels matches for the current pair of images.
    """
    D, H, W = img1_pred.size()

    # Select descriptors associate with the input img1 pixels.
    img1_descriptors = img1_pred[:, img1_pixels[:, 1], img1_pixels[:, 0]]
    
    # Permute from (D, N) to (N, D) and normalize them.
    img1_descriptors = img1_descriptors.permute(1, 0)
    img1_descriptors = F.normalize(img1_descriptors, p=2, dim=1)

    # Reshape img2 descriptors from (D, H, W) to (H*W, D) and normalize them.
    img2_descriptors = img2_pred.reshape(D, -1).permute(1, 0)
    img2_descriptors = F.normalize(img2_descriptors, p=2, dim=1)

    # Compute the cosine similarity between each img1 descriptor and all the img2 descriptors.
    descriptors_distances = torch.mm(img1_descriptors, img2_descriptors.T) # Shape: (N, H*W)
    
    # Find the indices of the MAXIMUM values along each row (i.e., the closest descriptors to each img1_descriptor).
    # MAXIMUM since closer descriptors have higher cosine similarity.
    min_indices = torch.argmax(descriptors_distances, dim=1)

    # Convert the flat indices to 2D coordinates.
    d_v, d_u = torch.unravel_index(min_indices, (H, W)) # Two tensors of shape (N, ).

    predicted_matches = torch.stack([img1_pixels[:, 0], img1_pixels[:, 1], d_u, d_v], dim=1)

    return predicted_matches

def find_patches_matches(img1_pred: torch.Tensor, img2_pred: torch.Tensor, img1_pixels: torch.Tensor, K: int = 5, S: int = 2) -> torch.Tensor:
    """
    For each pixels in `img1_pixels` extract from `img1_pred` a `KxK` descriptors patch containing the indexed pixels
    and pair each patch with the closest`img_2_pred` `KxK` descriptors patch. 
    
    Pairs are returned as a tensor of predicted matches of shape Nx4. For each matched patch in `img2_pred` its center is returned.  
    
    Args:
    - img1_pred (`torch.tensor`): tensor of shape DHW containing the model descriptors prediction on the first image.
    - img2_pred (`torch.tensor`): tensor of shape DHW containing the model descriptors prediction on the second image.
    - img1_pixels (`torch.tensor`): tensor of shape Nx2 containing the pixels positions of the img1 descriptors that we want to match.
    - K (`int`, Optional): size of the patch kernel. Default=5.
    - S (`int`, Optional): sliding window stride. Default=2.

    Returns:
    - `torch.tensor`: tensor containing the model predicted pixels matches for the current pair of images.
    """
    D, H, W = img1_pred.size()
    N = len(img1_pixels)

    device = img1_pred.device

    # Define the dimensions of the output tensor obtained by sliding over the input H*W image a kernel of shape K*K with stride S.
    K_W = ((W - K) // S) + 1 # = how many sliding windows i can fit horizontally (patches per row).
    K_H = ((H - K) // S) + 1 # = how many sliding windows i can fit vertically (patches per column).

    # Compute the unfolded kernel patch index inside the first image. (Indexes of the patches in img1_pred containing the pixels in 'img1_pixels')
    # Rows and columns indexes are clamped to number of patches in each row/columns. 
    kernel_cols_indexes = (img1_pixels[:, 0] - K // 2) // S
    kernel_rows_indexes = (img1_pixels[:, 1] - K // 2) // S

    kernel_cols_indexes = torch.clamp(kernel_cols_indexes, min=0, max=K_W-1)
    kernel_rows_indexes = torch.clamp(kernel_rows_indexes, min=0, max=K_H-1)

    kernel_indexes = (kernel_rows_indexes * K_W) + kernel_cols_indexes

    # Extracts the kernels from the first image.
    img1_pred_unfolded = F.unfold(img1_pred.unsqueeze(0), kernel_size=(K, K), stride=S).squeeze(0) # Shape: (D*K*K, K_H*K_W)
    kernels_flattened = img1_pred_unfolded[:, kernel_indexes].permute(1, 0)

    # Create a sliding window view of 'img2_pred'.
    img2_pred_unfolded = F.unfold(img2_pred.unsqueeze(0), kernel_size=(K, K), stride=S).squeeze(0) # Shape: (D*K*K, K_H*K_W)

    # Compute the distance between each img1_pred kernel patch and all the img2_pred sliding window patches.
    # Note: this operation "broadcasts vertically" the img2_pred_unfolded N times (one copy for each img1 kernel). Large values of N may lead to memory problems.
    patches_distances = torch.linalg.vector_norm(img2_pred_unfolded[None, :, :] - kernels_flattened[:, :, None], dim=1) # Shape: (N, K_H*K_W) 

    # Compute the pixels positions inside the "convoluted" K_H*K_W image of the patches closest to the input kernels. 
    closest_patches_pixels_position = torch.unravel_index(torch.argmin(patches_distances, dim=1), (K_H, K_W)) # Shape: (N, 2)
    closest_patches_pixels_position = torch.stack(closest_patches_pixels_position, dim=1).view(N, 2) # 'torch.unravel_index'returns a tuple of tensors.

    # Map the pixels coordinates positions from the "convoluted" K_H*K_W image to the original H*W image.
    closest_patches_pixels_position = (closest_patches_pixels_position * S) + K//2 # Shape: (N, 2)

    # Clamp values outside of the original image dimensions.
    min_pos, max_pos = torch.tensor([[0, 0]], device=device), torch.tensor([[H, W]], device=device)
    matches_pixels = torch.clamp(closest_patches_pixels_position, min=min_pos, max=max_pos).int()

    predicted_matches = torch.stack([img1_pixels[:, 0], img1_pixels[:, 1], matches_pixels[:, 1], matches_pixels[:, 0]], dim=1)

    return predicted_matches
