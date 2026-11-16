# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import torch
import numpy as np

from typing import Tuple
from torchvision.transforms import v2

from dense_objects_nets.network.don_backbone import DonNetwork
from dense_objects_nets.utils.evaluation_utils import load_class
import dense_objects_nets.utils.visualization_utils as visualization_utils

class DonPredictor(object):

    def __init__(self, model_weights_path: str):
        """
        Class used to compute the predictions of a pretrained `DonNetwork` model on input RGB images.

        Args:
        - model_weights_path (`str`): path to the `state_dict` object that contains the pretrained DON model weights.
        """
        # Setup the training device.
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Load the model class from the checkpoint.
        model_checkpoint = torch.load(model_weights_path, map_location=device)
        model_class: DonNetwork = load_class(model_checkpoint['name'])

        # Load the pretrained model.
        self.don_model = model_class.from_checkpoint(model_weights_path, device)
        self.don_model.eval()

    def compute_predictions(self, img: torch.Tensor | np.ndarray, return_numpy: bool = True) -> Tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray] :
        """
        Computes the model prediction on the input RGB image.
        Returns both the model original HWD output descriptors and a HW3 'descriptors_image' obtained by rendering the
        model predictions to RGB values.  

        Args:
        - img (`MatLike`): the input image. Either a `torch.tensor` of shape 3HW or a `numpy.ndarray` of shape HW3.
        - return_numpy (`bool`): if `True`, the output images are returned as `numpy.ndarray`, otherwise as `torch.tensor`.

        Returns:
        - `tuple`: a tuple containing two `numpy.ndarray`: the first a HWD array containing the model original output
        predictions/descriptors, the second a HW3 RGB 'descriptors_image' obtained by rendering in RGB the above model predictions. 
        """
        # Convert the image to PyTorch Image tensor.
        img_torch = v2.ToImage()(img)
        img_torch = v2.ToDtype(torch.float32, scale=True)(img_torch)

        # Add extra channel (batch num) to the image. (Required by the network)
        img_torch = img_torch.unsqueeze(0)

        # Move tensor to model device.
        img_torch = img_torch.to(self.don_model.device)

        # Forward pass.
        with torch.no_grad():
            img_pred = self.don_model.forward(img_torch)

        # Remove batch channel from the images.
        img_torch = img_torch.squeeze(0)
        img_pred = img_pred.squeeze(0)

        # Move images from DHW to HWD (as required by the visualization functions).
        img_pred = img_pred.permute(1, 2, 0)

        # Compute a 'descriptors image' for the current model prediction.
        descriptors_img = visualization_utils.render_descriptors_image(predictions_images=[img_pred])

        # If required convert output images to numpy.
        if return_numpy:
            img_pred = img_pred.cpu().numpy()
            descriptors_img = descriptors_img.cpu().numpy()

        return img_pred, descriptors_img

    @property
    def device(self) -> torch.device:
        return self.don_model.device
    
    @property
    def prediction_shape(self) -> Tuple[int, int, int]:
        _, D, H ,W = self.don_model.output_shape
        return H, W, D