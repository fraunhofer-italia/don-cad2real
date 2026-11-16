# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

# GUI functions must be imported before any 'cv2' import to avoid conflicts with PyQt5
from dense_objects_nets.utils.gui_utils import select_files, select_runs_folders

import os
import torch
import argparse
import cv2 as cv

from copy import deepcopy
from dense_objects_nets.utils import io_utils, visualization_utils, evaluation_utils
from dense_objects_nets.utils.don_snapshot import DonSnapshot

# Parameters needed by the callback function
H, W = 0, 0
win_name = None
evaluator = None
best_model_pred_img1, best_model_pred_img2 = None, None
last_model_pred_img1, last_model_pred_img2 = None, None
out_img, original_prediction_img = None, None


def match_clicked_descriptors(event, x, y, flags, param):
    """
    OpenCv window event callback function.
    
    Listen for double left clicks on the rendered windows and, starting from the registered pixel position, computes and
    visualizes the associated "best model" and "last model" predicted matches.
    
    Matches are only plotted when clicking on a point in "Img1", either for the "best model" or for the "last model".

    """
    if event == cv.EVENT_LBUTTONDBLCLK:
        # Check that we clicked on img1 either on the "best model" or in the "last model" and adjust the pixel position.
        if (x < W): u1 = x                            # Clicked on "best model img1" -> no horizontal adjustment needed.                        
        elif (x >= 2*W and x < 3*W): u1 = x - 2*W     # Clicked on "best model img1" -> need to account for images concatenation.
        else: return                                  # Clicked on "best model img2" or "last model img2" -> invalid pixel position.

        # Adjust the vertical pixel position to account for the images concatenation.
        if (y >= 2*H): v1 = y - 2*H     # Clicked on the bottom of the three images. ((0,0) is the top-left corner)
        elif (y >= H): v1 = y - H       # Clicked on teh middle image.
        else: v1 = y                    # Clicked on the top image.

        # Adjusted clicked img1 pixel position.
        img1_clicked_pixel = torch.tensor([[u1, v1]], device=best_model_pred_img1.device)

        # Find the descriptor match for each of the two model.
        bm_match = evaluator.find_descriptors_matches(best_model_pred_img1, best_model_pred_img2, img1_clicked_pixel)        
        lm_match = evaluator.find_descriptors_matches(last_model_pred_img1, last_model_pred_img2, img1_clicked_pixel)

        # Deepcopy the original pred img to avoid printing all the matches on top of each other. 
        out_img = deepcopy(original_prediction_img)

        # Draw the computed descriptor matches for both the "best" and the "last" model.
        for i in range(0, 3): # First iteration draws matches for the "original" img, then "descriptors" and finally for the "overlay" one.
            shifted_bm_match = bm_match.cpu().numpy() + [[0, i*H, W, i*H]]
            shifted_lm_match = lm_match.cpu().numpy() + [[2*W, i*H, 3*W, i*H]]
                    
            # Draw the "best model" matches.
            out_img = visualization_utils.draw_matches(
                out_img,
                img1_pixels=shifted_bm_match[:, :2],
                img2_pixels=shifted_bm_match[:, 2:],
                matches_colors=["#FF00FF"], # Pink.
                marker=cv.MARKER_CROSS
            )
            
            # Draw the "last model" matches.
            out_img = visualization_utils.draw_matches(
                out_img,
                img1_pixels=shifted_lm_match[:, :2],
                img2_pixels=shifted_lm_match[:, 2:],
                matches_colors=["#00FF00"], # Green.
                marker=cv.MARKER_STAR
            )
                
        # Display the updated image.
        cv.imshow(win_name, out_img)

def compute_and_plot_predictions(model, data, plot_object_only=False):
    s1 = DonSnapshot.from_dict(data["s1"])
    s2 = DonSnapshot.from_dict(data["s2"])
    s1.to(model.device)
    s2.to(model.device)
    
    # Get the model prediction on the two (augmented) images.
    with torch.no_grad():
        img1_pred, img2_pred = model.forward(torch.stack([s1.rgb, s2.rgb]))

    images_masks = [s1.mask, s2.mask] if plot_object_only else None

    out_img = visualization_utils.plot_predictions(
        rgb_images=[s1.rgb, s2.rgb], predictions_images=[img1_pred, img2_pred], images_masks=images_masks
    )

    return out_img, img1_pred, img2_pred


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_object", "-mo", required=False, action="store_true", default=False,
                        help="Whether to plot descriptors for the whole image or only for the masked object area.")
    args = parser.parse_args()

    run_folders, _ = select_runs_folders(base_dir="${WS_DIR}/runs", query_file="evaluation_config.yml")
    if len(run_folders) != 1:
        print("Please select a single run folder containing an evaluation config file.")
        exit(0)

    # Select the evaluation dataset
    eval_dataset_filenames, eval_dataset_names = select_files(base_dir="${WS_DIR}/config/dataset", query_file="**.yml")
    if len(eval_dataset_filenames) != 1:
        print("Please select a single dataset for evaluation.")
        exit(0)

    evaluation_config_filename = os.path.join(run_folders[0], "evaluation_config.yml")
    evaluation_config = io_utils.load_config_file(filename=evaluation_config_filename)

    evaluator, eval_dataset, best_model, last_model, training_params = (
        evaluation_utils.load_evaluation_data(evaluation_config, eval_dataset_filenames[0], mask_object=args.mask_object)
    )

    best_model_iteration = evaluation_config["model_best"].removesuffix('.pth').split("_")[-1]
    last_model_iteration = evaluation_config["model_last"].removesuffix('.pth').split("_")[-1]
    win_name = f"Best model: iter {best_model_iteration}   /   Last model: iter {last_model_iteration}"

    # Add the click-listener callback to the window
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.setMouseCallback(win_name, match_clicked_descriptors)

    print(f"Started predictions evaluation with:")
    print(f"- Evaluation config: {evaluation_config_filename}")
    print(f"- Dataset: {eval_dataset_names[0]}")
    print(f"- Models: best (iter {best_model_iteration}), last (iter {last_model_iteration})")
    print("")

    while True:
        # Get an evaluation sample.
        samples = eval_dataset.load_evaluation_samples(num_samples=1)

        if not samples:
            print("Model evaluation error! Failed to get a valid evaluation sample.")
            continue

        data = samples[0]
        bm_out_img, bm_pred_img1, bm_pred_img2 = compute_and_plot_predictions(best_model, data, plot_object_only=args.mask_object)
        lm_out_img, lm_pred_img1, lm_pred_img2 = compute_and_plot_predictions(last_model, data, plot_object_only=args.mask_object)

        # Concatenate the two images.
        pred_img = cv.hconcat([bm_out_img, lm_out_img])
        
        # Set the parameters needed by the callback function
        _, H, W = data["s1"]["rgb"].size()
        original_prediction_img = deepcopy(pred_img)
        best_model_pred_img1, best_model_pred_img2 = bm_pred_img1, bm_pred_img2
        last_model_pred_img1, last_model_pred_img2 = lm_pred_img1, lm_pred_img2

        # Visualize the windows
        cv.imshow(win_name, original_prediction_img)
        print(f"Press any key to switch to a different pair of images and 'q' to exit the program.", end="\r")
        if cv.waitKey(0) == ord('q'):
            print("\nDescriptor matches visualization ended.")
            break
