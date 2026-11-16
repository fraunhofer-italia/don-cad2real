# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

"""
Script used to convert scenes from the original DON2018 dataset to our 'DonDataset' expected standard.
    
Objects scenes are expected to be stored in the directory '$WS_DIR/data/don-2018/original' each inside a folder
named after the associated object id:
- $WS_DIR/
    - data/
        - don-2018/
            - original/
                - obj1/
                    - 2024-06-... (first scene)
                    - 2024-06-... (second scene)
                    - ...
                - obj2/
                    - ...
                - ...
    
Converted scenes will be placed inside the '$WS_DIR/data/don-2018/converted' folder.

Original dataset is available at:
    https://github.com/RobotLocomotion/pytorch-dense-correspondence/blob/master/doc/tutorial_getting_started.md#step-1-download-the-data
"""
    
import os
import yaml
import shutil

BASE_PATH = os.path.expandvars("$WS_DIR/data/don-2018")

def convert_pose_data(src_file_path, out_file_path):
    with open(src_file_path, "r") as file:
        pose_data = yaml.safe_load(file)

    snapshots_count = 0
    new_pose_data = {}
    for _, data in pose_data.items():
        # Load the pose info
        camera_to_world = {
            "rotation": data["camera_to_world"]["quaternion"],
            "translation": data["camera_to_world"]["translation"]
        }

        depth_img = data["depth_image_filename"]
        rgb_img = data["rgb_image_filename"]

        # Create the new data entry
        new_data = {
            "camera_to_world": camera_to_world,
            "depth_img_filename": depth_img,
            "rgb_img_filename": rgb_img
        }

        # Update the new dict
        new_pose_data[snapshots_count] = new_data
        
        snapshots_count += 1

    # Save the updated data
    with open(out_file_path, "w") as file:
        yaml.dump(new_pose_data, file, default_flow_style=False)
        
    return snapshots_count
        
def create_scene_info(category, snapshots_count, scene_type, out_file_path):
    # Create the scene info dict
    scene_info_data = {"object_id": category, "scene_type": scene_type, "snapshots_count": snapshots_count}
    
    # Save the scene info
    with open(out_file_path, "w") as file:
        yaml.dump(scene_info_data, file, default_flow_style=False)

def convert_camera_info_data(src_file_path, out_file_path):
    with open(src_file_path, "r") as file:
        camera_info = yaml.safe_load(file)

    # Load the camera parameters
    distortion_coefficients = camera_info["distortion_coefficients"]["data"]
    camera_matrix = camera_info["camera_matrix"]["data"]
    projection_matrix = camera_info["projection_matrix"]["data"]
    rectification_matrix = camera_info["rectification_matrix"]["data"]
    distortion_model = camera_info["distortion_model"]
    height = camera_info["image_height"]
    width = camera_info["image_width"]

    camera_model = {"D": distortion_coefficients,
                    "K": camera_matrix,
                    "P": projection_matrix,
                    "R": rectification_matrix,
                    "distortion_model": distortion_model,
                    "height": height,
                    "width": width
                    }
    new_camera_info = {"camera_info": camera_model}

     # Save the updated data
    with open(out_file_path, "w") as file:
        yaml.dump(new_camera_info, file, default_flow_style=False)

def remove_old_data(scene_path):
    shutil.rmtree(os.path.join(scene_path, "image_masks"))
    shutil.rmtree(os.path.join(scene_path, "rendered_images"))
    
    os.remove(os.path.join(scene_path, "fusion_mesh.ply"))
    os.remove(os.path.join(scene_path, "fusion_pointcloud.ply"))
    os.remove(os.path.join(scene_path, "tsdf.bin"))


if __name__ == "__main__":
    BASE_PATH = os.path.expandvars("$WS_DIR/data/don-2018")

    # Path to dir of the original dataset
    orig_data_path = os.path.join(BASE_PATH, "original")
    
    # Create dir for the 'converted' dataset
    conv_data_path = os.path.join(BASE_PATH, "converted")
    os.makedirs(conv_data_path, exist_ok=True)

    # Get paths to all 'category directories' in the original dataset
    category_dirs = os.listdir(orig_data_path)
    category_dirs.sort()

    for category_id in category_dirs:
        # Get paths to all scenes in the original dataset for the current category 
        category_dir_path = os.path.join(orig_data_path, category_id)
        scene_dirs = os.listdir(category_dir_path)
        scene_dirs.sort()
        
        # Save the names of all the scenes for the current category
        with open(os.path.join(conv_data_path, category_id, "scenes_list.yml"), "w") as file:
            yaml.dump({"scenes_list": [f"{category_id}/{scene}" for scene in scene_dirs]}, file, default_flow_style=False)
        
        for scene in scene_dirs:
            print(f"Processing category: {category_id}, scene: {scene}")
            
            original_scene_path = os.path.join(orig_data_path, category_id, scene)
            converted_scene_path = os.path.join(conv_data_path, category_id, scene)

            # Copy the dir and remove the `processed` subfolder
            shutil.copytree(os.path.join(original_scene_path, "processed"), converted_scene_path, dirs_exist_ok=True)
            
            # Move and convert `pose_data.yaml` to our dataset standard
            old_pose_data_path = os.path.join(converted_scene_path, "images", "pose_data.yaml")
            new_pose_data_path = os.path.join(converted_scene_path, "pose_data.yml")
            snapshots_count =  convert_pose_data(old_pose_data_path, new_pose_data_path)
            os.remove(old_pose_data_path)

            # Create the 'scene_info.yml' file
            new_scene_info_data_path = os.path.join(converted_scene_path, "scene_info.yml")
            create_scene_info(category_id, snapshots_count, "SINGLE_OBJECT_SCENE", new_scene_info_data_path)

            # Move and convert `camera_info.yaml` to our dataset standard
            old_camera_info_data_path = os.path.join(converted_scene_path, "images", "camera_info.yaml")
            new_camera_info_data_path = os.path.join(converted_scene_path, "camera_info.yml")
            convert_camera_info_data(old_camera_info_data_path, new_camera_info_data_path)
            os.remove(old_camera_info_data_path)
            
            # Remove old data
            remove_old_data(converted_scene_path)