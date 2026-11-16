# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import bpy
import os
import argparse
import glob
    

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)

def import_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".ply":
        bpy.ops.wm.ply_import(filepath=filepath, import_colors="SRGB")
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=filepath)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def export_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".ply":
        bpy.ops.wm.ply_export(filepath=filepath, export_colors="SRGB")
    elif ext == ".obj":
        bpy.ops.wm.obj_export(filepath=filepath)
    elif ext == ".stl":
        bpy.ops.wm.stl_export(filepath=filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def scale_objects(scale_factor):
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.scale = [s * scale_factor for s in obj.scale]
    bpy.ops.object.transform_apply(scale=True)

def main():
    parser = argparse.ArgumentParser(description="Batch scale 3D files using Blender's bpy.")
    parser.add_argument("--input_folder", "-in", required=True, help="Path to input folder containing 3D files.")
    parser.add_argument("--extension", "-ext", required=True, help="File extension to load (e.g. .obj, .stl, .fbx).")
    parser.add_argument("--scale", "-s", type=float, required=True, help="Scale factor to apply to all objects.")
    args = parser.parse_args()

    input_files = glob.glob(os.path.join(args.input_folder, f"*{args.extension}"))

    for filepath in sorted(input_files):
        clear_scene()
        import_file(filepath)
        scale_objects(args.scale)
        output_path = os.path.splitext(filepath)[0] + f"_scaled{args.extension}"
        export_file(output_path)

if __name__ == "__main__":
    main()
