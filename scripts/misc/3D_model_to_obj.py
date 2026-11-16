# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import os
import glob
import argparse
import pymeshlab


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch scale 3D files using Blender's bpy.")
    parser.add_argument("--input_folder", "-in", required=True, help="Path to input folder containing 3D files.")
    parser.add_argument("--query", "-q", required=True, help="Query filename pattern (e.g. .ply).")
    
    args = parser.parse_args()

    input_files = glob.glob(os.path.join(args.input_folder, f"*{args.query}"))

    for filepath in sorted(input_files):
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(filepath)
        extension = os.path.splitext(filepath)[1].lower()
        out_path = filepath.replace(extension, '.obj')
        ms.save_current_mesh(
            out_path,
            save_face_color=True,
            save_wedge_texcoord=True,
        )
        print(f"Converted {filepath} to {out_path}")