# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

bl_info = {
    "name": "Move Origin to Bounding Box Corner",
    "blender": (3, 0, 0),
    "category": "Object",
    "author": "ChatGPT",
    "description": "Move the object's origin to a selected bounding box corner",
}

import bpy
from mathutils import Vector

def get_bbox_corners(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()

    vertices = [obj.matrix_world @ v.co for v in mesh.vertices]

    min_corner = Vector((
        min(v.x for v in vertices),
        min(v.y for v in vertices),
        min(v.z for v in vertices)
    ))
    max_corner = Vector((
        max(v.x for v in vertices),
        max(v.y for v in vertices),
        max(v.z for v in vertices)
    ))

    bbox_corners = [
        Vector((min_corner.x, min_corner.y, min_corner.z)),
        Vector((min_corner.x, min_corner.y, max_corner.z)),
        Vector((min_corner.x, max_corner.y, min_corner.z)),
        Vector((min_corner.x, max_corner.y, max_corner.z)),
        Vector((max_corner.x, min_corner.y, min_corner.z)),
        Vector((max_corner.x, min_corner.y, max_corner.z)),
        Vector((max_corner.x, max_corner.y, min_corner.z)),
        Vector((max_corner.x, max_corner.y, max_corner.z)),
    ]

    obj_eval.to_mesh_clear()
    return bbox_corners

class OBJECT_OT_move_origin_to_bbox_corner(bpy.types.Operator):
    """Move Origin to Bounding Box Corner"""
    bl_idname = "object.move_origin_to_bbox_corner"
    bl_label = "Move Origin to Bounding Box Corner"
    bl_options = {'REGISTER', 'UNDO'}

    corner_index: bpy.props.IntProperty(
        name="Corner Index",
        description="Select which bounding box corner (0-7)",
        default=0,
        min=0,
        max=7
    )

    def execute(self, context):
        obj = context.active_object
        if obj is None:
            self.report({'WARNING'}, "No active object selected.")
            return {'CANCELLED'}

        bbox_corners = get_bbox_corners(obj)

        corner_location = bbox_corners[self.corner_index]
        context.scene.cursor.location = corner_location

        bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

        return {'FINISHED'}

class OBJECT_PT_bbox_corner_panel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Bounding Box Origin Tools"
    bl_idname = "OBJECT_PT_bbox_corner_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Origin Tools'

    def draw(self, context):
        layout = self.layout
        layout.label(text="Move Origin to BB Corner:")
        layout.operator("object.move_origin_to_bbox_corner")

def register():
    bpy.utils.register_class(OBJECT_OT_move_origin_to_bbox_corner)
    bpy.utils.register_class(OBJECT_PT_bbox_corner_panel)

def unregister():
    bpy.utils.unregister_class(OBJECT_OT_move_origin_to_bbox_corner)
    bpy.utils.unregister_class(OBJECT_PT_bbox_corner_panel)

if __name__ == "__main__":
    register()
