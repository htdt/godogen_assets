"""Rigid prop preview in Blender. Offsets use glTF character axes at bind, independent of imported bone rolls."""
import math
import re
import bpy
from mathutils import Matrix, Vector, Euler


def triple(value):
    values = tuple(float(x) for x in value.split(','))
    if len(values) != 3 or not all(math.isfinite(x) for x in values):
        raise ValueError('expected three finite numbers separated by commas')
    return values


def attach(arm, path, bone='RightHand', offset=(0, 0, 0), rotation=(0, 0, 0), scale=1, grip=(0, 0, 0)):
    matches = [pb for pb in arm.pose.bones if pb.name == bone
               or re.sub(r'^mixamorig\d*[:_]?', '', pb.name) == bone]
    if len(matches) != 1:
        raise ValueError(f'expected one bone {bone!r}, found {len(matches)}')
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('prop scale must be finite and positive')
    pb = matches[0]
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=path, disable_bone_shape=True)
    objects = set(bpy.context.scene.objects) - before
    if any(o.type == 'ARMATURE' or o.animation_data for o in objects):
        raise ValueError('the preview prop must be rigid and unanimated')
    holder = bpy.data.objects.new('attachment_preview', None)
    bpy.context.scene.collection.objects.link(holder)
    for obj in objects:
        if obj.parent not in objects:
            world = obj.matrix_world.copy()
            obj.parent = holder
            obj.matrix_world = world
    # glTF Y-up -> Blender Z-up. A grip point is in the prop's original glTF scene coordinates.
    basis = Matrix.Rotation(math.pi / 2, 4, 'X')
    bind = arm.matrix_world @ pb.bone.matrix_local
    placement = (Matrix.Translation(Vector(offset)) @ Euler(tuple(math.radians(x) for x in rotation), 'XYZ').to_matrix().to_4x4()
                 @ Matrix.Scale(scale, 4) @ Matrix.Translation(-Vector(grip)))
    local = bind.inverted() @ Matrix.Translation(bind.translation) @ basis @ placement @ basis.inverted()

    def update():
        holder.matrix_world = arm.matrix_world @ pb.matrix @ local
        bpy.context.view_layer.update()

    update()
    return update
