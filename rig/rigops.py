"""
Pose operations on Mixamo-named Blender armatures, shared by normalize_rig.py (Blender 4.5) and
mia_rig.py (MIA's bpy 4.3).
"""
import math
from mathutils import Vector, Matrix

UP = Vector((0, 0, 1))


def pose_bone(arm, name):
    """Pose bone by Mixamo role name, with or without the 'mixamorig:' prefix."""
    return next((b for b in arm.pose.bones if b.name.split(':')[-1] == name), None)


def world_head(arm, name):
    return (arm.matrix_world @ pose_bone(arm, name).matrix).translation.copy()


def rotate_world(arm, name, q, pivot, view_layer):
    b = pose_bone(arm, name)
    W = arm.matrix_world
    new = Matrix.Translation(pivot) @ q.to_matrix().to_4x4() @ Matrix.Translation(-pivot) @ (W @ b.matrix)
    b.matrix = W.inverted() @ new
    view_layer.update()


def upright_spine(arm, view_layer, min_deg=0.5):
    """Pose the spine chain so hips -> chest (top spine joint) is vertical (world Z). Returns the corrected lean in
    degrees (0 when below min_deg and nothing was posed). The caller applies the pose as the new rest."""
    chest = next(n for n in ('Spine2', 'Spine1', 'Spine') if pose_bone(arm, n) is not None)
    v0 = world_head(arm, chest) - world_head(arm, 'Hips')
    lean = math.degrees(v0.angle(UP))
    if lean < min_deg:
        return 0.0
    for _ in range(4):  # the pivot (Spine head) sits just above Hips, so a few iterations converge
        v = world_head(arm, chest) - world_head(arm, 'Hips')
        rotate_world(arm, 'Spine', v.normalized().rotation_difference(UP), world_head(arm, 'Spine'), view_layer)
    return lean
