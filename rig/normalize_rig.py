"""
Ground and centre a rigged humanoid GLB (Mixamo bone names): soles at y=0, origin between the ankles. MIA keeps
TRELLIS's centred frame, with the hips at the origin and the feet ~1 m below it. A pure translation of the rest bones
and the mesh, so animations in the file are kept.

    asset-blender rig/normalize_rig.py -- in.glb out.glb
"""
import os
import sys
import bpy
from mathutils import Vector, Matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rigops  # noqa: E402

argv = sys.argv[sys.argv.index('--') + 1:]
src, dst = argv[0], argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src, disable_bone_shape=True)
scene = bpy.context.scene
vl = bpy.context.view_layer
arm = next(o for o in scene.objects if o.type == 'ARMATURE')
meshes = [o for o in scene.objects if o.type == 'MESH']

if arm.animation_data:
    arm.animation_data.action = None
    for track in arm.animation_data.nla_tracks:
        track.mute = True
for b in arm.pose.bones:
    b.matrix_basis = Matrix.Identity(4)
vl.update()

# Blender Z-up == glTF Y-up: shift rest bones and mesh by the same world offset
dg = bpy.context.evaluated_depsgraph_get()
minz = float('inf')
for m in meshes:
    evaluated = m.evaluated_get(dg)
    mesh = evaluated.to_mesh()
    minz = min(((evaluated.matrix_world @ v.co).z for v in mesh.vertices), default=minz)
    evaluated.to_mesh_clear()
mid = (rigops.world_head(arm, 'LeftFoot') + rigops.world_head(arm, 'RightFoot')) / 2
d = Vector((-mid.x, -mid.y, -minz))
bpy.ops.object.select_all(action='DESELECT')
arm.select_set(True)
vl.objects.active = arm
bpy.ops.object.mode_set(mode='EDIT')
dl = arm.matrix_world.inverted().to_3x3() @ d
for eb in arm.data.edit_bones:
    eb.head += dl
    eb.tail += dl
bpy.ops.object.mode_set(mode='OBJECT')
for m in meshes:
    ancestor = m
    while ancestor and ancestor.parent_type != 'BONE':
        ancestor = ancestor.parent
    if ancestor:  # rigid attachments already follow the shifted rest bone
        continue
    ml = m.matrix_world.inverted().to_3x3() @ d
    blocks = [k.data for k in m.data.shape_keys.key_blocks] if m.data.shape_keys else [m.data.vertices]
    for vertices in blocks:
        for v in vertices:
            v.co += ml

bpy.ops.export_scene.gltf(filepath=dst)
print('NORMALIZED', dst, {'ground_shift_m': [round(x, 3) for x in d]})

if 'asset_result' in globals():
    asset_result({'output': dst})
