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
bpy.ops.import_scene.gltf(filepath=src)
scene = bpy.context.scene
vl = bpy.context.view_layer
arm = next(o for o in scene.objects if o.type == 'ARMATURE')
meshes = [o for o in scene.objects if o.type == 'MESH' and len(o.data.vertices) > 100]
for o in [o for o in scene.objects if o.type == 'MESH' and o not in meshes]:
    bpy.data.objects.remove(o)  # the glTF importer's bone display shapes

for b in arm.pose.bones:
    b.matrix_basis = Matrix.Identity(4)
vl.update()

# Blender Z-up == glTF Y-up: shift rest bones and mesh by the same world offset
dg = bpy.context.evaluated_depsgraph_get()
minz = min((m.matrix_world @ v.co).z for m in meshes for v in m.evaluated_get(dg).to_mesh().vertices)
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
    ml = m.matrix_world.inverted().to_3x3() @ d
    for v in m.data.vertices:
        v.co += ml

bpy.ops.export_scene.gltf(filepath=dst)
print('NORMALIZED', dst, {'ground_shift_m': [round(x, 3) for x in d]})
