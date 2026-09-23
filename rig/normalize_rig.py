"""
Normalize the bind pose of a rigged humanoid GLB (Mixamo bone names) for retargeting pipelines such as
kimodo-practical, whose certification assumes "Y-up, meters, feet flat and pointing forward at bind":

  * always: ground and centre (soles at y=0, origin between the ankles). MIA keeps TRELLIS's centred frame, with
    the hips at the origin and the feet ~1 m below it, which also silently disables kimodo's footGround gate;
  * --upright-spine: hips -> chest vertical. The retargeter builds its pelvis frame from hips -> chest, so a torso
    that leans at bind (e.g. a heroic lean-back in the source image) skews every transferred foot rotation
    (kimodo footFlat fails from ~7 deg of lean). mia-rig already applies this inside MIA;
  * --straighten-legs: hip -> ankle vertical in the frontal plane (wide-stance inputs);
  * --feet-forward: ankle -> toe pointing straight forward.

Pose changes are baked as a new rest pose (the mesh is deformed once by its own skin weights, the posed skeleton
becomes the bind), and animations in the file are then dropped. Grounding alone is a pure translation and keeps them.

    asset-blender rig/normalize_rig.py -- in.glb out.glb [--upright-spine] [--straighten-legs]
        [--feet-forward]
"""
import os
import sys
import math
import bpy
from mathutils import Vector, Matrix, Quaternion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rigops  # noqa: E402

argv = sys.argv[sys.argv.index('--') + 1:]
src, dst = argv[0], argv[1]
upright = '--upright-spine' in argv
straighten = '--straighten-legs' in argv
feet_fwd = '--feet-forward' in argv

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
scene = bpy.context.scene
vl = bpy.context.view_layer
arm = next(o for o in scene.objects if o.type == 'ARMATURE')
meshes = [o for o in scene.objects if o.type == 'MESH' and len(o.data.vertices) > 100]
for o in [o for o in scene.objects if o.type == 'MESH' and o not in meshes]:
    bpy.data.objects.remove(o)  # the glTF importer's bone display shapes


def activate(objs, active):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    vl.objects.active = active


for b in arm.pose.bones:
    b.matrix_basis = Matrix.Identity(4)
vl.update()
fwd = Vector((0, -1, 0))  # glTF +Z forward imports as Blender -Y

report = {}
if upright:
    report['spine_lean_fixed_deg'] = round(rigops.upright_spine(arm, vl), 1)
for side in ('Left', 'Right'):
    if straighten:
        hip, ankle = rigops.world_head(arm, f'{side}UpLeg'), rigops.world_head(arm, f'{side}Foot')
        frontal = Vector(((ankle - hip).x, 0, (ankle - hip).z))  # splay only
        q = frontal.normalized().rotation_difference(Vector((0, 0, -1)))
        rigops.rotate_world(arm, f'{side}UpLeg', q, hip, vl)
        report[f'{side}_splay_fixed_deg'] = round(math.degrees(q.angle), 1)
    if feet_fwd:
        ankle, toe = rigops.world_head(arm, f'{side}Foot'), rigops.world_head(arm, f'{side}ToeBase')
        flat = Vector(((toe - ankle).x, (toe - ankle).y))
        if flat.length > 1e-6:
            ang = flat.normalized().angle_signed(fwd.xy)
            rigops.rotate_world(arm, f'{side}Foot', Quaternion(rigops.UP, -ang), ankle, vl)
            report[f'{side}_yaw_fixed_deg'] = round(math.degrees(ang), 1)

reposed = any(b.matrix_basis != Matrix.Identity(4) for b in arm.pose.bones)
if reposed:
    if arm.animation_data:
        arm.animation_data_clear()
    for m in meshes:  # bake the pose into the mesh, then make the pose the new rest
        mod = next(md for md in m.modifiers if md.type == 'ARMATURE')
        activate([m], m)
        bpy.ops.object.modifier_apply(modifier=mod.name)
    activate([arm], arm)
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    for m in meshes:
        m.modifiers.new('Armature', 'ARMATURE').object = arm

# ground + centre (Blender Z-up == glTF Y-up): shift rest bones and mesh by the same world offset
vl.update()
dg = bpy.context.evaluated_depsgraph_get()
minz = min((m.matrix_world @ v.co).z for m in meshes for v in m.evaluated_get(dg).to_mesh().vertices)
mid = (rigops.world_head(arm, 'LeftFoot') + rigops.world_head(arm, 'RightFoot')) / 2
d = Vector((-mid.x, -mid.y, -minz))
activate([arm], arm)
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
report['ground_shift_m'] = [round(x, 3) for x in d]

bpy.ops.export_scene.gltf(filepath=dst, export_animations=not reposed)
print('NORMALIZED', dst, report)
