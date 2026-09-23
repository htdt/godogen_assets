"""
Rebuild Make-It-Animatable's skeleton templates (data/Mixamo/bones.fbx, bones_vroid.fbx) without the gated
jasongzy/Mixamo dataset.

bones.fbx is the standard 65-bone Mixamo skeleton in its T-pose bind pose. Every Mixamo animation FBX contains
exactly that armature, so it is taken from the "Standard Run.fbx" clip that ships with MIA's public weights, with the
animation removed. bones_vroid.fbx (only read at import time; used by MIA's optional rabbit-ear/fox-tail model, which
this setup does not enable) is the same skeleton plus those 7 extra bones.

    rig/.conda/bin/python rig/make_templates.py      (setup.sh runs it)
"""
import os
import bpy
from mathutils import Vector

MIA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Make-It-Animatable')
SRC = os.path.join(MIA, 'data', 'Standard Run.fbx')
DST = os.path.join(MIA, 'data', 'Mixamo')
os.makedirs(DST, exist_ok=True)


def load_rest_armature():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=SRC)
    arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    arm.animation_data_clear()
    for act in list(bpy.data.actions):
        bpy.data.actions.remove(act)
    for pb in arm.pose.bones:
        pb.matrix_basis.identity()
    return arm


def export(path):
    bpy.ops.export_scene.fbx(filepath=path, object_types={'ARMATURE'}, add_leaf_bones=False, bake_anim=False,
                             use_selection=False)
    print('wrote', path)


load_rest_armature()
export(os.path.join(DST, 'bones.fbx'))

arm = load_rest_armature()
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='EDIT')
eb = arm.data.edit_bones
head, hips = eb['mixamorig:Head'], eb['mixamorig:Hips']
eb.remove(eb['mixamorig:HeadTop_End'])  # the ears take its place as the head's children
up = (head.tail - head.head).normalized()
for side, name in ((1, 'mixamorig:LRabbitEar2'), (-1, 'mixamorig:RRabbitEar2')):
    b = eb.new(name)
    b.head = head.tail + side * 0.05 * head.length * up.orthogonal()
    b.tail = b.head + up * head.length * 0.5
    b.parent = head
prev, base = hips, hips.head.copy()
for i in range(1, 6):
    b = eb.new(f'mixamorig:FoxTail{i}')
    # armature space of a Mixamo FBX is Y-up / Z-forward: go backwards (-Z) and slightly down (-Y)
    b.head = base + Vector((0, -(i - 1) * 2.0, -(i - 1) * 8.0))
    b.tail = base + Vector((0, -i * 2.0, -i * 8.0))
    b.parent = prev
    prev = b
bpy.ops.object.mode_set(mode='OBJECT')
export(os.path.join(DST, 'bones_vroid.fbx'))
