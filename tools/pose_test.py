"""
Deformation stress test for a rigged humanoid (Mixamo bone names): pose it into 6 key poses and render them.

    asset-blender tools/pose_test.py -- rigged.glb out.png [--res 384] [--save posed.glb]

Poses: rest, arms down, arms up, elbows/knees bent (boxing guard), squat, twist + kick. Every rotation is given in
world space as "turn this bone towards that direction" and computed from the rest pose, so the test does not depend on
the bone roll convention of the rig. The facing direction is taken from the feet (heel -> toe).
"""
import bpy
import sys
import os
import math
import numpy as np
from mathutils import Vector, Matrix, Quaternion

argv = sys.argv[sys.argv.index('--') + 1:]
src, out = argv[0], argv[1]
res = int(argv[argv.index('--res') + 1]) if '--res' in argv else 384
save = argv[argv.index('--save') + 1] if '--save' in argv else None

bpy.ops.wm.read_factory_settings(use_empty=True)
(bpy.ops.import_scene.fbx if src.lower().endswith('.fbx') else bpy.ops.import_scene.gltf)(filepath=src)
scene = bpy.context.scene
arm = next(o for o in scene.objects if o.type == 'ARMATURE')
# skip the glTF importer's bone display shapes (a 1 m icosphere at the origin that would skew the framing)
meshes = [o for o in scene.objects if o.type == 'MESH' and len(o.data.vertices) > 100]
if arm.animation_data:
    arm.animation_data.action = None
    for tr in arm.animation_data.nla_tracks:
        tr.mute = True


def bone(name):
    for b in arm.pose.bones:
        if b.name.split(':')[-1] == name:
            return b
    return None


for pb in arm.pose.bones:
    pb.matrix_basis = Matrix.Identity(4)
bpy.context.view_layer.update()
W = arm.matrix_world
rest = {pb.name: (W @ pb.matrix).copy() for pb in arm.pose.bones}  # world-space rest matrices


def wdir(name):
    m = rest[bone(name).name]
    return (m.to_3x3() @ Vector((0, 1, 0))).normalized()


up = Vector((0, 0, 1))
fwd = sum((wdir(f'{s}Foot') for s in ('Left', 'Right')), Vector()) + \
    sum((wdir(f'{s}ToeBase') for s in ('Left', 'Right')), Vector())
fwd.z = 0
fwd.normalize()
left = up.cross(fwd).normalized()  # character's left (screen right when facing the camera)
if (rest[bone('LeftArm').name].translation - rest[bone('RightArm').name].translation).dot(left) < 0:
    left = -left


def d(x, y, z):
    """direction in character space: x = character's left, y = forward, z = up"""
    return (left * x + fwd * y + up * z).normalized()


def s(v, side):  # mirror a character-space direction for the right side
    return v if side == 'Left' else Vector((-v[0], v[1], v[2]))


POSES = [
    ('rest', {}),
    ('arms down', {f'{sd}Arm': s(Vector((0.35, 0, -1)), sd) for sd in ('Left', 'Right')}),
    ('arms up', {**{f'{sd}Arm': s(Vector((0.3, 0.1, 1)), sd) for sd in ('Left', 'Right')}}),
    ('guard', {**{f'{sd}Arm': s(Vector((0.5, 0.4, -0.6)), sd) for sd in ('Left', 'Right')},
               **{f'{sd}ForeArm': s(Vector((-0.1, 1, 0.6)), sd) for sd in ('Left', 'Right')},
               'LeftUpLeg': Vector((0.15, 0.5, -1)), 'LeftLeg': Vector((0.1, -0.35, -1))}),
    ('squat', {**{f'{sd}UpLeg': s(Vector((0.3, 1, -0.25)), sd) for sd in ('Left', 'Right')},
               **{f'{sd}Leg': s(Vector((0.1, -0.1, -1)), sd) for sd in ('Left', 'Right')},
               **{f'{sd}Arm': s(Vector((0.2, 1, 0)), sd) for sd in ('Left', 'Right')},
               'Spine': Vector((0, 0.35, 1))}),
    ('twist+kick', {'Spine1': ('twist', 35), 'Head': ('twist', -40),
                    'RightUpLeg': Vector((-0.1, 1, 0.15)), 'RightLeg': Vector((-0.1, 1, 0.1)),
                    'LeftArm': Vector((1, -0.3, -0.3)), 'RightArm': Vector((-0.6, 0.8, 0.5))}),
]
ORDER = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
         'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
         'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot']


def apply_pose(targets):
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    for name in ORDER:
        pb = bone(name)
        if pb is None or name not in targets:
            continue
        cur = W @ pb.matrix  # current world matrix (parents already posed)
        cur_dir = (cur.to_3x3() @ Vector((0, 1, 0))).normalized()
        t = targets[name]
        if isinstance(t, tuple):  # twist about the up axis
            q = Quaternion(up, math.radians(t[1]))
        else:
            q = cur_dir.rotation_difference(d(*t))
        new = Matrix.Translation(cur.translation) @ q.to_matrix().to_4x4() @ Matrix.Translation(-cur.translation) @ cur
        pb.matrix = W.inverted() @ new
        bpy.context.view_layer.update()


# ---- render setup ----
scene.render.engine = 'CYCLES'
prefs = bpy.context.preferences.addons['cycles'].preferences
for backend in (() if os.environ.get('RENDER_CPU') else ('OPTIX', 'CUDA')):
    try:
        prefs.compute_device_type = backend
        prefs.get_devices()
        if any(dv.type == backend for dv in prefs.devices):
            for dv in prefs.devices:
                dv.use = dv.type == backend
            scene.cycles.device = 'GPU'
            break
    except TypeError:
        continue
scene.cycles.samples = 24
scene.cycles.use_denoising = True
scene.render.resolution_x = scene.render.resolution_y = res
scene.view_settings.view_transform = 'Standard'
world = bpy.data.worlds.new('w')
scene.world = world
world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (1, 1, 1, 1)
world.node_tree.nodes['Background'].inputs[1].default_value = 0.7
sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN'))
sun.data.energy = 3
sun.rotation_euler = (math.radians(50), 0, math.radians(30))
scene.collection.objects.link(sun)
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
cam.data.lens = 50
scene.collection.objects.link(cam)
scene.camera = cam

pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
center, size = (lo + hi) / 2, hi - lo
fov = 2 * math.atan(cam.data.sensor_width / 2 / cam.data.lens)
dist = max(size) / 2 * 1.25 / math.tan(fov / 2)
view = (fwd * math.cos(math.radians(35)) + left * math.sin(math.radians(35))).normalized()  # 3/4 front view
cam.location = center + (view + Vector((0, 0, 0.15))).normalized() * dist
cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()

tmp = []
for i, (label, targets) in enumerate(POSES):
    apply_pose(targets)
    scene.render.filepath = f'{out}.p{i}.png'
    bpy.ops.render.render(write_still=True)
    tmp.append(scene.render.filepath)
    if save and label == 'squat':
        bpy.ops.export_scene.gltf(filepath=save, export_animations=False)

Wd = res * len(tmp)
strip = bpy.data.images.new('strip', Wd, res, alpha=False)
px = np.zeros((res, Wd, 4), dtype=np.float32)
for i, p in enumerate(tmp):
    im = bpy.data.images.load(p)
    a = np.empty(res * res * 4, dtype=np.float32)
    im.pixels.foreach_get(a)
    px[:, i * res:(i + 1) * res] = a.reshape(res, res, 4)
    os.remove(p)
strip.pixels.foreach_set(px.ravel())
strip.filepath_raw = out
strip.file_format = 'PNG'
strip.save()
print('WROTE', out, [p[0] for p in POSES])
