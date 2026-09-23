"""
Render a mouth-rigged character (lipsync/mouth_rig.py output) in a grid of mouth poses, close up on the face.

    asset-blender lipsync/face_test.py -- mouth.glb out.png [--res 384] [--yaw 25] [--zoom 1.0]
        [--weights]      (show the jaw skin weights as colour instead of the texture: 0 = grey, 1 = red)
        [--views front,34,side,section]   (section: sagittal cut through the middle of the mouth)

Poses (columns): rest, jaw 40 %, jaw 100 %, wide, round, wide + jaw 30 %, round + jaw 40 %. Each view is one row.
"""
import bpy
import sys
import os
import math
import json
import numpy as np
from mathutils import Vector, Quaternion

argv = sys.argv[sys.argv.index('--') + 1:]
src, out = argv[0], argv[1]
res = int(argv[argv.index('--res') + 1]) if '--res' in argv else 384
zoom = float(argv[argv.index('--zoom') + 1]) if '--zoom' in argv else 1.0
views = argv[argv.index('--views') + 1].split(',') if '--views' in argv else ['front', '34', 'side']
show_weights = '--weights' in argv
hide = argv[argv.index('--hide') + 1] if '--hide' in argv else None  # debug: hide the body or the mouth parts
face = json.load(open(os.path.splitext(src)[0] + '.face.json'))

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
scene = bpy.context.scene
arm = next(o for o in scene.objects if o.type == 'ARMATURE')
meshes = [o for o in scene.objects if o.type == 'MESH' and (o.parent is not None or len(o.data.vertices) > 100)]
body = max(meshes, key=lambda o: len(o.data.vertices))
if arm.animation_data:
    arm.animation_data.action = None
jaw = arm.pose.bones[face['jaw_bone']]
jaw.rotation_mode = 'QUATERNION'
keys = body.data.shape_keys.key_blocks if body.data.shape_keys else {}
JMAX = face['jaw_open_max_rad']
POSES = [('rest', 0, {}), ('jaw 40%', 0.4, {}), ('jaw 100%', 1.0, {}), ('wide', 0, {'wide': 1}),
         ('round', 0, {'round': 1}), ('wide+jaw30', 0.3, {'wide': 1}), ('round+jaw40', 0.4, {'round': 1})]


def pose(j, sk):
    jaw.rotation_quaternion = Quaternion((1, 0, 0), j * JMAX)
    for kb in list(keys)[1:]:
        kb.value = sk.get(kb.name, 0.0)
    bpy.context.view_layer.update()


for o in meshes:
    if hide and (o.name.startswith('mouth') == (hide == 'mouth')):
        o.hide_render = True
if show_weights:
    for o in meshes:
        gi = o.vertex_groups.get(face['jaw_bone'])
        w = np.zeros(len(o.data.vertices))
        if gi is not None:
            for v in o.data.vertices:
                for g in v.groups:
                    if g.group == gi.index:
                        w[v.index] = g.weight
        ca = o.data.color_attributes.new('jaww', 'FLOAT_COLOR', 'POINT')
        cols = np.stack([0.6 + 0.4 * w, 0.6 * (1 - w), 0.6 * (1 - w), np.ones_like(w)], 1)
        ca.data.foreach_set('color', cols.ravel())
        m = bpy.data.materials.new('wview')
        m.use_nodes = True
        nt = m.node_tree
        at = nt.nodes.new('ShaderNodeAttribute')
        at.attribute_name = 'jaww'
        nt.links.new(at.outputs['Color'], nt.nodes['Principled BSDF'].inputs['Base Color'])
        o.data.materials.clear()
        o.data.materials.append(m)

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
scene.cycles.samples = 32
scene.cycles.use_denoising = True
scene.render.resolution_x = scene.render.resolution_y = res
scene.view_settings.view_transform = 'Standard'
world = bpy.data.worlds.new('w')
scene.world = world
world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (1, 1, 1, 1)
world.node_tree.nodes['Background'].inputs[1].default_value = 0.6
fr = face['frame']
left, up, fwd = (Vector(fr[k]) for k in ('left', 'up', 'fwd'))
sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN'))
sun.data.energy = 2.5
sun.rotation_euler = (-(fwd * 0.8 + up * 0.6 + left * 0.3)).to_track_quat('-Z', 'Y').to_euler()
scene.collection.objects.link(sun)
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
cam.data.lens = 85
scene.collection.objects.link(cam)
scene.camera = cam
mw = face['mouth_width_m']
target = Vector(face['lip_center_world']) + up * (0.9 * mw)  # between the mouth and the nose
size = 5.0 * mw / zoom                                      # framed height: from the chin to the eyes
fov = 2 * math.atan(cam.data.sensor_width / 2 / cam.data.lens)
dist = size / 2 / math.tan(fov / 2)
YAW = {'front': 0, '34': 35, 'side': 80, 'section': 90}

tmp = []
for vname in views:
    y = math.radians(YAW[vname])
    d = (fwd * math.cos(y) + left * math.sin(y) + up * 0.08).normalized()
    cam.location = target + d * dist
    cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()
    if vname == 'section':  # sagittal cut: side view whose near clip plane passes through the middle of the mouth
        cam.data.type = 'ORTHO'
        cam.data.ortho_scale = size
        cam.location = target - up * (0.6 * mw) + left * dist
        cam.rotation_euler = (-left).to_track_quat('-Z', 'Y').to_euler()
        cam.data.clip_start = dist
        # inside the head no light reaches: Workbench, lit from the view, one colour per material
        scene.render.engine = 'BLENDER_WORKBENCH'
        scene.display.shading.light = 'STUDIO'
        scene.display.shading.color_type = 'MATERIAL'
        for m in bpy.data.materials:
            m.diffuse_color = {'mouth_cavity': (0.35, 0.05, 0.05, 1), 'teeth': (1, 1, 0.9, 1),
                               'tongue': (0.9, 0.35, 0.45, 1)}.get(m.name, (0.85, 0.7, 0.55, 1))
    else:
        scene.render.engine = 'CYCLES'
        cam.data.type = 'PERSP'
        cam.data.clip_start = 0.01
    for i, (label, j, sk) in enumerate(POSES):
        pose(j, sk)
        scene.render.filepath = f'{out}.{vname}{i}.png'
        bpy.ops.render.render(write_still=True)
        tmp.append(scene.render.filepath)

ncol, nrow = len(POSES), len(views)
strip = bpy.data.images.new('strip', res * ncol, res * nrow, alpha=False)
px = np.zeros((res * nrow, res * ncol, 4), dtype=np.float32)
for k, p in enumerate(tmp):
    r_, c_ = divmod(k, ncol)
    im = bpy.data.images.load(p)
    a = np.empty(res * res * 4, dtype=np.float32)
    im.pixels.foreach_get(a)
    px[(nrow - 1 - r_) * res:(nrow - r_) * res, c_ * res:(c_ + 1) * res] = a.reshape(res, res, 4)
    os.remove(p)
strip.pixels.foreach_set(px.ravel())
strip.filepath_raw = out
strip.file_format = 'PNG'
strip.save()
print('WROTE', out, [p[0] for p in POSES], views)
