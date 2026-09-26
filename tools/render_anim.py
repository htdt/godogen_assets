"""
Render evenly spaced frames of an animated, skinned GLB/FBX with Blender (to check rigging/skinning quality).

    asset-blender tools/render_anim.py -- anim.glb out.png [--frames 8] [--res 384] [--yaw 30]
        [--action NAME]      (a GLB with several clips: render the one named NAME, e.g. a Kimodo move)
        [--prop sword.glb --bone RightHand --grip 0,-0.4,0 --scale 1 --offset 0,0,0 --rotation 0,0,0]

Writes one horizontal strip PNG (frame i left to right). The camera is fixed and framed on the whole animation's
bounding box so root motion stays visible.
"""
import bpy
import sys
import os
import math
import argparse
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attachments import attach, triple
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('src')
ap.add_argument('out')
ap.add_argument('--frames', type=int, default=8)
ap.add_argument('--res', type=int, default=384)
ap.add_argument('--yaw', type=float, default=30)
ap.add_argument('--action')
ap.add_argument('--prop')
ap.add_argument('--bone', default='RightHand')
ap.add_argument('--grip', type=triple, default=(0, 0, 0))
ap.add_argument('--offset', type=triple, default=(0, 0, 0))
ap.add_argument('--rotation', type=triple, default=(0, 0, 0))
ap.add_argument('--scale', type=float, default=1)
args = ap.parse_args(argv)
src, out, nframes, res, yaw_deg, want = args.src, args.out, args.frames, args.res, args.yaw, args.action
if nframes < 1 or res < 1:
    raise ValueError('--frames and --res must be positive')
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
if src.lower().endswith('.fbx'):
    bpy.ops.import_scene.fbx(filepath=src)
else:
    bpy.ops.import_scene.gltf(filepath=src, disable_bone_shape=True)
scene = bpy.context.scene
arms = [o for o in scene.objects if o.type == 'ARMATURE']

# glTF imports clips as NLA strips, including separate slots for morph weights. Activate only one clip everywhere.
owners = list(scene.objects) + [o.data.shape_keys for o in scene.objects if o.type == 'MESH' and o.data.shape_keys]
available = {tr.name for o in owners if o.animation_data for tr in o.animation_data.nla_tracks}
available.update(a.name for a in bpy.data.actions)
if want is None:
    want = next((o.animation_data.action.name for o in arms if o.animation_data and o.animation_data.action), None)
    want = want or next(iter(sorted(available)), None)
if want is None or want not in available:
    raise ValueError(f'no action {want!r}; have: {sorted(available)}')
ranges = []
for owner in owners:
    ad = owner.animation_data
    if not ad:
        continue
    active = (ad.action, ad.action_slot) if ad.action and ad.action.name == want else None
    for track in ad.nla_tracks:
        track.mute = True
        for strip in track.strips:
            if strip.action and (track.name == want or strip.action.name == want):
                active = (strip.action, strip.action_slot)
    ad.action = None
    if active:
        ad.action, ad.action_slot = active
        ranges.append(active[0].frame_range)
if not ranges:
    raise ValueError(f'action {want!r} has no bound object slots')
f0, f1 = min(r[0] for r in ranges), max(r[1] for r in ranges)
update_prop = lambda: None
if args.prop:
    if len(arms) != 1:
        raise ValueError('--prop needs exactly one character armature')
    update_prop = attach(arms[0], os.path.abspath(args.prop), args.bone, args.offset, args.rotation, args.scale, args.grip)
meshes = [o for o in scene.objects if o.type == 'MESH']


def frame_set(f):
    scene.frame_set(f)
    update_prop()


frames = [round(f0 + (f1 - f0) * i / max(nframes - 1, 1)) for i in range(nframes)]

# bounding box over all sampled frames (evaluated, i.e. deformed)
lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
for f in frames:
    frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    for o in meshes:
        ev = o.evaluated_get(dg)
        m = ev.to_mesh()
        for k in range(0, len(m.vertices), max(1, len(m.vertices) // 5000)):
            p = ev.matrix_world @ m.vertices[k].co
            lo = Vector(map(min, lo, p))
            hi = Vector(map(max, hi, p))
        ev.to_mesh_clear()
center, size = (lo + hi) / 2, hi - lo

scene.render.engine = 'CYCLES'
prefs = bpy.context.preferences.addons['cycles'].preferences
for backend in (() if os.environ.get('RENDER_CPU') else ('OPTIX', 'CUDA')):
    try:
        prefs.compute_device_type = backend
        prefs.get_devices()
        if any(d.type == backend for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == backend
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
fov = 2 * math.atan(cam.data.sensor_width / 2 / cam.data.lens)
dist = max(size) / 2 * 1.15 / math.tan(fov / 2)
yaw, pitch = math.radians(-90 + yaw_deg), math.radians(10)
cam.location = center + Vector((math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch))) * dist
cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()

tmp = []
for i, f in enumerate(frames):
    frame_set(f)
    scene.render.filepath = f'{out}.f{i}.png'
    bpy.ops.render.render(write_still=True)
    tmp.append(scene.render.filepath)

# stitch with Blender's image API + numpy (no PIL inside Blender)
import numpy as np
W = res * len(tmp)
strip = bpy.data.images.new('strip', W, res, alpha=False)
px = np.zeros((res, W, 4), dtype=np.float32)
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
print('WROTE', out, 'frames', frames)

if 'asset_result' in globals():
    asset_result({'output': out})
