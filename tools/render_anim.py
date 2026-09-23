"""
Render evenly spaced frames of an animated, skinned GLB/FBX with Blender (to check rigging/skinning quality).

    asset-blender tools/render_anim.py -- anim.glb out.png [--frames 8] [--res 384] [--yaw 30]
        [--action NAME]      (a GLB with several clips: render the one named NAME, e.g. a Kimodo move)

Writes one horizontal strip PNG (frame i left to right). The camera is fixed and framed on the whole animation's
bounding box so root motion stays visible.
"""
import bpy
import sys
import os
import math
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
src, out = argv[0], argv[1]
nframes = int(argv[argv.index('--frames') + 1]) if '--frames' in argv else 8
res = int(argv[argv.index('--res') + 1]) if '--res' in argv else 384
yaw_deg = float(argv[argv.index('--yaw') + 1]) if '--yaw' in argv else 30.0
want = argv[argv.index('--action') + 1] if '--action' in argv else None

bpy.ops.wm.read_factory_settings(use_empty=True)
if src.lower().endswith('.fbx'):
    bpy.ops.import_scene.fbx(filepath=src)
else:
    bpy.ops.import_scene.gltf(filepath=src)
scene = bpy.context.scene
# skip the glTF importer's bone display shapes (a 1 m icosphere at the origin that would skew the framing)
meshes = [o for o in scene.objects if o.type == 'MESH' and len(o.data.vertices) > 100]
arms = [o for o in scene.objects if o.type == 'ARMATURE']

# frame range from the armature's action (glTF import may put it in NLA tracks)
f0, f1 = scene.frame_start, scene.frame_end
for a in arms:
    ad = a.animation_data
    act = ad.action if ad and ad.action else None
    if want:
        act = next((x for x in bpy.data.actions if x.name == want or x.name.startswith(want + '_')), None)
        if act is None:
            sys.exit(f'no action {want!r}; have: {[x.name for x in bpy.data.actions]}')
        ad = ad or a.animation_data_create()
        for tr in ad.nla_tracks:
            tr.mute = True
        ad.action = act
        if hasattr(act, 'slots') and act.slots:
            ad.action_slot = act.slots[0]
    if act is None and ad and ad.nla_tracks:
        for tr in ad.nla_tracks:
            for st in tr.strips:
                act = st.action
                ad.action = act
                tr.mute = True
                break
            if act:
                break
    if act:
        f0, f1 = int(act.frame_range[0]), int(act.frame_range[1])
frames = [round(f0 + (f1 - f0) * i / max(nframes - 1, 1)) for i in range(nframes)]

# bounding box over all sampled frames (evaluated, i.e. deformed)
lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
for f in frames:
    scene.frame_set(f)
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
    scene.frame_set(f)
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
