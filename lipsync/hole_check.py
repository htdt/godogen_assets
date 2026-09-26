"""
Hole check for a mouth-rigged GLB: render the face from 8 directions with backface culling (as glTF viewers draw it)
on a transparent film, paint everything see-through magenta. Any magenta spot on the face is a hole.

    asset-blender lipsync/hole_check.py -- mouth.glb out.png [--jaw 0.6] [--zoom]

--jaw     jaw opening 0..1 (fraction of face.json jaw_open_max_rad); holes inside the mouth only show when it is open
--zoom    frame the mouth (lips to chin) instead of the lower face
The GLB needs its <name>.face.json next to it (lipsync/mouth_rig.py writes it); a rig without a jaw bone (the MIA input,
for a before/after comparison) works with the same JSON passed as --face path.json.
Views: front, ±30° (zoom) / ±35°, ±55° / ±80°, from below, from above, 3/4 below; writes one 4 × 2 sheet.
"""
import bpy
import sys
import os
import json
import math
import numpy as np
from mathutils import Vector, Quaternion

argv = sys.argv[sys.argv.index('--') + 1:]
src, out = argv[0], argv[1]
jaw_open = float(argv[argv.index('--jaw') + 1]) if '--jaw' in argv else 0.0
zoom = '--zoom' in argv
face_json = argv[argv.index('--face') + 1] if '--face' in argv else os.path.splitext(src)[0] + '.face.json'
face = json.load(open(face_json))
RES = 320

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src, disable_bone_shape=True)
sc = bpy.context.scene
arm = next(o for o in sc.objects if o.type == 'ARMATURE')
if arm.animation_data:
    arm.animation_data.action = None
jb = arm.pose.bones.get(face['jaw_bone'])
if jb is not None and jaw_open:
    jb.rotation_mode = 'QUATERNION'
    jb.rotation_quaternion = Quaternion((1, 0, 0), jaw_open * face['jaw_open_max_rad'])
for m in bpy.data.materials:
    m.use_backface_culling = True
sc.render.engine = 'BLENDER_EEVEE_NEXT'
sc.eevee.taa_render_samples = 8
sc.render.resolution_x = sc.render.resolution_y = RES
sc.render.film_transparent = True
sc.view_settings.view_transform = 'Standard'
w = bpy.data.worlds.new('w')
sc.world = w
w.use_nodes = True
w.node_tree.nodes['Background'].inputs[1].default_value = 0.8
sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN'))
sun.data.energy = 2
sc.collection.objects.link(sun)

fr = face['frame']
left, up, fwd = (Vector(fr[k]) for k in ('left', 'up', 'fwd'))
c = Vector(face['lip_center_world'])
lc = face['report']['chin_below_lips']
target, size = (c, 2.2 * lc) if zoom else (c + up * 1.2 * lc, 5.5 * lc)
views = ([(0, 0), (30, 0), (-30, 0), (55, 0), (-55, 0), (0, -30), (0, 30), (20, -20)] if zoom else
         [(0, 5), (35, 5), (-35, 5), (80, 0), (-80, 0), (0, -35), (25, 35), (180, 5)])
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
cam.data.lens = 50
sc.collection.objects.link(cam)
sc.camera = cam
dist = size / 2 / math.tan(2 * math.atan(cam.data.sensor_width / 2 / cam.data.lens) / 2)
sheet = np.zeros((RES * 2, RES * 4, 4), np.float32)
for i, (yaw, pitch) in enumerate(views):
    y, p = math.radians(yaw), math.radians(pitch)
    d = (fwd * math.cos(y) + left * math.sin(y)) * math.cos(p) + up * math.sin(p)
    cam.location = target + d.normalized() * dist
    cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sun.rotation_euler = cam.rotation_euler
    sc.render.filepath = f'{out}.{i}.png'
    bpy.ops.render.render(write_still=True)
    im = bpy.data.images.load(sc.render.filepath)
    a = np.empty(RES * RES * 4, np.float32)
    im.pixels.foreach_get(a)
    a = a.reshape(RES, RES, 4)
    al = a[..., 3:4]
    a = a * al + np.array([1, 0, 1, 1], np.float32) * (1 - al)  # see-through -> magenta
    a[..., 3] = 1
    r_, c_ = divmod(i, 4)
    sheet[(1 - r_) * RES:(2 - r_) * RES, c_ * RES:(c_ + 1) * RES] = a
    os.remove(sc.render.filepath)
img = bpy.data.images.new('sheet', RES * 4, RES * 2, alpha=False)
img.pixels.foreach_set(sheet.ravel())
img.filepath_raw = out
img.file_format = 'PNG'
img.save()
print('WROTE', out)

if 'asset_result' in globals():
    asset_result({'output': out})
