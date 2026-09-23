"""
Render a GLB from 4 angles (textured + clay) with Blender Cycles and print mesh stats as JSON.

    asset-blender tools/render_glb.py -- model.glb out_dir [--res 512] [--samples 32]

Writes out_dir/<name>_{tex,clay}_{0..3}.png and out_dir/<name>_stats.json. Rendering the exported file (not
TRELLIS's internal representation) checks what a game engine or DCC tool will actually receive.
"""
import bpy
import bmesh
import sys
import os
import json
import math
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
glb, out_dir = argv[0], argv[1]
res = int(argv[argv.index('--res') + 1]) if '--res' in argv else 512
samples = int(argv[argv.index('--samples') + 1]) if '--samples' in argv else 32
views = int(argv[argv.index('--views') + 1]) if '--views' in argv else 4
name = os.path.splitext(os.path.basename(glb))[0]
os.makedirs(out_dir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']

# ---- stats ----
stats = {'file': glb, 'mb': round(os.path.getsize(glb) / 2**20, 2), 'objects': len(meshes),
         'verts': 0, 'faces': 0, 'tris': 0, 'non_manifold_edges': 0, 'boundary_edges': 0, 'loose_parts': 0,
         'textures': []}
dg = bpy.context.evaluated_depsgraph_get()
for o in meshes:
    bm = bmesh.new()
    bm.from_mesh(o.data)
    # glTF splits vertices along UV seams; weld them so topology stats describe the surface itself
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    stats['verts'] += len(bm.verts)
    stats['faces'] += len(bm.faces)
    stats['tris'] += sum(len(f.verts) - 2 for f in bm.faces)
    stats['boundary_edges'] += sum(1 for e in bm.edges if e.is_boundary)
    stats['non_manifold_edges'] += sum(1 for e in bm.edges if not e.is_manifold and not e.is_boundary)
    # connected components
    seen = set()
    for v in bm.verts:
        if v.index in seen:
            continue
        stats['loose_parts'] += 1
        stack = [v]
        seen.add(v.index)
        while stack:
            cur = stack.pop()
            for e in cur.link_edges:
                w = e.other_vert(cur)
                if w.index not in seen:
                    seen.add(w.index)
                    stack.append(w)
    bm.free()
for img in bpy.data.images:
    if img.size[0]:
        stats['textures'].append(f'{img.name}:{img.size[0]}x{img.size[1]}')

# ---- normalize: center at origin, fit in unit sphere ----
pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
center, size = (lo + hi) / 2, (hi - lo)
stats['bbox'] = [round(v, 4) for v in size]
radius = max(size) / 2 * 1.2

# ---- scene ----
scene = bpy.context.scene
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
scene.cycles.samples = samples
scene.cycles.use_denoising = True
scene.render.resolution_x = scene.render.resolution_y = res
scene.render.film_transparent = False
scene.view_settings.view_transform = 'Standard'

world = bpy.data.worlds.new('w')
scene.world = world
world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (1, 1, 1, 1)
world.node_tree.nodes['Background'].inputs[1].default_value = 0.6

def add_light(rot, energy):
    d = bpy.data.lights.new('sun', 'SUN')
    d.energy = energy
    ob = bpy.data.objects.new('sun', d)
    ob.rotation_euler = rot
    scene.collection.objects.link(ob)

add_light((math.radians(50), 0, math.radians(30)), 3.0)
add_light((math.radians(60), 0, math.radians(200)), 1.0)

cam_data = bpy.data.cameras.new('cam')
cam_data.lens = 50
cam = bpy.data.objects.new('cam', cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
fov = 2 * math.atan(cam_data.sensor_width / 2 / cam_data.lens)
dist = radius / math.tan(fov / 2)

clay = bpy.data.materials.new('clay')
clay.use_nodes = True
clay.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.22, 0.22, 0.24, 1)
clay.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.55

# glTF is +Y up / -Z forward; Blender imports it Z-up with the model's front facing -Y.
for mode in ('tex', 'clay'):
    scene.view_layers[0].material_override = clay if mode == 'clay' else None
    for k in range(views):
        yaw = math.radians(-90 + k * 360 / views)  # k=0: camera on -Y looking at the front
        pitch = math.radians(15)
        cam.location = center + Vector((math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch))) * dist
        cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()
        scene.render.filepath = os.path.join(out_dir, f'{name}_{mode}_{k}.png')
        bpy.ops.render.render(write_still=True)

with open(os.path.join(out_dir, f'{name}_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)
print('STATS ' + json.dumps(stats))
