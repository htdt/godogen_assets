"""
Lip-sync a mouth-rigged character (lipsync/mouth_rig.py output) to a speech clip.

    asset-blender lipsync/lipsync.py -- mouth.glb speech.wav out.glb
        [--text transcript.txt]   (helps Rhubarb's recognizer; optional)
        [--phonetic]              (Rhubarb's language-independent recognizer, for non-English lines)
        [--cues cues.json]        (Rhubarb JSON made elsewhere; skips running Rhubarb)
        [--fps 30] [--action speak]
        [--face face.json]        (default: <mouth.glb without .glb>.face.json; pass it when the input is an earlier
                                   output, e.g. to add a second line: lipsync.py -- x_lines.glb line2.wav x_lines.glb
                                   --face x_mouth.face.json --action line_02)
        [--render out.mp4 [--res 512] [--head-motion] [--engine eevee|cycles]]

Audio -> Rhubarb Lip Sync (mouth shapes A-H, X) -> per-frame targets for the jaw bone and the "wide" / "round" shape
keys (VISEMES below) -> light smoothing -> one glTF animation (jaw rotation + morph weights) in out.glb. The clip only
touches the jaw bone and the morph targets, so a game can layer it over any body animation.

--render draws the clip twice (face close-up and head-and-shoulders, arms lowered) and muxes the audio with ffmpeg.
--head-motion adds a small head sway in the render only (not in out.glb).
"""
import bpy
import sys
import os
import json
import math
import subprocess
import numpy as np
from mathutils import Vector, Quaternion, Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
RHUBARB = os.path.join(HERE, 'rhubarb', 'rhubarb')
argv = sys.argv[sys.argv.index('--') + 1:]
src, wav, out = argv[0], argv[1], argv[2]
opt = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d  # noqa: E731
fps = int(opt('--fps', 30))
action_name = opt('--action', 'speak')
render_to = opt('--render')
res = int(opt('--res', 512))
engine = opt('--engine', 'eevee')
face = json.load(open(opt('--face', os.path.splitext(src)[0] + '.face.json')))

# Rhubarb shapes (https://github.com/DanielSWolf/rhubarb-lip-sync#mouth-shapes) -> (jaw open 0..1, wide, round)
VISEMES = {
    'X': (0.00, 0.0, 0.0),   # rest / silence
    'A': (0.00, 0.0, 0.0),   # M B P: lips closed
    'B': (0.10, 0.6, 0.0),   # K S T EE: teeth nearly together, lips apart
    'C': (0.40, 0.35, 0.0),  # EH AE: open
    'D': (0.75, 0.15, 0.0),  # AA: wide open
    'E': (0.35, 0.0, 0.5),   # AO ER: slightly rounded
    'F': (0.12, 0.0, 1.0),   # UW OW W: puckered
    'G': (0.06, 0.3, 0.0),   # F V: upper teeth on the lower lip
    'H': (0.35, 0.2, 0.0),   # L: tongue up, mouth open
}

# ---- cues ----
cues_path = opt('--cues')
if cues_path is None:
    cues_path = os.path.splitext(out)[0] + '.rhubarb.json'
    cmd = [RHUBARB, '-q', '-f', 'json', '--extendedShapes', 'GHX', '-o', cues_path]
    if opt('--text'):
        cmd += ['-d', opt('--text')]
    if '--phonetic' in argv:
        cmd += ['-r', 'phonetic']
    subprocess.run(cmd + [wav], check=True)
cues = json.load(open(cues_path))['mouthCues']
duration = max(c['end'] for c in cues)
nfr = int(math.ceil(duration * fps)) + 1

# per-frame targets (value of the cue active at the frame time), then a short Gaussian (~35 ms) so shapes blend
# instead of popping; the jaw gets a bit more smoothing than the lips (it is the heavier articulator)
T = np.arange(nfr) / fps
tgt = np.zeros((nfr, 3))
for c in cues:
    tgt[(T >= c['start']) & (T < c['end'])] = VISEMES.get(c['value'], VISEMES['X'])


def smooth(x, sigma_s):
    s = sigma_s * fps
    r = int(math.ceil(3 * s))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode='edge'), k, mode='valid')


curves = np.stack([smooth(tgt[:, 0], 0.045), smooth(tgt[:, 1], 0.035), smooth(tgt[:, 2], 0.035)], 1)

# ---- scene ----
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
scene = bpy.context.scene
scene.render.fps = fps
scene.frame_start, scene.frame_end = 1, nfr
arm = next(o for o in scene.objects if o.type == 'ARMATURE')
meshes = [o for o in scene.objects if o.type == 'MESH' and (o.parent is not None or len(o.data.vertices) > 100)]
for o in [o for o in scene.objects if o.type == 'MESH' and o not in meshes]:
    bpy.data.objects.remove(o)
body = max(meshes, key=lambda o: len(o.data.vertices))
jaw = arm.pose.bones[face['jaw_bone']]
jaw.rotation_mode = 'QUATERNION'
keys = body.data.shape_keys
JMAX = face['jaw_open_max_rad']

# the input may already carry body clips; a tool that keys every joint keys the jaw too (at rest). Drop their jaw
# channels so the speech layer alone owns the jaw when a game plays both.
def fcurve_sets(a):
    if getattr(a, 'layers', None):
        for layer in a.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    yield cb.fcurves
    else:
        yield a.fcurves


n_stripped = 0
for other in list(bpy.data.actions):
    if any(fc.data_path.startswith('key_blocks') for fcs in fcurve_sets(other) for fc in fcs):
        continue  # an earlier speech clip (it drives the shape keys): keep its jaw
    for fcs in fcurve_sets(other):
        for fc in [fc for fc in fcs if f'pose.bones["{face["jaw_bone"]}"]' in fc.data_path]:
            fcs.remove(fc)
            n_stripped += 1
if n_stripped:
    print('removed %d jaw channels from %d existing clips' % (n_stripped, len(bpy.data.actions)))

# one action with two slots (armature + shape keys): the glTF exporter writes it as one animation
act = bpy.data.actions.new(action_name)
arm.animation_data_create().action = act
keys.animation_data_create().action = act
for f in range(nfr):
    jaw.rotation_quaternion = Quaternion((1, 0, 0), float(curves[f, 0]) * JMAX)
    jaw.keyframe_insert('rotation_quaternion', frame=f + 1)
    for k, name in ((1, 'wide'), (2, 'round')):
        kb = keys.key_blocks[name]
        kb.value = float(np.clip(curves[f, k], 0, 1))
        kb.keyframe_insert('value', frame=f + 1)
# no sampling: only the keyed channels (jaw + morph weights) go into the clip, not every bone of the skeleton
bpy.ops.export_scene.gltf(filepath=out, export_animations=True, export_animation_mode='ACTIONS', export_morph=True,
                          export_morph_normal=False, export_try_sparse_sk=True, export_force_sampling=False)
print('WROTE', out, 'frames', nfr, 'fps', fps, 'cues', len(cues))
if not render_to:
    sys.exit(0)

# ---- render: arms down, optional head sway, two shots ----
fr = face['frame']
left, up, fwd = (Vector(fr[k]) for k in ('left', 'up', 'fwd'))
Wm = arm.matrix_world


def bone(short):
    return next(pb for pb in arm.pose.bones if pb.name.split(':')[-1] == short)


bpy.context.view_layer.update()
for sd, sgn in (('Left', 1), ('Right', -1)):
    pb = bone(f'{sd}Arm')
    cur = Wm @ pb.matrix
    cdir = (cur.to_3x3() @ Vector((0, 1, 0))).normalized()
    q = cdir.rotation_difference((left * (0.25 * sgn) - up).normalized())
    new = Matrix.Translation(cur.translation) @ q.to_matrix().to_4x4() @ Matrix.Translation(-cur.translation) @ cur
    pb.matrix = Wm.inverted() @ new
    bpy.context.view_layer.update()
if '--head-motion' in argv:
    head = bone('Head')
    head.rotation_mode = 'XYZ'
    t = np.arange(nfr) / fps
    energy = smooth(np.clip(curves[:, 0] * 2, 0, 1), 0.4)  # nod a little more while talking
    for f in range(nfr):
        head.rotation_euler = (math.radians(1.5 * math.sin(2.1 * t[f]) + 2.5 * energy[f] * math.sin(5.3 * t[f])),
                               math.radians(2.0 * math.sin(0.9 * t[f] + 1.0)),
                               math.radians(1.2 * math.sin(1.3 * t[f] + 2.0)))
        head.keyframe_insert('rotation_euler', frame=f + 1)

if engine == 'cycles':
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
else:
    scene.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in [e.identifier for e in
                                                   bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items] \
        else 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = 32
scene.render.resolution_x = scene.render.resolution_y = res
scene.view_settings.view_transform = 'Standard'
world = bpy.data.worlds.new('w')
scene.world = world
world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (0.8, 0.82, 0.85, 1)
world.node_tree.nodes['Background'].inputs[1].default_value = 0.7
key = bpy.data.objects.new('key', bpy.data.lights.new('key', 'SUN'))
key.data.energy = 2.5
key.data.angle = math.radians(10)
key.rotation_euler = (-(fwd * 0.8 + up * 0.6 + left * 0.35)).to_track_quat('-Z', 'Y').to_euler()
scene.collection.objects.link(key)
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
scene.collection.objects.link(cam)
scene.camera = cam
lc = face['report']['chin_below_lips']  # lips to chin: a face-size unit that also fits cartoon proportions
lipc = Vector(face['lip_center_world'])
fov_of = lambda lens: 2 * math.atan(cam.data.sensor_width / 2 / lens)  # noqa: E731
SHOTS = [('close', 85, lipc + up * (1.2 * lc), 5.2 * lc, 20),     # face: chin to hairline
         ('medium', 50, lipc - up * (1.8 * lc), 13.0 * lc, 12)]   # head and shoulders
tmpdir = os.path.splitext(render_to)[0] + '_frames'
os.makedirs(tmpdir, exist_ok=True)
scene.render.image_settings.file_format = 'PNG'
for name, lens, target, size, yaw in SHOTS:
    cam.data.lens = lens
    dist = size / 2 / math.tan(fov_of(lens) / 2)
    y = math.radians(yaw)
    d = (fwd * math.cos(y) + left * math.sin(y) + up * 0.05).normalized()
    cam.location = target + d * dist
    cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()
    scene.render.filepath = os.path.join(tmpdir, name + '_')
    bpy.ops.render.render(animation=True)
w0 = os.path.join(tmpdir, 'close_%04d.png')
w1 = os.path.join(tmpdir, 'medium_%04d.png')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', w0, '-framerate', str(fps), '-i', w1,
                '-i', wav, '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]', '-map', '[v]', '-map', '2:a',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-c:a', 'aac', '-b:a', '160k', '-shortest',
                render_to], check=True)
print('WROTE', render_to)
