"""
Add a speaking mouth to a rigged TRELLIS.2 / Make-It-Animatable character (Mixamo skeleton with a Head bone).

    asset-blender lipsync/mouth_rig.py -- rigged.glb out.glb [--work dir] [--hinge-depth 0.62]

All automatic, no per-character tuning (write-up: lipsync/README.md). The numbered sections below:
   1. weld the glTF seam duplicates (smooth normals instead of the imported custom normals);
   2. render the head from the front (orthographic Cycles), find 478 face landmarks with MediaPipe
      (face_landmarks.py in lipsync/.venv) and ray-cast them onto the mesh;
   3. face frame from the landmarks: lip-line curve, mouth width (the size unit), chin, nose base, eyes, face width;
   4. resurface: TRELLIS closes mouths with folds and inner walls, so the mouth area (nose to chin, past the corners)
      is deleted down to 1 mouth width behind the skin and refilled with the visible skin re-sampled as a height field
      (constrained Delaunay stitched to the traced outline, own re-sampled texture);
   5. densify (only if 4 was skipped): refine long edges around the mouth (red-green, no flipped slivers);
   6. cut the closed lips along the landmark lip line (level-set cut through the triangles, visible crossings only),
      split the cut, label the two paths between the corners upper / lower lip;
   7. fallback clean-up: delete hidden faces behind the lips inside the future cavity;
   8. jaw skin weights: harmonic (lower lip + chin = 1, above/behind the hinge and the neck = 0), then hidden layers
      take the weight of the skin in front of them; taken out of Head + Neck;
   9. close the opening inwards: a short inner lip band that curls in (solid lip colour from the texture) and a dark
      mouth cavity capped at the back, kept inside the head;
  10. jaw bone (child of Head, hinge in front of the ears, tail at the chin, +X local rotation opens);
  11. procedural teeth (upper row on Head, lower row on the jaw) and a tongue (jaw), sized from the mouth width, with
      baked occlusion as vertex colours;
  12. lip shape keys (glTF morph targets): "wide" (corners out, lips part) and "round" (corners in, lips forward);
  13. export the GLB (T-pose bind, original skeleton + jaw bone, no animation) and <out>.face.json (bone names, jaw
      axis and range, face frame, measurements, the step report).

Jaw convention: rotating the jaw bone about its local +X axis opens the mouth; face.json "jaw_open_max_rad" is the
full opening (0.30 rad for realistic faces, more for wide cartoon mouths).
"""
import bpy
import bmesh
import sys
import os
import json
import math
import subprocess
from collections import defaultdict
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from mathutils.geometry import delaunay_2d_cdt

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[sys.argv.index('--') + 1:]
src, out = argv[0], argv[1]
work = argv[argv.index('--work') + 1] if '--work' in argv else os.path.splitext(out)[0] + '_mouth'
HINGE_K = float(argv[argv.index('--hinge-depth') + 1]) if '--hinge-depth' in argv else 0.62
FACE_IMAGE = argv[argv.index('--face-image') + 1] if '--face-image' in argv else None
os.makedirs(work, exist_ok=True)
report = {'src': src}

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src, disable_bone_shape=True)
scene = bpy.context.scene
arm = next(o for o in scene.objects if o.type == 'ARMATURE')
body = max((o for o in scene.objects if o.type == 'MESH'), key=lambda o: len(o.data.vertices))
if arm.animation_data:
    arm.animation_data.action = None
for pb in arm.pose.bones:
    pb.matrix_basis = Matrix.Identity(4)
bpy.context.view_layer.update()
me = body.data


def bone_name(short):
    return next(b.name for b in arm.data.bones if b.name.split(':')[-1] == short)


HEAD = bone_name('Head')
NECK = bone_name('Neck')
JAW = HEAD[:-len('Head')] + 'Jaw'
MW = body.matrix_world.copy()
MWi = MW.inverted()
Wa = arm.matrix_world.copy()


def np_world(vs):
    M = np.array(MW)
    return np.array([v.co for v in vs]) @ M[:3, :3].T + M[:3, 3]


# ---- 0. character frame: up = +Z, forward from the feet (as in pose_test.py), left = up x forward ----
def bone_dir(short):
    b = arm.data.bones[bone_name(short)]
    return (Wa.to_3x3() @ (b.tail_local - b.head_local)).normalized()


up = Vector((0, 0, 1))
fwd = sum((bone_dir(f'{s}Foot') + bone_dir(f'{s}ToeBase') for s in ('Left', 'Right')), Vector())
fwd.z = 0
fwd.normalize()
left = up.cross(fwd).normalized()
if (Wa @ arm.data.bones[bone_name('LeftArm')].head_local - Wa @ arm.data.bones[bone_name('RightArm')].head_local).dot(left) < 0:
    left = -left
Lv, Uv, Fv = (np.array(v) for v in (left, up, fwd))

# ---- 1. weld: glTF splits vertices along every UV-chart border, and MIA's pose reset moved the duplicates up to
# ~0.5 mm apart; welding gives the connected surface the cut, the extrusion and the weight diffusion need (the glTF
# exporter splits seams again). Custom normals go (they would be wrong around the new geometry): smooth shading. ----
bm = bmesh.new()
bm.from_mesh(me)
dl = bm.verts.layers.deform.verify()
hi_, ni_ = body.vertex_groups[HEAD].index, body.vertex_groups[NECK].index
height_local = (MWi.to_3x3() @ Vector((0, 0, 1.0))).length  # local units per metre
n0 = len(bm.verts)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0006 * height_local)
report['weld'] = [n0, len(bm.verts)]
# game-budget input (e.g. gen3d --faces 30000): coarser teeth and densify, so the mouth adds ~10 % instead of ~25 %
LOWPOLY = len(bm.faces) < 100000
bm.to_mesh(me)
for name in ('custom_normal', 'sharp_edge', 'sharp_face'):
    if name in me.attributes:
        me.attributes.remove(me.attributes[name])
me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
me.update()
bm.free()
print('welded', report['weld'])

nv = len(me.vertices)
co = np.empty(nv * 3)
me.vertices.foreach_get('co', co)
M = np.array(MW)
X = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
nrm = np.empty(nv * 3)
me.vertices.foreach_get('normal', nrm)
Nw = nrm.reshape(-1, 3) @ np.linalg.inv(M[:3, :3])  # normals transform with the inverse transpose (rows here)
Nw /= np.linalg.norm(Nw, axis=1, keepdims=True) + 1e-12
w_head = np.zeros(nv)
w_neck = np.zeros(nv)
for v in me.vertices:
    for g in v.groups:
        if g.group == hi_:
            w_head[v.index] = g.weight
        elif g.group == ni_:
            w_neck[v.index] = g.weight
# the head's extent for the front render: head-weighted skin near the Head bone's axis. Rigid parts skinned to the
# head out to the sides (pauldron spikes, a helmet's horns) would otherwise shrink the face in the frame.
hb = np.array(Wa @ arm.data.bones[HEAD].head_local)
hp = (X[w_head > 0.5] - hb) @ np.array([Lv, Uv, Fv]).T
top = hp[:, 1].max()
hp = hp[(np.abs(hp[:, 0]) < 0.8 * top) & (np.abs(hp[:, 2]) < top)]
lo, hi = hp.min(0), hp.max(0)
head_center = Vector(hb + ((lo + hi) / 2) @ np.array([Lv, Uv, Fv]))
head_size = hi - lo

# ---- 2. front render + landmarks ----
RES = 1024
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
cam.data.type = 'ORTHO'
cam.data.ortho_scale = float(max(head_size[0], head_size[1])) * 1.1
cam.data.clip_end = 100
scene.collection.objects.link(cam)
scene.camera = cam
cam.location = head_center + fwd * 2.0
cam.rotation_euler = (-fwd).to_track_quat('-Z', 'Y').to_euler()
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
scene.render.resolution_x = scene.render.resolution_y = RES
scene.view_settings.view_transform = 'Standard'
world = bpy.data.worlds.new('w')
scene.world = world
world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (1, 1, 1, 1)
world.node_tree.nodes['Background'].inputs[1].default_value = 1.0
face_png = os.path.join(work, 'face_front.png')
scene.render.filepath = face_png
bpy.ops.render.render(write_still=True)

lm_json = os.path.join(work, 'landmarks_2d.json')
lm_src = FACE_IMAGE or face_png  # --face-image: an edit of face_front.png with the same layout (lipsync/README.md)
r = subprocess.run([os.path.join(HERE, '.venv', 'bin', 'python'), os.path.join(HERE, 'face_landmarks.py'),
                    lm_src, lm_json, '--debug', os.path.join(work, 'face_landmarks.png'), '--draw-on', face_png],
                   env={'HOME': os.environ['HOME'], 'PATH': '/usr/bin:/bin'})
if r.returncode != 0:
    sys.exit(f'landmark detection failed ({r.returncode}): no face found in {lm_src}')
lm_data = json.load(open(lm_json))
if (lm_data['width'], lm_data['height']) != (RES, RES):
    sys.exit(f'{lm_src} is {lm_data["width"]}x{lm_data["height"]}: a face image must keep face_front.png\'s '
             f'{RES}x{RES} layout')
lm2 = np.array(lm_data['points'])[:, :2]
cam_right = cam.matrix_world.to_3x3() @ Vector((1, 0, 0))
cam_up = cam.matrix_world.to_3x3() @ Vector((0, 1, 0))
S_ = cam.data.ortho_scale
polys = [list(p.vertices) for p in me.polygons]
bvh = BVHTree.FromPolygons([Vector(p) for p in X], polys)  # world space, welded surface before the cut
LM = np.full((len(lm2), 3), np.nan)
on_head = []
for i, (px, py) in enumerate(lm2):
    o = cam.location + cam_right * ((px / RES - 0.5) * S_) + cam_up * ((0.5 - py / RES) * S_)
    hit, _, poly, _ = bvh.ray_cast(o, -fwd)
    if hit is not None:
        LM[i] = hit
        on_head.append(w_head[polys[poly]].mean() + w_neck[polys[poly]].mean() > 0.5)
report['landmarks_on_mesh'] = int(np.isfinite(LM[:, 0]).sum())
report['landmarks_on_head'] = int(sum(on_head))
if sum(on_head) < 0.8 * len(lm2):  # a "face" found on something else: a skull on a pauldron, a shield
    sys.exit(f'the face found is not on the head ({sum(on_head)} of {len(lm2)} landmarks); see '
             f'{os.path.join(work, "face_landmarks.png")}')

# ---- 3. face frame: origin c0 = centre of the lip line; a = left, b = up, d = forward ----
UP_IN = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LO_IN = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
c0 = (LM[13] + LM[14]) / 2


def frame(P):
    Q = np.asarray(P) - c0
    return Q @ Lv, Q @ Uv, Q @ Fv


def to_world(a, b, d):
    return Vector(c0 + a * Lv + b * Uv + d * Fv)


mid = (LM[UP_IN] + LM[LO_IN]) / 2
am, bm_, dm = frame(mid)
order = np.argsort(am)
am, bm_, dm = am[order], bm_[order], dm[order]
aR, aL = am[0], am[-1]
a_mid = (aR + aL) / 2
mw = abs(frame(LM[291])[0] - frame(LM[61])[0])  # outer mouth corners: the size unit for everything below
hw = mw / 2
b_poly = np.polyfit(am, bm_, 4)
d_poly = np.polyfit(am, dm, 2)


def b_lip(a):
    return np.polyval(b_poly, np.clip(a, aR, aL))


def d_lip(a):
    return np.polyval(d_poly, np.clip(a, aR, aL))


fa = lambda i: frame(LM[i])  # noqa: E731
_, b_chin, d_chin = fa(152)
_, b_nb, _ = fa(2)  # nose base (subnasale)
b_eye = np.mean([fa(i)[1] for i in (33, 133, 362, 263)])
fw = abs(fa(454)[0] - fa(234)[0])  # face width at the cheeks
b_h = b_nb + 0.4 * (b_eye - b_nb)  # hinge height: ear tragus level, a bit below the middle of the nose
d_h = d_lip(a_mid) - HINGE_K * fw  # hinge depth: in front of the ear
hinge = to_world(a_mid, b_h, d_h)
chin = Vector(LM[152])
report.update(mouth_width=mw, face_width=fw, lip_fit_residual=float(np.abs(np.polyval(b_poly, am) - bm_).max() / mw),
              hinge_below_nosebase=float(b_nb - b_h), hinge_behind_lips=float(-d_h), chin_below_lips=float(-b_chin))
print('mouth width %.4f, face width %.4f, lip fit residual %.3f mw' % (mw, fw, report['lip_fit_residual']))


# ---- 4. resurface: TRELLIS closes mouths with folds and inner walls (a second skin layer behind the lips and chin).
# Replace the mouth region with a clean single-layer surface: delete every face there, the hidden layers included (down
# to 1 mouth width behind the skin), re-sample the visible skin as a height field seen from the front, stitch it to the
# untouched mesh around it (constrained Delaunay inside the hole's boundary loop) and give it its own small texture
# baked from the original (the atlas UVs cannot be interpolated across chart borders). ----
def image_pixels(img):
    px = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    return px.reshape(img.size[1], img.size[0], 4)


def bilinear(px, uv):
    h, w = px.shape[:2]
    x = np.clip((uv[:, 0] % 1.0) * w - 0.5, 0, w - 1.001)
    y = np.clip((uv[:, 1] % 1.0) * h - 0.5, 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return (px[y0, x0] * (1 - fx) * (1 - fy) + px[y0, x0 + 1] * fx * (1 - fy) +
            px[y0 + 1, x0] * (1 - fx) * fy + px[y0 + 1, x0 + 1] * fx * fy)


def barycentric(P, T0, T1, T2):
    v0, v1, v2 = T1 - T0, T2 - T0, P - T0
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01 + 1e-30
    v, w_ = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
    return np.clip(np.stack([1 - v - w_, v, w_], 1), 0, 1)


def resurface():
    polys_ = np.array([list(p.vertices) for p in me.polygons])
    if polys_.ndim != 2 or polys_.shape[1] != 3:
        return 'input is not all triangles'
    mat_idx = np.empty(len(me.polygons), dtype=np.int64)
    me.polygons.foreach_get('material_index', mat_idx)
    uv_all = np.empty(len(me.loops) * 2)
    me.uv_layers.active.data.foreach_get('uv', uv_all)
    uv_all = uv_all.reshape(-1, 2)
    lstart = np.empty(len(me.polygons), dtype=np.int64)
    me.polygons.foreach_get('loop_start', lstart)
    b_top = max(0.75 * b_nb, 0.25 * mw)          # below the nostrils
    b_bot = min(0.72 * b_chin, -0.35 * mw)       # above the chin tip
    RA, bc, RB = 1.45 * hw, (b_top + b_bot) / 2, (b_top - b_bot) / 2

    def shape(a, b):  # superellipse: < 1 inside the patch
        return ((a - a_mid) / RA) ** 4 + ((b - bc) / RB) ** 4

    def front(a, b):  # visible skin at (a, b): (world point, original face index)
        hit = bvh.ray_cast(to_world(a, b, fw), -fwd, 3 * fw)
        return (hit[0], hit[2]) if hit[0] is not None else (None, None)

    # 1. the visible skin inside the outline, grown from the lip centre (faces seen from the front)
    bm_ = bmesh.new()
    bm_.from_mesh(me)
    bm_.verts.ensure_lookup_table()
    bm_.faces.ensure_lookup_table()
    dl_ = bm_.verts.layers.deform.verify()
    uvl_ = bm_.loops.layers.uv.active
    Af, Bf, Df = frame(X[polys_].mean(1))
    inside_f = shape(Af, Bf) < 1
    front_f = np.zeros(len(polys_), bool)
    front_d = np.full(len(polys_), np.nan)
    for fi_ in np.nonzero(inside_f)[0]:
        hit = bvh.ray_cast(to_world(Af[fi_], Bf[fi_], fw), -fwd, 3 * fw)
        if hit[0] is not None:
            front_d[fi_] = fw - hit[3]
            front_f[fi_] = hit[2] == fi_ or abs(front_d[fi_] - Df[fi_]) < 0.05 * mw
    seed = bvh.ray_cast(Vector(c0 + Fv * fw), -fwd, 3 * fw)[2]
    if seed is None or not front_f[seed]:
        bm_.free()
        return 'no visible skin at the lip centre'
    R = {bm_.faces[seed]}
    stack = [bm_.faces[seed]]
    while stack:
        f = stack.pop()
        for e in f.edges:
            for f2 in e.link_faces:
                if f2 not in R and front_f[f2.index]:
                    R.add(f2)
                    stack.append(f2)
    # 2. its boundary, traced by walking around each vertex through the region's faces (exact at pinch points and
    # at TRELLIS's non-manifold spots, unlike chaining boundary edges)
    def rfaces(e):
        return [f for f in e.link_faces if f in R]

    def other_edge(f, v, e):
        return next(e2 for e2 in f.edges if e2 is not e and v in e2.verts)

    bnd = {e for f in R for e in f.edges if len(rfaces(e)) == 1}
    loops, used = [], set()
    for e0 in bnd:
        if e0 in used:
            continue
        f0 = rfaces(e0)[0]
        lp, e, f, v = [], e0, f0, e0.verts[1]
        ok = True
        for _ in range(100000):
            used.add(e)
            lp.append(v)
            e2, f2 = other_edge(f, v, e), f
            for _ in range(200):  # rotate around v through region faces until the next boundary edge
                if e2 in bnd:
                    break
                nxt = [g for g in rfaces(e2) if g is not f2]
                if not nxt:
                    break
                f2 = nxt[0]
                e2 = other_edge(f2, v, e2)
            else:
                ok = False
            if e2 not in bnd:
                ok = False
            if not ok:
                break
            e, f, v = e2, f2, e2.other_vert(v)
            if e is e0:
                break
        if ok and len(lp) > 8:
            loops.append(lp)
    if not loops:
        bm_.free()
        return 'could not trace the boundary of the mouth region'

    def area2d(lp):
        a, b, _ = frame(np_world(lp))
        return 0.5 * (np.dot(a, np.roll(b, -1)) - np.dot(b, np.roll(a, -1)))

    rim = max(loops, key=lambda lp: abs(area2d(lp)))
    if area2d(rim) < 0:  # counter-clockwise seen from the front
        rim = rim[::-1]
    if len(set(rim)) != len(rim):
        bm_.free()
        return 'boundary touches itself (pinched region)'
    ra_ = np.array([frame(np.array(MW @ v.co))[:2] for v in rim])
    la, lb = ra_[:, 0], ra_[:, 1]

    def in_poly(pa, pb):  # even-odd rule against the rim polygon
        res_ = np.zeros(len(pa), bool)
        xa, ya, xb, yb = la, lb, np.roll(la, -1), np.roll(lb, -1)
        for i in range(len(la)):
            res_ ^= ((ya[i] > pb) != (yb[i] > pb)) & (pa < (xb[i] - xa[i]) * (pb - ya[i]) / (yb[i] - ya[i] + 1e-30) + xa[i])
        return res_

    # 3. delete everything inside the rim polygon down to 1 mouth width behind the skin: the region itself, the folds
    # and inner walls behind it, and any skin islands the visibility test missed
    fin = np.nonzero(in_poly(Af, Bf))[0]
    rim_set = set(rim)
    dead_i = [fi_ for fi_ in fin if bm_.faces[fi_] in R or
              (np.isfinite(front_d[fi_]) and Df[fi_] > front_d[fi_] - 1.0 * mw and
               not any(v in rim_set for v in bm_.faces[fi_].verts))]  # faces outside the rim keep their rim vertices
    dead = [bm_.faces[fi_] for fi_ in dead_i]
    if len(dead) < 20:
        bm_.free()
        return 'no faces in the mouth region'
    dead_mat = np.bincount([f.material_index for f in dead]).argmax()
    rim_edges = [bm_.edges.get((rim[i], rim[(i + 1) % len(rim)])) for i in range(len(rim))]
    bmesh.ops.delete(bm_, geom=dead, context='FACES')
    if any(not v.is_valid for v in rim):
        bm_.free()
        return 'rim vertex lost'
    gaps = sum(1 for e in rim_edges if e is None or not e.is_valid)
    # 4. interior samples: hexagonal grid + two rows hugging the lip line (the lip cut will run between them)
    s = mw / (12 if LOWPOLY else 22)
    ga, gb = np.meshgrid(np.arange(la.min(), la.max(), s), np.arange(lb.min(), lb.max(), s * 0.866))
    ga = ga + (np.arange(ga.shape[0])[:, None] % 2) * s / 2
    ga, gb = ga.ravel(), gb.ravel()
    near_lip = (ga > aR - s) & (ga < aL + s) & (np.abs(gb - b_lip(ga)) < 0.7 * s)
    ga, gb = ga[~near_lip], gb[~near_lip]
    rr = np.arange(aR + s / 3, aL - s / 3, s / 1.5)
    ga = np.concatenate([ga, rr, rr])
    gb = np.concatenate([gb, b_lip(rr) + 0.3 * s, b_lip(rr) - 0.3 * s])
    keep = in_poly(ga, gb) & (np.min(np.hypot(ga[:, None] - la[None], gb[:, None] - lb[None]), 1) > 0.6 * s)
    ga, gb = ga[keep], gb[keep]
    nL = len(rim)
    pts2 = [Vector((float(a), float(b))) for a, b in zip(la, lb)] + [Vector((float(a), float(b))) for a, b in zip(ga, gb)]
    cdt_v, _, cdt_f, cdt_orig, _, _ = delaunay_2d_cdt(pts2, [(i, (i + 1) % nL) for i in range(nL)], [list(range(nL))],
                                                      1, 1e-9 * mw, True)
    # 5. interior vertices on the original visible skin; skin weights interpolated from the face they sit on
    groups = {}
    vmap = {}
    for k, (vc, orig) in enumerate(zip(cdt_v, cdt_orig)):
        if orig and min(orig) < nL:  # merged with a rim vertex
            vmap[k] = rim[min(orig)]
            continue
        p, fi_ = front(vc.x, vc.y)
        if p is None:
            continue
        tri = polys_[fi_]
        bw = barycentric(np.array([p]), X[tri[0]][None], X[tri[1]][None], X[tri[2]][None])[0]
        v = bm_.verts.new(MWi @ Vector(p))
        wsum = defaultdict(float)
        for c_ in range(3):
            if tri[c_] not in groups:
                groups[tri[c_]] = {g_.group: g_.weight for g_ in me.vertices[tri[c_]].groups}
            for g_, wt in groups[tri[c_]].items():
                wsum[g_] += wt * bw[c_]
        for g_, wt in wsum.items():
            if wt > 1e-4:
                v[dl_][g_] = wt
        vmap[k] = v
    # 6. patch texture: every texel re-sampled from the original atlas (base colour + metal/rough maps)
    mat0 = me.materials[int(dead_mat)]
    if not (mat0.use_nodes and any(n_.type == 'TEX_IMAGE' and n_.image for n_ in mat0.node_tree.nodes)):
        bm_.free()
        return 'no image texture on the face material'
    TS = 256 if LOWPOLY else 512
    a0, b0 = la.min() - s, lb.min() - s
    span = max(la.max() - la.min(), lb.max() - lb.min()) + 2 * s
    ta, tb = np.meshgrid(a0 + (np.arange(TS) + 0.5) / TS * span, b0 + (np.arange(TS) + 0.5) / TS * span)
    ta, tb = ta.ravel(), tb.ravel()
    hp, hf = np.zeros((TS * TS, 3)), np.full(TS * TS, -1)
    orig_ = c0 + ta[:, None] * Lv + tb[:, None] * Uv + fw * Fv
    down = -fwd
    for k in range(TS * TS):
        hit = bvh.ray_cast(Vector(orig_[k]), down, 3 * fw)
        if hit[0] is not None:
            hp[k], hf[k] = hit[0], hit[2]
    ok_ = hf >= 0
    tri_ = polys_[hf[ok_]]
    bt = barycentric(hp[ok_], X[tri_[:, 0]], X[tri_[:, 1]], X[tri_[:, 2]])
    atlas_uv = (uv_all[lstart[hf[ok_]][:, None] + np.arange(3)[None]] * bt[:, :, None]).sum(1)
    patch_mat = mat0.copy()
    patch_mat.name = 'face_patch'
    for node in [n_ for n_ in patch_mat.node_tree.nodes if n_.type == 'TEX_IMAGE' and n_.image]:
        out_px = np.zeros((TS * TS, 4), dtype=np.float32)
        out_px[ok_] = bilinear(image_pixels(node.image), atlas_uv)
        out_px[~ok_] = out_px[ok_].mean(0)                              # texels off the skin (outside the patch)
        img = bpy.data.images.new(f'{node.image.name}_mouth', TS, TS, alpha=True)
        img.colorspace_settings.name = node.image.colorspace_settings.name
        img.pixels.foreach_set(out_px.ravel())
        img.pack()
        node.image = img
    me.materials.append(patch_mat)
    patch_mi = len(me.materials) - 1
    n_new_f, n_fail = 0, defaultdict(int)
    for f_ in cdt_f:
        if any(k not in vmap for k in f_):
            n_fail['off_surface'] += 1
            continue
        vs = [vmap[k] for k in f_]
        if len(set(vs)) < 3:
            n_fail['degenerate'] += 1
            continue
        try:
            nf = bm_.faces.new(vs)  # counter-clockwise in (a, b) = facing the viewer, like the skin around it
        except ValueError:
            n_fail['exists'] += 1
            continue
        nf.material_index, nf.smooth = patch_mi, True
        for l_, k in zip(nf.loops, f_):
            l_[uvl_].uv = ((cdt_v[k].x - a0) / span, (cdt_v[k].y - b0) / span)
        n_new_f += 1
    open_rim = sum(1 for e in rim_edges if e is not None and e.is_valid and len(e.link_faces) < 2)
    bm_.to_mesh(me)
    me.update()
    bm_.free()
    report['resurface'] = {'deleted_faces': len(dead), 'new_faces': n_new_f, 'new_verts': len(vmap) - nL,
                           'rim_verts': nL, 'rim_gaps': gaps, 'open_rim_edges': open_rim, 'loops': len(loops),
                           'failed_faces': dict(n_fail),
                           'texture': TS}
    return None


why = resurface()
if why:
    print('WARNING: resurface skipped:', why)
    report['resurface'] = {'skipped': why}
else:
    print('resurface:', report['resurface'])
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', co)
    X = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    nrm = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('normal', nrm)
    Nw = nrm.reshape(-1, 3) @ np.linalg.inv(M[:3, :3])
    Nw /= np.linalg.norm(Nw, axis=1, keepdims=True) + 1e-12
    bvh = BVHTree.FromPolygons([Vector(p) for p in X], [list(p.vertices) for p in me.polygons])

# ---- 5. densify (only when the resurface was skipped): game-budget meshes (30k triangles) have only a few triangles across the mouth; refine the lips and
# chin until edges are shorter than mw / 10. Red-green refinement: each touched triangle is re-split explicitly in its
# own vertex order (bmesh subdivide's fill patterns + re-triangulation left flipped slivers = dark specks) ----
def refine(bm, edges):
    uvl_ = bm.loops.layers.uv.active
    dl_ = bm.verts.layers.deform.verify()
    edges = {e for e in edges if all(len(f.verts) == 3 for f in e.link_faces)}
    mid = {}
    for e in edges:
        va, vb = e.verts
        m = bm.verts.new((va.co + vb.co) / 2)
        wa, wb = dict(va[dl_].items()), dict(vb[dl_].items())
        for k in set(wa) | set(wb):
            m[dl_][k] = (wa.get(k, 0.0) + wb.get(k, 0.0)) / 2
        mid[e] = m
    todo = []
    for f in {f for e in edges for f in e.link_faces}:
        Ls = list(f.loops)
        P = [(l_.vert, l_[uvl_].uv.copy()) for l_ in Ls]
        Q = [(mid[l_.edge], (Ls[i][uvl_].uv + Ls[(i + 1) % 3][uvl_].uv) / 2) if l_.edge in mid else None
             for i, l_ in enumerate(Ls)]
        k = sum(q is not None for q in Q)
        if k == 3:
            tris = [(P[0], Q[0], Q[2]), (P[1], Q[1], Q[0]), (P[2], Q[2], Q[1]), (Q[0], Q[1], Q[2])]
        elif k == 1:
            i = next(i for i in range(3) if Q[i])
            tris = [(P[i], Q[i], P[(i + 2) % 3]), (Q[i], P[(i + 1) % 3], P[(i + 2) % 3])]
        else:
            j = next(i for i in range(3) if Q[i] is None)
            p0, p1, p2, q1, q2 = P[j], P[(j + 1) % 3], P[(j + 2) % 3], Q[(j + 1) % 3], Q[(j + 2) % 3]
            tris = [(q1, p2, q2)]
            if (p0[0].co - q1[0].co).length < (p1[0].co - q2[0].co).length:
                tris += [(p0, p1, q1), (p0, q1, q2)]
            else:
                tris += [(p0, p1, q2), (p1, q1, q2)]
        todo.append((f.material_index, f.smooth, tris))
    bmesh.ops.delete(bm, geom=list({f for e in edges for f in e.link_faces}), context='FACES_ONLY')
    for mi, smooth, tris in todo:
        for tri in tris:
            try:
                nf = bm.faces.new([vt for vt, _ in tri])
            except ValueError:  # duplicate face (non-manifold input)
                continue
            nf.material_index, nf.smooth = mi, smooth
            for l_, (_, uv) in zip(nf.loops, tri):
                l_[uvl_].uv = uv
    bmesh.ops.delete(bm, geom=[e for e in edges if e.is_valid and not e.link_faces], context='EDGES')
    return len(edges)


bm = bmesh.new()
bm.from_mesh(me)
target_len = mw / (8 if LOWPOLY else 10)
n_sub = 0
# the rebuilt area already has the right density; refining flat triangles outside it only adds shading facets
for _ in range(0 if report['resurface'].get('new_faces') else 5):
    Xd = np_world(bm.verts)
    Ad, Bd, Dd = frame(Xd)
    inreg = (np.abs(Ad - a_mid) < 1.5 * hw) & (Bd - b_lip(Ad) < 0.8 * mw) & (Bd - b_lip(Ad) > -1.3 * mw) & \
        (Dd > d_lip(Ad) - 0.6 * mw)
    bm.verts.index_update()
    long_e = [e for e in bm.edges if inreg[e.verts[0].index] and inreg[e.verts[1].index] and
              (Xd[e.verts[0].index] - Xd[e.verts[1].index]) @ (Xd[e.verts[0].index] - Xd[e.verts[1].index]) >
              target_len ** 2]
    if not long_e:
        break
    n_sub += refine(bm, long_e)
if n_sub:
    bm.to_mesh(me)
    me.update()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', co)
    X = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    nrm = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('normal', nrm)
    Nw = nrm.reshape(-1, 3) @ np.linalg.inv(M[:3, :3])
    Nw /= np.linalg.norm(Nw, axis=1, keepdims=True) + 1e-12
    bvh = BVHTree.FromPolygons([Vector(p) for p in X], [list(p.vertices) for p in me.polygons])
bm.free()
report['densify_edges_split'] = n_sub
print('densify: %d edges split' % n_sub)

A, B, D = frame(X)
S = B - b_lip(A)
Nf = Nw @ Fv

# ---- 6. level-set cut along the lip line ----
bm = bmesh.new()
bm.from_mesh(me)
bm.verts.ensure_lookup_table()
bm.edges.ensure_lookup_table()
dl = bm.verts.layers.deform.verify()
uvl = bm.loops.layers.uv.active
reg = (A > aR - 0.2 * mw) & (A < aL + 0.2 * mw) & (np.abs(S) < 0.25 * mw) & (np.abs(D - d_lip(A)) < 0.25 * mw) & (Nf > -0.2)
eps = 1e-4 * mw
S = np.where(np.abs(S) < eps, eps, S)
ev = np.empty(len(me.edges) * 2, dtype=np.int64)
me.edges.foreach_get('vertices', ev)
ev = ev.reshape(-1, 2)
i0, i1 = ev[:, 0], ev[:, 1]
t = S[i0] / (S[i0] - S[i1] + 1e-30)
ac = A[i0] + t * (A[i1] - A[i0])
cand = np.nonzero(reg[i0] & reg[i1] & (S[i0] * S[i1] < 0) & (ac > aR) & (ac < aL))[0]
# TRELLIS closes mouths with folds: the lip line plane also crosses inner layers behind the lips. Cut only the
# crossings seen from the front (the landmarks were found on that surface).
cross = []
for ei in cand:
    pc = X[i0[ei]] + t[ei] * (X[i1[ei]] - X[i0[ei]])
    hit = bvh.ray_cast(Vector(pc + Fv * fw), -fwd, 2 * fw)
    if hit[0] is not None and abs(hit[3] - fw) < 0.03 * mw:
        cross.append(ei)
report['cut_hidden_crossings_skipped'] = len(cand) - len(cross)
new_verts = []
for ei, e, v0 in [(ei, bm.edges[ei], bm.verts[i0[ei]]) for ei in cross]:  # splitting invalidates the index table
    p0, p1 = v0.co.copy(), e.other_vert(v0).co.copy()
    _, nvx = bmesh.utils.edge_split(e, v0, float(t[ei]))  # interpolates UVs and skin weights
    nvx.co = p0.lerp(p1, float(t[ei]))
    new_verts.append(nvx)
fmap = defaultdict(list)
for nvx in new_verts:
    for f in nvx.link_faces:
        fmap[f].append(nvx)
seam = []
for f, vs in fmap.items():
    if len(vs) == 2:
        _, loop = bmesh.utils.face_split(f, vs[0], vs[1])
        seam.append(loop.edge)
print('cut: %d edges crossed, %d seam edges' % (len(cross), len(seam)))
split = bmesh.ops.split_edges(bm, edges=seam)
bnd = [e for e in set(seam) | set(split['edges']) if e.is_valid and e.is_boundary]


seam_vs = {v for e in bnd for v in e.verts}


def side_of_face(f):
    """level-set sign of the face's original vertices (a face centre can fall on the other side of a curved lip line)"""
    s = [frame(np.array(MW @ v.co)) for v in f.verts if v not in seam_vs]
    s = sum(b - b_lip(a) for a, b, _ in s) if s else 0.0
    return 1 if s > 0 else -1


va_ = lambda v: frame(np.array(MW @ v.co))[0]  # noqa: E731
# the split seam is a closed loop of boundary edges: its extreme vertices along a are the mouth corners (not split),
# the two paths between them are the upper and lower lip edges. Each path is labelled by a majority vote of its faces
# (a single sliver face on the cut can be ambiguous; the whole path is not).
adj = defaultdict(list)
for e in bnd:
    a_, b_ = e.verts
    adj[a_].append((b_, e))
    adj[b_].append((a_, e))
comps, seen = [], set()
for v0 in adj:
    if v0 in seen:
        continue
    stack, comp = [v0], []
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        comp.append(v)
        stack += [w for w, _ in adj[v]]
    comps.append(comp)
report['seam_loops'] = len(comps)
if len(comps) != 1:
    print('WARNING: lip cut is not one clean opening: %d loops (keeping the widest)' % len(comps))
loop_c = max(comps, key=lambda c: max(map(va_, c)) - min(map(va_, c)))
if any(len(adj[v]) != 2 for v in loop_c):
    sys.exit('lip cut failed: the opening is not a simple loop')
cR, cL = min(loop_c, key=va_), max(loop_c, key=va_)
paths = []
for first, e0 in adj[cR]:
    path, edges, prev, cur = [cR, first], [e0], cR, first
    while cur is not cL:
        nxt, e_ = next((w, e_) for w, e_ in adj[cur] if w is not prev)
        prev, cur = cur, nxt
        path.append(cur)
        edges.append(e_)
    paths.append((sum(side_of_face(e_.link_faces[0]) for e_ in edges), path))
paths.sort(key=lambda x: -x[0])
upper, lower = paths[0][1], paths[1][1]
json.dump({'upper': [[list(frame(np.array(MW @ v.co))) for v in upper]],
           'lower': [[list(frame(np.array(MW @ v.co))) for v in lower]],
           'crossings': [[float(A[i0[e]] + t[e] * (A[i1[e]] - A[i0[e]])), float(D[i0[e]] + t[e] * (D[i1[e]] - D[i0[e]]))]
                         for e in cross],
           'mw': mw, 'aR': aR, 'aL': aL}, open(os.path.join(work, 'cut_debug.json'), 'w'), default=float)
report['seam_verts'] = [len(upper), len(lower)]
report['seam_side_votes'] = [paths[0][0], paths[1][0]]
corners = {upper[0], upper[-1]}

# ---- 7. fallback clean-up (does little after a resurface): the fold layers inside the future mouth opening would
# show through it (and stretch like a membrane): delete the
# hidden faces in the cavity volume. Visible skin is never touched; the lips themselves are < 0.15 mw deep.
seam_set = set(upper) | set(lower)
cavf = set()
bm.verts.ensure_lookup_table()
Av, Bv, Dv = frame(np_world(bm.verts))
for i in np.nonzero((np.abs(Av - a_mid) < 0.95 * hw) & (Bv - b_lip(Av) > -0.45 * mw) & (Bv - b_lip(Av) < 0.3 * mw) &
                    (Dv > d_lip(a_mid) - 1.3 * mw) & (Dv < d_lip(Av) - 0.15 * mw))[0]:
    cavf.update(bm.verts[i].link_faces)
dead = []
for f in cavf:
    if any(v in seam_set for v in f.verts):
        continue
    a, b, d = frame(np.array(MW @ f.calc_center_median()))
    hit = bvh.ray_cast(to_world(a, b, fw), -fwd, 2 * fw)
    # hidden, and hidden by the lips (not by the nose or a skin fold: those faces stay)
    if hit[0] is not None and (fw - hit[3]) > d + 0.05 * mw and \
            abs(frame(np.array(hit[0]))[1] - b_lip(frame(np.array(hit[0]))[0])) < 0.3 * mw:
        dead.append(f)
bmesh.ops.delete(bm, geom=dead, context='FACES')
report['hidden_faces_deleted'] = len(dead)
print('deleted %d hidden faces inside the mouth volume' % len(dead))

# ---- 8. jaw weights: harmonic diffusion on the cut surface ----
bm.verts.index_update()
bm.verts.ensure_lookup_table()
bm.normal_update()
n = len(bm.verts)
Xb = np_world(bm.verts)
Ab, Bb, Db = frame(Xb)
Sb = Bb - b_lip(Ab)
Nb = np.array([MW.to_3x3() @ v.normal for v in bm.verts])
Nb = (Nb / (np.linalg.norm(Nb, axis=1, keepdims=True) + 1e-12)) @ Fv
wh = np.array([v[dl].get(hi_, 0.0) for v in bm.verts])
wn = np.array([v[dl].get(ni_, 0.0) for v in bm.verts])
E = np.array([(e.verts[0].index, e.verts[1].index) for e in bm.edges])
dist = np.linalg.norm(Xb - c0, axis=1)
region = ((wh + wn) > 0.02) & (dist < 1.2 * fw)
b_neck = b_chin - 0.5 * abs(b_chin)
fixed0 = (~region) | (Bb > b_h) | (Db < d_h) | (Bb < b_neck) | ((Sb > 0.1 * mw) & (np.abs(Ab - a_mid) < 1.1 * hw))
fixed1 = (np.abs(Ab - a_mid) < 0.75 * hw) & (Sb < -0.08 * mw) & (Bb > b_chin + 0.1 * abs(b_chin)) & (Nb > 0.3) & \
    (Db > d_h + 0.5 * (d_lip(a_mid) - d_h))
for v in upper:
    if v not in corners:
        fixed0[v.index], fixed1[v.index] = True, False
for v in lower:
    if v not in corners:
        fixed1[v.index], fixed0[v.index] = True, False
fixed0 &= ~fixed1
free = ~(fixed0 | fixed1)
g = fixed1.astype(float)
Ef = E[free[E[:, 0]] | free[E[:, 1]]]
deg = np.bincount(Ef[:, 0], minlength=n) + np.bincount(Ef[:, 1], minlength=n)


def lap(x):  # graph Laplacian (uniform weights) restricted to edges touching free vertices
    return deg * x - np.bincount(Ef[:, 0], weights=x[Ef[:, 1]], minlength=n) - \
        np.bincount(Ef[:, 1], weights=x[Ef[:, 0]], minlength=n)


REG = 1e-4  # tiny pull to 0: keeps islands without any fixed vertex well-posed
fi = np.nonzero(free)[0]
rhs = -lap(g)[fi]
x = np.zeros(len(fi))
ext = np.zeros(n)


def op(xf):
    ext[:] = 0
    ext[fi] = xf
    return lap(ext)[fi] + REG * xf


r_ = rhs - op(x)
p = r_.copy()
rr = r_ @ r_
for it in range(5000):
    Ap = op(p)
    al = rr / (p @ Ap)
    x += al * p
    r_ -= al * Ap
    rr2 = r_ @ r_
    if math.sqrt(rr2) < 1e-6 * math.sqrt(len(fi)):
        break
    p = r_ + (rr2 / rr) * p
    rr = rr2
w = g.copy()
w[fi] = np.clip(x, 0, 1)
# hidden layers: TRELLIS closes the mouth with folds and inner walls (e.g. a second surface ~1 cm behind the chin,
# joined at the lips). Diffusion gives them in-between weights, so they lag behind and poke through the skin when the
# jaw opens. Every vertex hidden behind the skin takes the weight of the skin right in front of it.
bfaces = [[v.index for v in f.verts] for f in bm.faces]
bvh_cut = BVHTree.FromPolygons([Vector(p) for p in Xb], bfaces)
seam_idx = {v.index for v in upper + lower}
reach = 0.6 * mw
n_hidden = 0
for i in np.nonzero(region & (dist < 0.8 * fw))[0]:
    if i in seam_idx:
        continue
    hit = bvh_cut.ray_cast(Vector(Xb[i] + Fv * reach), Vector(-Fv), reach)
    if hit[0] is not None and hit[3] < reach - 0.02 * mw:
        fv = bfaces[hit[2]]
        k = 1 / (np.linalg.norm(Xb[fv] - np.array(hit[0]), axis=1) + 1e-9)
        w[i] = float(k @ w[fv] / k.sum())
        n_hidden += 1
report['weights'] = {'free': int(free.sum()), 'jaw_fixed': int(fixed1.sum()), 'cg_iters': it + 1, 'hidden': n_hidden}
print('jaw weights: %d free vertices, CG %d iterations' % (free.sum(), it + 1))

bpy_jaw_group = body.vertex_groups.new(name=JAW)
ji_ = bpy_jaw_group.index
for v in bm.verts:
    wj = w[v.index]
    if wj > 1e-3:
        d = v[dl]
        h, nk = d.get(hi_, 0.0), d.get(ni_, 0.0)
        d[ji_] = wj * (h + nk)
        if h:
            d[hi_] = h * (1 - wj)
        if nk:
            d[ni_] = nk * (1 - wj)

# ---- 9. inner lips + mouth cavity ----
# colours from the character's own lips: the texture a little away from the seam (TRELLIS paints a dark crease line
# right on it). The inner lip band gets a solid, darker lip colour (stretched texture streaks), the cavity a very dark one.
def base_color_image(m):
    tex = [n_ for n_ in m.node_tree.nodes if n_.type == 'TEX_IMAGE' and n_.image] if m and m.use_nodes else []
    linked = [n_ for n_ in tex if any(l_.to_socket.name == 'Base Color' for l_ in n_.outputs['Color'].links)]
    return (linked or tex or [None])[0]


lip_rgb = np.array([0.55, 0.25, 0.25])
lipv = np.nonzero((np.abs(Ab - a_mid) < 0.6 * hw) & (np.abs(Sb) > 0.04 * mw) & (np.abs(Sb) < 0.14 * mw) &
                  (Nb > 0.3) & (Db > d_lip(Ab) - 0.1 * mw))[0]
samples, pix_cache = [], {}
for i in lipv:
    for l_ in bm.verts[i].link_loops:  # each face samples its own material's texture (the resurfaced patch has one)
        mi = l_.face.material_index
        if mi not in pix_cache:
            node = base_color_image(me.materials[mi])
            pix_cache[mi] = (image_pixels(node.image), node.image.colorspace_settings.name == 'sRGB') if node else None
        if pix_cache[mi]:
            px_, srgb = pix_cache[mi]
            c_ = bilinear(px_, np.array([l_[uvl].uv]))[0, :3]
            samples.append(c_ ** 2.2 if srgb else c_)
if samples:
    lip_rgb = np.median(np.array(samples), axis=0)
report['lip_color_linear'] = lip_rgb.round(3).tolist()


def flat_mat(name, rgb, rough):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b_ = m.node_tree.nodes['Principled BSDF']
    b_.inputs['Base Color'].default_value = (*[float(c) for c in rgb], 1)
    b_.inputs['Roughness'].default_value = rough
    me.materials.append(m)
    return len(me.materials) - 1


lip_mi = flat_mat('lip_inner', lip_rgb * 0.45, 0.45)
cav_mi = flat_mat('mouth_cavity', lip_rgb * 0.12 + 0.005, 0.8)
d1 = 0.15 * mw  # depth of the inner lip band
loop_verts = upper + lower[1:-1][::-1]  # closed loop: right corner -> upper -> left corner -> lower (back)
side = {v: 1 for v in upper}
side.update({v: -1 for v in lower})
for v in corners:
    side[v] = 0
P0 = to_world(a_mid, 0.0, d_lip(a_mid) - 1.2 * d1)  # a point inside the head, behind the closed lips


def inside(target, frac=0.8):
    """pull a point that would stick out of the head back inside (ray from behind the lips)"""
    dv = target - P0
    L_ = dv.length
    hit = bvh_cut.ray_cast(P0, dv.normalized(), L_ * 1.3)
    if hit[0] is not None and hit[3] < L_ * 1.15:
        return P0 + dv.normalized() * (hit[3] * frac)
    return target


def copy_deform(src_v, dst_v):
    for k, val in src_v[dl].items():
        dst_v[dl][k] = val


ring1, ring2 = {}, {}
Rx, Ry_up, Ry_dn, d2 = 0.55 * mw, 0.30 * mw, 0.45 * mw, 0.35 * mw
d_back = d_lip(a_mid) - d1 - d2
for v in loop_verts:
    pw = MW @ v.co
    a, b, dd = frame(np.array(pw))
    # the band curls in (upper edge back and down, lower back and up), like the inside of real lips: an open mouth
    # shows rounded lips instead of a flat frame
    p1 = to_world(a_mid + (a - a_mid) * 0.95, b - side[v] * 0.04 * mw, dd - d1)
    n1 = bm.verts.new(MWi @ p1)
    copy_deform(v, n1)
    ring1[v] = n1
    xn = float(np.clip((a - a_mid) / hw, -1, 1))
    yy = math.sqrt(max(0.0, 1 - xn * xn)) * (Ry_up if side[v] > 0 else Ry_dn) * side[v]
    p2 = inside(to_world(a_mid + xn * Rx, yy, d_back))
    n2 = bm.verts.new(MWi @ p2)
    copy_deform(v, n2)
    ring2[v] = n2
capw = inside(to_world(a_mid, -0.05 * mw, d_back - 0.25 * mw))
cap = bm.verts.new(MWi @ capw)
cap[dl][hi_], cap[dl][ji_] = 0.5, 0.5
new_faces_lip, new_faces_cav = [], []
nl = len(loop_verts)
for k in range(nl):
    va, vb = loop_verts[k], loop_verts[(k + 1) % nl]
    e = bm.edges.get((va, vb))
    f = e.link_faces[0]
    lp = next(l for l in f.loops if l.vert is va)
    if lp.link_loop_next.vert is vb:  # f runs va -> vb, the new band must run vb -> va
        va, vb = vb, va
        lp = next(l for l in f.loops if l.vert is va)
    lpb = next(l for l in f.loops if l.vert is vb)
    uv_a, uv_b = lp[uvl].uv.copy(), lpb[uvl].uv.copy()
    # f runs vb -> va here; band quads: (va, vb, ring1 vb, ring1 va) keeps the orientation
    q1 = bm.faces.new((va, vb, ring1[vb], ring1[va]))
    q1.material_index = lip_mi
    for l, uv in zip(q1.loops, (uv_a, uv_b, uv_b, uv_a)):
        l[uvl].uv = uv
    q2 = bm.faces.new((ring1[va], ring1[vb], ring2[vb], ring2[va]))
    q3 = bm.faces.new((ring2[va], ring2[vb], cap))
    for q in (q2, q3):
        q.material_index = cav_mi
    new_faces_lip.append(q1)
    new_faces_cav += [q2, q3]
bm.to_mesh(me)
me.update()
bm.free()

# ---- 10. jaw bone ----
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='EDIT')
Wai = Wa.inverted()
eb = arm.data.edit_bones.new(JAW)
eb.head = Wai @ hinge
eb.tail = Wai @ chin
eb.parent = arm.data.edit_bones[HEAD]
eb.use_connect = False
rdir = (chin - hinge).normalized()
eb.align_roll((Wai.to_3x3() @ left.cross(rdir)).normalized())  # local X = character's left: +X rotation opens
bpy.ops.object.mode_set(mode='OBJECT')

# ---- 11. teeth + tongue (one skinned mesh, rigid on Head / Jaw) ----
def mat(name, rgb, rough):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b_ = m.node_tree.nodes['Principled BSDF']
    b_.inputs['Base Color'].default_value = (*rgb, 1)
    b_.inputs['Roughness'].default_value = rough
    return m


def superellipsoid(bmm, center, ax, size, ex=0.35, nu=10, nv_=7, mi=0, group=None):
    """rounded box: ax = 3 world axes (width, height, depth), size = full extents"""
    ring = []
    for i in range(nv_ + 1):
        phi = -math.pi / 2 + math.pi * i / nv_
        row = []
        for j in range(nu):
            th = 2 * math.pi * j / nu
            cp, sp, ct, st = math.cos(phi), math.sin(phi), math.cos(th), math.sin(th)
            sx = math.copysign(abs(cp) ** ex * abs(ct) ** ex, cp * ct)
            sy = math.copysign(abs(sp) ** ex, sp)
            sz = math.copysign(abs(cp) ** ex * abs(st) ** ex, cp * st)
            p = center + ax[0] * (sx * size[0] / 2) + ax[1] * (sy * size[1] / 2) + ax[2] * (sz * size[2] / 2)
            row.append(bmm.verts.new(MWi @ p))
        ring.append(row)
    for i in range(nv_):
        for j in range(nu):
            f = bmm.faces.new((ring[i][j], ring[i][(j + 1) % nu], ring[i + 1][(j + 1) % nu], ring[i + 1][j]))
            f.material_index = mi
            f.smooth = True
    bmesh.ops.remove_doubles(bmm, verts=[v for r_ in (ring[0], ring[-1]) for v in r_], dist=1e-9)
    bmesh.ops.recalc_face_normals(bmm, faces=[f for f in bmm.faces if f.material_index == mi])


mouth_me = bpy.data.meshes.new('mouth_parts')
mouth = bpy.data.objects.new('mouth_parts', mouth_me)
scene.collection.objects.link(mouth)
for m in (mat('teeth', (1, 1, 1), 0.35), mat('tongue', (1, 1, 1), 0.55)):
    # base colour = the baked colour attribute below (the glTF exporter writes it as COLOR_0)
    vc = m.node_tree.nodes.new('ShaderNodeVertexColor')
    vc.layer_name = 'Col'
    m.node_tree.links.new(vc.outputs['Color'], m.node_tree.nodes['Principled BSDF'].inputs['Base Color'])
    mouth_me.materials.append(m)
g_head = mouth.vertex_groups.new(name=HEAD)
g_jaw = mouth.vertex_groups.new(name=JAW)
bmm = bmesh.new()
dlm = bmm.verts.layers.deform.verify()
K_ARCH = 1.6 / mw  # dental arch: canines ~0.35 mw to the side sit ~0.2 mw behind the incisors
# (width, visible height, thickness) in mouth widths, from the midline outwards
UPPER = [(0.17, 0.21, 0.12), (0.13, 0.18, 0.11), (0.15, 0.20, 0.14), (0.14, 0.16, 0.17), (0.14, 0.15, 0.18),
         (0.20, 0.13, 0.20)]
LOWER = [(0.11, 0.18, 0.11), (0.12, 0.18, 0.11), (0.14, 0.19, 0.13), (0.14, 0.16, 0.16), (0.14, 0.15, 0.17),
         (0.21, 0.13, 0.20)]


def arch_row(spec, d_front, k, b_edge, updir, gi):
    """teeth along the parabola d = d_front - k a^2 (face frame), incisal edges at b_edge, crowns towards updir"""
    ts = np.linspace(0, 0.8 * mw, 400)
    ds = d_front - k * ts ** 2
    arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(ts), np.diff(ds)))])
    s = 0.0
    for wdt, hgt, thk in spec:
        wdt, hgt, thk = wdt * mw, hgt * mw, thk * mw
        sc = s + wdt / 2
        s += wdt
        tc = float(np.interp(sc, arc, ts))
        for sgn in (1, -1):
            ac = a_mid + sgn * tc
            tan = (Lv * sgn - Fv * 2 * k * tc)
            tan /= np.linalg.norm(tan)
            nrm_ = np.cross(tan, Uv) * sgn  # points forward (out of the arch)
            front = c0 + ac * Lv + (d_front - k * tc * tc) * Fv
            embed = 0.12 * mw
            b0 = b_edge + updir * (hgt + embed) / 2
            cen = Vector(front - nrm_ * thk / 2 + b0 * Uv)
            n_before = len(bmm.verts)
            superellipsoid(bmm, cen, (Vector(tan), up, Vector(nrm_)), (wdt * 0.99, hgt + embed, thk), mi=0,
                           nu=6 if LOWPOLY else 10, nv_=4 if LOWPOLY else 7)
            bmm.verts.ensure_lookup_table()
            for v in bmm.verts[n_before:]:
                v[dlm][gi] = 1.0


d_teeth = d_lip(a_mid) - d1 - 0.03 * mw
b_up_tip, b_lo_tip = b_lip(a_mid) - 0.05 * mw, b_lip(a_mid) - 0.03 * mw  # small overbite, lower row mostly hidden
arch_row(UPPER, d_teeth, K_ARCH, b_up_tip, 1, g_head.index)
arch_row(LOWER, d_teeth - 0.07 * mw, K_ARCH * 1.15, b_lo_tip, -1, g_jaw.index)
n_before = len(bmm.verts)
superellipsoid(bmm, to_world(a_mid, -0.14 * mw, d_teeth - 0.45 * mw), (left, up, fwd),
               (0.62 * mw, 0.16 * mw, 0.75 * mw), ex=0.7, nu=10 if LOWPOLY else 16, nv_=5 if LOWPOLY else 8, mi=1)
bmm.verts.ensure_lookup_table()
for v in bmm.verts[n_before:]:
    v[dlm][g_jaw.index] = 1.0
bmm.to_mesh(mouth_me)
bmm.free()
# baked occlusion: a real-time renderer does not darken the inside of the mouth, so flat-lit teeth read as a white
# bar. Darken with depth behind the lips (back teeth, tongue) and towards the gums; the lower row a little more.
nm = len(mouth_me.vertices)
cm = np.empty(nm * 3)
mouth_me.vertices.foreach_get('co', cm)
Am, Bm, Dm = frame(cm.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
is_tongue = np.zeros(nm, bool)
for poly in mouth_me.polygons:
    if poly.material_index == 1:
        is_tongue[list(poly.vertices)] = True
is_lower = np.array([any(g_.group == g_jaw.index for g_ in v.groups) for v in mouth_me.vertices]) & ~is_tongue
back = (d_lip(a_mid) - Dm) / mw
occl = np.clip(1.25 - 1.3 * back, 0.25, 1.0)
gum = np.where(is_lower, (b_lo_tip - Bm), (Bm - b_up_tip)) / (0.2 * mw)
occl *= np.where(is_tongue, 1.0, 1 - 0.5 * np.clip(gum, 0, 1)) * np.where(is_lower, 0.8, 1.0)
base = np.where(is_tongue[:, None], [0.45, 0.14, 0.14], [0.82, 0.78, 0.70])
col = mouth_me.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
col.data.foreach_set('color', np.hstack([base * occl[:, None], np.ones((nm, 1))]).ravel())
mouth.parent = arm
mouth.matrix_world = MW
mod = mouth.modifiers.new('Armature', 'ARMATURE')
mod.object = arm

# ---- 12. lip shape keys ----
co = np.empty(len(me.vertices) * 3)
me.vertices.foreach_get('co', co)
Xs = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
As, Bs, Ds = frame(Xs)
xs = (As - a_mid) / hw                  # -1 .. 1 between the mouth corners
ys = (Bs - b_lip(As)) / hw              # above (+) / below (-) the lip line
zs = (Ds - d_lip(As)) / hw              # in front (+) / behind (-) the lip surface
# upper (+1) / lower (-1) lip from the jaw share of the skin weights: the split seam vertices coincide in space
jf = np.zeros(len(me.vertices))
for v in me.vertices:
    tot = sum(gg.weight for gg in v.groups if gg.group in (hi_, ni_, ji_))
    jf[v.index] = sum(gg.weight for gg in v.groups if gg.group == ji_) / tot if tot > 0 else 0
sgn = 1 - 2 * jf
fall = np.exp(-(np.maximum(np.abs(xs) - 0.8, 0) / 0.55) ** 2 - (ys / 0.75) ** 2 - (np.minimum(zs + 0.3, 0) / 0.6) ** 2)
lips = np.exp(-(np.maximum(np.abs(xs) - 0.9, 0) / 0.3) ** 2 - (ys / 0.35) ** 2)  # the lips proper
xc = np.clip(xs, -1.2, 1.2)
opening = np.clip(1 - xs * xs, 0, 1) * lips
KEYS = {
    # corners out and slightly back, lips part a little (shows the teeth for EE / S / T)
    'wide': (0.22 * xc * fall, 0.07 * sgn * opening, -0.10 * xc * xc * fall),
    # corners in, lips forward (more in the middle), round opening
    'round': (-0.42 * xc * fall, 0.16 * sgn * opening, 0.45 * (1 - 0.5 * np.minimum(xs * xs, 1)) * fall),
}
body.shape_key_add(name='Basis', from_mix=False)
R3 = np.linalg.inv(M[:3, :3])
for name, (dx, dy, dz) in KEYS.items():
    disp = (dx[:, None] * Lv + dy[:, None] * Uv + dz[:, None] * Fv) * hw
    kb = body.shape_key_add(name=name, from_mix=False)
    kb.data.foreach_set('co', (co.reshape(-1, 3) + disp @ R3.T).ravel())

# ---- 13. export ----
for o in (cam,):
    bpy.data.objects.remove(o)
jb = arm.data.bones[JAW]
# full opening: 0.30 rad for realistic proportions (mouth width / hinge distance ~0.42); cartoon mouths that are wide
# for their jaw open further, else a 12 cm grin only ever shows a slit
jaw_max = float(np.clip(0.30 * math.sqrt(mw / (Vector(c0) - hinge).length / 0.42), 0.22, 0.42))
face = {
    'jaw_bone': JAW, 'head_bone': HEAD, 'jaw_open_axis': 'X', 'jaw_open_max_rad': jaw_max,
    'shape_keys': list(KEYS), 'mesh': body.name, 'mouth_mesh': mouth.name,
    'hinge_world': list(hinge), 'chin_world': list(chin), 'lip_center_world': list(c0),
    'frame': {'left': list(left), 'up': list(up), 'fwd': list(fwd)},
    'mouth_width_m': mw, 'report': report,
}
# no morph normals: their deltas are exported dense (8 MB of near-zeros on a 1024 mesh); the lip shapes are small
bpy.ops.export_scene.gltf(filepath=out, export_animations=False, export_morph=True, export_morph_normal=False,
                          export_try_sparse_sk=True)
json.dump(face, open(os.path.splitext(out)[0] + '.face.json', 'w'), indent=1, default=float)
print('WROTE', out, json.dumps(report, default=float))

if 'asset_result' in globals():
    asset_result({'output': out})
