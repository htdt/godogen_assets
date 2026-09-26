"""
add_moves.py: put baked Kimodo move sets onto a Mixamo-named rig as ordinary glTF animations. Run by bin/add-moves.

    python add_moves.py RIG.glb OUT.glb ROOTMOTION.json BAKED_DIR [BAKED_DIR ...]

Kimodo's SOMA skeleton shares the Mixamo layout and the T-pose zero pose, so the transfer is direct: every mapped
bone takes its source joint's world rotation from rest, applied to the bone's own bind. Two rest corrections, both
static: SOMA's T-pose hand bends ~18 deg off the forearm while rig hands bind straight (the baked clips' restQuat
already straightens it), and thighs, shins, upper arms and forearms are aimed along the source T-pose once, so a
bind that is not an exact T-pose (legs sloping, arms a little low) does not skew every pose. Feet and hands keep their
bind as neutral: flat and straight on both skeletons. The root is scaled by the leg-length ratio.

Clips are in place (hips X/Z at bind, height kept); the horizontal hips path goes to ROOTMOTION.json for the game
to move the entity. Only mapped bones are keyed (a mouth rig's jaw stays free for speech); other animations in the
input are kept, one with a move's name is replaced; across move sets, a later set wins a name clash. Needs numpy only.
"""
import os
import re
import sys
import json
import struct
import shutil
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from cli_args import ArgumentParser

# Mixamo bone (prefix stripped) -> SOMA joint. Fingers: SOMA's first finger joint is the metacarpal.
MAP = {'Hips': 'Hips', 'Spine': 'Spine1', 'Spine1': 'Spine2', 'Spine2': 'Chest', 'Neck': 'Neck1', 'Head': 'Head'}
for _s in ('Left', 'Right'):
    MAP.update({f'{_s}Shoulder': f'{_s}Shoulder', f'{_s}Arm': f'{_s}Arm', f'{_s}ForeArm': f'{_s}ForeArm',
                f'{_s}Hand': f'{_s}Hand', f'{_s}UpLeg': f'{_s}Leg', f'{_s}Leg': f'{_s}Shin', f'{_s}Foot': f'{_s}Foot',
                f'{_s}ToeBase': f'{_s}ToeBase'})
    MAP.update({f'{_s}HandThumb{i}': f'{_s}HandThumb{i}' for i in (1, 2, 3)})
    MAP.update({f'{_s}Hand{f}{i}': f'{_s}Hand{f}{i + 1}'
                for f in ('Index', 'Middle', 'Ring', 'Pinky') for i in (1, 2, 3)})
REQUIRED = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head'] + [
    f'{s}{b}' for s in ('Left', 'Right') for b in ('Arm', 'ForeArm', 'Hand', 'UpLeg', 'Leg', 'Foot')]
ALIGN = {'UpLeg': 'Leg', 'Leg': 'Foot', 'Arm': 'ForeArm', 'ForeArm': 'Hand'}  # bone -> child that sets its direction
PLANTED_SPEED = 0.1  # m/s: a heel or toe slower than this carries weight (planted feet move < 0.05)
MIN_RUN = 3          # frames: shorter planted or lifted runs are flicker


def planted(pos, idx, fps, loop):
    """Per frame [left, right], 1 = the foot is planted (heel or toe at rest, on the floor or on anything else)."""
    flags = []
    for side in ('Left', 'Right'):
        p = pos[:, [idx[side + 'Foot'], idx[side + 'ToeBase']]]
        slow = (np.linalg.norm(np.diff(p, axis=0), axis=-1) * fps < PLANTED_SPEED).any(axis=1)
        f = np.append(slow, slow[0] if loop else slow[-1]).astype(int)       # a loop's last frame is its first
        for value in (0, 1):  # fill short gaps, then drop short contacts
            runs = np.flatnonzero(np.diff(np.r_[-1, f, -1]) != 0)
            for a, b in zip(runs[:-1], runs[1:]):
                if f[a] == value and b - a < MIN_RUN and 0 < a and b < len(f):
                    f[a:b] = 1 - value
        flags.append(f)
    return np.stack(flags, 1).tolist()


def strip(name):
    return re.sub(r'^mixamorig\d*[:_]?', '', name)


# ---- quaternions, xyzw like glTF, broadcasting over leading axes
def qmul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def qinv(q):
    return q * np.array([-1, -1, -1, 1.0])


def qrot(q, v):
    return qmul(qmul(q, np.concatenate([v, np.zeros(v.shape[:-1] + (1,))], -1)), qinv(q))[..., :3]


def qfrom_mat(m):
    # Choose the largest component: antisymmetric signs alone lose the axis signs at exactly 180 degrees.
    q = np.empty(4)
    if np.trace(m) > 0:
        s = 2 * np.sqrt(1 + np.trace(m))
        q[:] = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
        q[i], q[j], q[k], q[3] = s / 4, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s, (m[k, j] - m[j, k]) / s
    return q / np.linalg.norm(q)


def qto_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def qbetween(a, b):
    if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-8:
        raise ValueError('cannot align a zero-length bone')
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    q = np.array([*np.cross(a, b), 1 + a @ b])
    if q[3] < 1e-8:  # opposite: any perpendicular axis
        axis = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        q = np.array([*axis, 0.0])
    return q / np.linalg.norm(q)


# ---- GLB
def read_glb(path):
    data = Path(path).read_bytes()
    if len(data) < 20:
        raise ValueError(f'{path}: truncated GLB')
    magic, version, length = struct.unpack_from('<III', data)
    if magic != 0x46546C67 or version != 2 or length != len(data):
        raise ValueError(f'{path}: not a complete glTF 2 GLB')
    off, gltf, blob = 12, None, bytearray()
    while off < length:
        if off + 8 > length:
            raise ValueError(f'{path}: truncated GLB chunk')
        size, kind = struct.unpack_from('<II', data, off)
        if size % 4 or off + 8 + size > length:
            raise ValueError(f'{path}: invalid GLB chunk length')
        chunk = data[off + 8:off + 8 + size]
        off += 8 + size
        if kind == 0x4E4F534A:
            gltf = json.loads(chunk)
        elif kind == 0x004E4942:
            blob = bytearray(chunk)
    if gltf is None or len(gltf.get('buffers', [])) != 1 or 'uri' in gltf['buffers'][0]:
        raise ValueError(f'{path}: expected one embedded buffer')
    if gltf['buffers'][0]['byteLength'] > len(blob):
        raise ValueError(f'{path}: truncated embedded buffer')
    return gltf, blob


def write_glb(path, gltf, blob):
    blob += b'\0' * (-len(blob) % 4)
    gltf['buffers'][0]['byteLength'] = len(blob)
    js = json.dumps(gltf, separators=(',', ':'), allow_nan=False).encode()
    js += b' ' * (-len(js) % 4)
    # Also safe when OUT is the input: a failed write must not truncate the character.
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix='.glb')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(struct.pack('<III', 0x46546C67, 2, 28 + len(js) + len(blob)))
            f.write(struct.pack('<II', len(js), 0x4E4F534A) + js + struct.pack('<II', len(blob), 0x004E4942) + blob)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)  # mkstemp creates private files
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def add_accessor(gltf, blob, arr, kind):
    blob += b'\0' * (-len(blob) % 4)
    raw = np.ascontiguousarray(arr, '<f4').tobytes()
    gltf.setdefault('bufferViews', []).append({'buffer': 0, 'byteOffset': len(blob), 'byteLength': len(raw)})
    blob += raw
    acc = {'bufferView': len(gltf['bufferViews']) - 1, 'componentType': 5126, 'count': len(arr), 'type': kind}
    if kind == 'SCALAR':
        acc.update(min=[float(arr.min())], max=[float(arr.max())])
    gltf.setdefault('accessors', []).append(acc)
    return len(gltf['accessors']) - 1


def node_matrix(n):
    if 'matrix' in n:
        return np.array(n['matrix'], float).reshape(4, 4).T
    m = np.eye(4)
    m[:3, :3] = qto_mat(np.array(n.get('rotation', [0, 0, 0, 1.0]))) * np.array(n.get('scale', [1, 1, 1.0]))
    m[:3, 3] = n.get('translation', [0, 0, 0])
    return m


def rotation(m):
    scale = np.linalg.norm(m[:3, :3], axis=0)
    if not np.isfinite(m).all() or min(scale) < 1e-8 or not np.allclose(scale, scale[0], rtol=1e-4):
        raise ValueError('skeleton transforms need finite, positive uniform scale; apply transforms before rigging')
    r = m[:3, :3] / scale
    if np.linalg.det(r) < 0 or not np.allclose(r.T @ r, np.eye(3), atol=1e-4):
        raise ValueError('mirrored or sheared skeleton; apply transforms before rigging')
    return qfrom_mat(r)


def main(rig_path, out_path, rm_path, *baked_dirs):
    gltf, blob = read_glb(rig_path)
    nodes = gltf['nodes']
    parent = {c: i for i, n in enumerate(nodes) for c in n.get('children', [])}
    if len(parent) != sum(len(n.get('children', [])) for n in nodes):
        raise ValueError('node has more than one parent')
    for i in range(len(nodes)):
        seen = set()
        while i in parent:
            if i in seen:
                raise ValueError('node hierarchy contains a cycle')
            seen.add(i)
            i = parent[i]
    world = {}

    def wm(i):
        if i not in world:
            world[i] = (wm(parent[i]) if i in parent else np.eye(4)) @ node_matrix(nodes[i])
        return world[i]

    if not gltf.get('skins'):
        raise ValueError(f'{rig_path}: no skinned skeleton (rig it with mia-rig first)')
    # A mouth/prop skin may precede the body skin. Shared joints count once; two humanoids are ambiguous.
    joints = set(j for skin in gltf['skins'] for j in skin['joints'])
    by = {}
    for j in sorted(joints):
        name = strip(nodes[j].get('name', ''))
        if name in MAP and name in by:
            raise ValueError(f'ambiguous bone {name}: expected one humanoid skeleton')
        by[name] = j
    missing = [b for b in REQUIRED if b not in by]
    if missing:
        raise ValueError(f'{rig_path}: not a Mixamo-named humanoid rig, missing ' + ', '.join(missing))
    hips = by['Hips']
    order, stack = [], [hips]  # include helper nodes between joints, parents first
    while stack:
        i = stack.pop()
        if i in order:
            raise ValueError('skeleton hierarchy is not a tree')
        order.append(i)
        stack += nodes[i].get('children', [])
    if any(by[b] not in order for b in REQUIRED):
        raise ValueError('all humanoid bones must descend from Hips')
    # Only ancestors of joints participate in retargeting; attached prop scales are unrestricted.
    needed = set()
    for j in joints.intersection(order):
        while j != hips:
            needed.add(j)
            j = parent[j]
    order = [i for i in order if i == hips or i in needed]
    bind_w = {i: rotation(wm(i)) for i in order}
    pos_w = {i: wm(i)[:3, 3] for i in order}
    hips_parent_w = rotation(wm(parent[hips])) if hips in parent else np.array([0, 0, 0, 1.0])
    hips_parent_inv = np.linalg.inv(wm(parent[hips])) if hips in parent else np.eye(4)
    keyed = [i for i in order if i in joints and strip(nodes[i].get('name', '')) in MAP]
    for i in keyed:
        if 'matrix' in nodes[i]:  # glTF forbids matrix on a node targeted by animation
            m = node_matrix(nodes[i])
            nodes[i].update(translation=m[:3, 3].tolist(), rotation=rotation(m).tolist(),
                            scale=np.linalg.norm(m[:3, :3], axis=0).tolist())
            del nodes[i]['matrix']
    char_leg = pos_w[hips][1] - (pos_w[by['LeftFoot']][1] + pos_w[by['RightFoot']][1]) / 2
    if char_leg <= 1e-6:
        raise ValueError('rig must be upright in glTF Y-up with hips above feet')

    moves = {}  # name -> (baked dir, manifest entry)
    for baked in baked_dirs:
        manifest = json.loads(Path(baked, 'manifest.json').read_text())
        moves.update((m['name'], (baked, m)) for m in (manifest['moves'] if isinstance(manifest, dict) else manifest))
    if not moves:
        raise ValueError('no accepted moves in the baked move sets')
    anims = [a for a in gltf.get('animations', []) if a.get('name') not in moves]
    rootmotion = {'scaleRoot': None, 'clips': {}}

    for baked, mv in moves.values():
        clip = json.loads(Path(baked, mv.get('file', mv['name'] + '.json')).read_text())
        idx = {n: k for k, n in enumerate(clip['names'])}
        missing = sorted({MAP[strip(nodes[i]['name'])] for i in keyed if strip(nodes[i]['name']) in REQUIRED}
                         .union({'LeftToeBase', 'RightToeBase'}) - idx.keys())
        if missing:
            raise ValueError(f'{mv["name"]}: source clip is missing joints: {", ".join(missing)}')
        # finger joints hang off the straightened hand frame: they take the hand's rest
        rest_q = np.array([clip['restQuat'][idx[m.group(1) + 'Hand'] if (m := re.match(r'(Left|Right)Hand.', n)) else k]
                           for k, n in enumerate(clip['names'])], float)
        quat = np.array(clip['quat'], float)             # (N, J, 4) world rotations
        pos = np.array(clip['pos'], float)               # (N, J, 3)
        rest = np.array(clip['rest'], float)
        n_frames, fps = len(quat), clip.get('fps', 30)
        nj = len(idx)
        if (nj != len(clip['names']) or n_frames < 2 or not np.isfinite(fps) or fps <= 0
                or quat.shape != (n_frames, nj, 4) or pos.shape != (n_frames, nj, 3)
                or rest.shape != (nj, 3) or rest_q.shape != (nj, 4)
                or not all(np.isfinite(a).all() for a in (quat, pos, rest, rest_q))):
            raise ValueError(f'{mv["name"]}: invalid clip shapes, frame rate or non-finite data')
        for q in (quat, rest_q):
            norm = np.linalg.norm(q, axis=-1, keepdims=True)
            if np.any(norm < 1e-8):
                raise ValueError(f'{mv["name"]}: zero quaternion')
            q /= norm
        yaw = qinv(rest_q[idx['Hips']])                  # source rest heading -> the rig's (+Z)
        delta = qmul(qmul(yaw, qmul(quat, qinv(rest_q))), qinv(yaw))
        src_leg = rest[idx['Hips'], 1] - (rest[idx['LeftFoot'], 1] + rest[idx['RightFoot'], 1]) / 2
        if src_leg <= 1e-6:
            raise ValueError(f'{mv["name"]}: source hips must be above feet')
        scale = char_leg / src_leg
        rootmotion['scaleRoot'] = round(float(scale), 4)

        world_q, local_q = {}, {}
        for i in order:
            name = strip(nodes[i].get('name', ''))
            pw = world_q[parent[i]] if i != hips else np.broadcast_to(hips_parent_w, (n_frames, 4))
            if i in keyed and MAP[name] in idx:
                bind = bind_w[i]
                side = next((s for s in ('Left', 'Right') if name.startswith(s)), '')
                part = name[len(side):]
                child = by.get(f'{side}{ALIGN[part]}') if side and part in ALIGN else None
                if child is not None and MAP[f'{side}{ALIGN[part]}'] in idx:
                    src_dir = qrot(yaw, rest[idx[MAP[f'{side}{ALIGN[part]}']]] - rest[idx[MAP[name]]])
                    bind = qmul(qbetween(pos_w[child] - pos_w[i], src_dir), bind)
                world_q[i] = qmul(delta[:, idx[MAP[name]]], bind)
                local_q[i] = qmul(qinv(pw), world_q[i])
            else:
                world_q[i] = qmul(pw, rotation(node_matrix(nodes[i])))
        # in place: hips keep their bind X/Z, the height follows the source
        rise = (pos[:, idx['Hips'], 1] - rest[idx['Hips'], 1]) * scale
        hips_w = np.tile(pos_w[hips], (n_frames, 1))
        hips_w[:, 1] += rise
        hips_local = (hips_parent_inv @ np.c_[hips_w, np.ones(n_frames)].T).T[:, :3]
        path = qrot(yaw, pos[:, idx['Hips']] - pos[0, idx['Hips']]) * scale

        for q in local_q.values():  # neighbouring keys on the same hemisphere, for LINEAR slerp
            for f in range(1, n_frames):
                if q[f] @ q[f - 1] < 0:
                    q[f] = -q[f]
        tracks = [(i, 'rotation', 'VEC4', q) for i, q in local_q.items()] + [(hips, 'translation', 'VEC3', hips_local)]
        times = add_accessor(gltf, blob, np.arange(n_frames) / fps, 'SCALAR')
        samplers, channels = [], []
        for i, prop, kind, data in tracks:
            samplers.append({'input': times, 'output': add_accessor(gltf, blob, data, kind), 'interpolation': 'LINEAR'})
            channels.append({'sampler': len(samplers) - 1, 'target': {'node': i, 'path': prop}})
        anims.append({'name': mv['name'], 'samplers': samplers, 'channels': channels})
        loop = bool(mv.get('loop', False))
        rootmotion['clips'][mv['name']] = {
            'fps': fps, 'numFrames': n_frames, 'loop': loop,
            'scaleRoot': round(float(scale), 4),
            'frameData': mv.get('frame_data'),
            'pelvisXZ': np.round(path[:, [0, 2]], 4).tolist(),
            'hipY': np.round(hips_w[:, 1], 4).tolist(),
            'footContact': planted(pos, idx, fps, loop),
        }
        print(f'{mv["name"]}: {n_frames} frames @ {fps} fps', file=sys.stderr)

    gltf['animations'] = anims
    write_glb(out_path, gltf, blob)
    with open(rm_path, 'w') as f:
        json.dump(rootmotion, f)
    print(f'{len(moves)} moves, {len(keyed)} bones keyed, scaleRoot {rootmotion["scaleRoot"]}', file=sys.stderr)


def cli():
    ap = ArgumentParser(description='Transfer baked Kimodo moves onto one Mixamo character.')
    ap.add_argument('rig')
    ap.add_argument('-o', '--out')
    ap.add_argument('--baked', action='append')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rig = os.path.realpath(a.rig)  # outputs and the face JSON live next to the real file
        name = os.path.splitext(rig)[0].removesuffix('_rigged').removesuffix('_mouth')
        out = os.path.abspath(a.out or name + '_moves.glb')
        rm = os.path.splitext(out)[0].removesuffix('_moves') + '_rootmotion.json'
        baked = [os.path.abspath(d if os.path.exists(d) else os.path.join(root, d))
                 for d in (a.baked or ['motion/basic'])]
        for d in baked:
            if not os.path.isfile(os.path.join(d, 'manifest.json')):
                raise ValueError(f'no baked move set at {d} (gen-moves, motion/README.md)')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        main(rig, out, rm, *baked)
        face, dst_face = os.path.splitext(rig)[0] + '.face.json', os.path.splitext(out)[0] + '.face.json'
        if os.path.isfile(face) and os.path.realpath(face) != os.path.realpath(dst_face):
            shutil.copyfile(face, dst_face)
        print(json.dumps({'output': out, 'rootmotion': rm, 'baked': baked}) if a.json else out)
        return 0
    except (ValueError, KeyError, OSError, TypeError, IndexError) as e:
        msg = f'{type(e).__name__}: {e}'
        print(f'add-moves: {msg}', file=sys.stderr)
        if a.json:
            print(json.dumps({'error': msg}))
        return 1


if __name__ == '__main__':
    sys.exit(cli())
