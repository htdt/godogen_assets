"""
add_moves.py: put a baked Kimodo move set onto a Mixamo-named rig as ordinary glTF animations. Run by bin/add-moves.

    python add_moves.py RIG.glb BAKED_DIR OUT.glb ROOTMOTION.json

Kimodo's SOMA skeleton shares the Mixamo layout and the T-pose zero pose, so the transfer is direct: every mapped
bone takes its source joint's world rotation from rest, applied to the bone's own bind. Two rest corrections, both
static: SOMA's T-pose hand bends ~18 deg off the forearm while rig hands bind straight (the baked clips' restQuat
already straightens it), and thighs, shins, upper arms and forearms are aimed along the source T-pose once, so a
bind that is not an exact T-pose (legs sloping, arms a little low) does not skew every pose. Feet and hands keep their
bind as neutral: flat and straight on both skeletons. The root is scaled by the leg-length ratio.

Clips are in place (hips X/Z at bind, height kept); the horizontal hips path goes to ROOTMOTION.json for the game
to move the entity. Only mapped bones are keyed (a mouth rig's jaw stays free for speech); other animations in the
input are kept, one with a move's name is replaced. Needs numpy only.
"""
import os
import re
import sys
import json
import struct
import numpy as np

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
    w = np.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def qto_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def qbetween(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    q = np.array([*np.cross(a, b), 1 + a @ b])
    if q[3] < 1e-8:  # opposite: any perpendicular axis
        axis = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        q = np.array([*axis, 0.0])
    return q / np.linalg.norm(q)


# ---- GLB
def read_glb(path):
    data = open(path, 'rb').read()
    magic, _, length = struct.unpack_from('<III', data)
    if magic != 0x46546C67:
        raise ValueError(f'{path}: not a GLB')
    off, gltf, blob = 12, None, bytearray()
    while off < length:
        size, kind = struct.unpack_from('<II', data, off)
        chunk = data[off + 8:off + 8 + size]
        off += 8 + size
        if kind == 0x4E4F534A:
            gltf = json.loads(chunk)
        elif kind == 0x004E4942:
            blob = bytearray(chunk)
    if gltf is None or len(gltf.get('buffers', [])) != 1 or 'uri' in gltf['buffers'][0]:
        raise ValueError(f'{path}: expected one embedded buffer')
    return gltf, blob


def write_glb(path, gltf, blob):
    blob += b'\0' * (-len(blob) % 4)
    gltf['buffers'][0]['byteLength'] = len(blob)
    js = json.dumps(gltf, separators=(',', ':')).encode()
    js += b' ' * (-len(js) % 4)
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, 28 + len(js) + len(blob)))
        f.write(struct.pack('<II', len(js), 0x4E4F534A) + js + struct.pack('<II', len(blob), 0x004E4942) + blob)


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


def main(rig_path, baked, out_path, rm_path):
    gltf, blob = read_glb(rig_path)
    nodes = gltf['nodes']
    parent = {c: i for i, n in enumerate(nodes) for c in n.get('children', [])}
    world = {}

    def wm(i):
        if i not in world:
            world[i] = (wm(parent[i]) if i in parent else np.eye(4)) @ node_matrix(nodes[i])
        return world[i]

    if not gltf.get('skins'):
        raise ValueError(f'{rig_path}: no skinned skeleton (rig it with mia-rig first)')
    joints = set(gltf['skins'][0]['joints'])
    by = {strip(nodes[j].get('name', '')): j for j in joints}
    missing = [b for b in REQUIRED if b not in by]
    if missing:
        raise ValueError(f'{rig_path}: not a Mixamo-named humanoid rig, missing ' + ', '.join(missing))
    hips = by['Hips']
    order, stack = [], [hips]  # joints under the hips, parents first
    while stack:
        i = stack.pop()
        order.append(i)
        stack += [c for c in nodes[i].get('children', []) if c in joints]
    rot = lambda m: qfrom_mat(m[:3, :3] / np.linalg.norm(m[:3, :3], axis=0))  # noqa: E731 (uniform scale)
    bind_w = {i: rot(wm(i)) for i in order}
    pos_w = {i: wm(i)[:3, 3] for i in order}
    hips_parent_w = rot(wm(parent[hips])) if hips in parent else np.array([0, 0, 0, 1.0])
    hips_parent_inv = np.linalg.inv(wm(parent[hips])) if hips in parent else np.eye(4)
    keyed = [i for i in order if strip(nodes[i]['name']) in MAP]
    char_leg = pos_w[hips][1] - (pos_w[by['LeftFoot']][1] + pos_w[by['RightFoot']][1]) / 2

    manifest = json.load(open(os.path.join(baked, 'manifest.json')))
    moves = manifest['moves'] if isinstance(manifest, dict) else manifest
    names = {m['name'] for m in moves}
    anims = [a for a in gltf.get('animations', []) if a.get('name') not in names]
    rootmotion = {'scaleRoot': None, 'clips': {}}

    for mv in moves:
        clip = json.load(open(os.path.join(baked, mv.get('file', mv['name'] + '.json'))))
        idx = {n: k for k, n in enumerate(clip['names'])}
        # finger joints hang off the straightened hand frame: they take the hand's rest
        rest_q = np.array([clip['restQuat'][idx[m.group(1) + 'Hand'] if (m := re.match(r'(Left|Right)Hand.', n)) else k]
                           for k, n in enumerate(clip['names'])], float)
        quat = np.array(clip['quat'], float)             # (N, J, 4) world rotations
        pos = np.array(clip['pos'], float)               # (N, J, 3)
        rest = np.array(clip['rest'], float)
        n_frames, fps = len(quat), clip.get('fps', 30)
        yaw = qinv(rest_q[idx['Hips']])                  # source rest heading -> the rig's (+Z)
        delta = qmul(qmul(yaw, qmul(quat, qinv(rest_q))), qinv(yaw))
        src_leg = rest[idx['Hips'], 1] - (rest[idx['LeftFoot'], 1] + rest[idx['RightFoot'], 1]) / 2
        scale = char_leg / src_leg
        rootmotion['scaleRoot'] = round(float(scale), 4)

        world_q, local_q = {}, {}
        for i in order:
            name = strip(nodes[i]['name'])
            pw = world_q[parent[i]] if i != hips else np.broadcast_to(hips_parent_w, (n_frames, 4))
            if i in keyed and MAP[name] in idx:
                bind = bind_w[i]
                side = next((s for s in ('Left', 'Right') if name.startswith(s)), '')
                part = name[len(side):]
                child = by.get(f'{side}{ALIGN[part]}') if side and part in ALIGN else None
                if child is not None:  # aim the bone along the source T-pose
                    src_dir = qrot(yaw, rest[idx[MAP[f'{side}{ALIGN[part]}']]] - rest[idx[MAP[name]]])
                    bind = qmul(qbetween(pos_w[child] - pos_w[i], src_dir), bind)
                world_q[i] = qmul(delta[:, idx[MAP[name]]], bind)
                local_q[i] = qmul(qinv(pw), world_q[i])
            else:
                world_q[i] = qmul(pw, np.array(nodes[i].get('rotation', [0, 0, 0, 1.0])))
        # in place: hips keep their bind X/Z, the height follows the source
        rise = (pos[:, idx['Hips'], 1] - rest[idx['Hips'], 1]) * scale
        hips_w = np.tile(pos_w[hips], (n_frames, 1))
        hips_w[:, 1] += rise
        hips_local = (hips_parent_inv @ np.c_[hips_w, np.ones(n_frames)].T).T[:, :3]
        path = qrot(yaw, pos[:, idx['Hips']] - rest[idx['Hips']]) * scale

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
        rootmotion['clips'][mv['name']] = {
            'fps': fps, 'numFrames': n_frames, 'loop': bool(mv.get('loop', False)),
            'frameData': mv.get('frame_data'),
            'pelvisXZ': np.round(path[:, [0, 2]], 4).tolist(),
            'hipY': np.round(hips_w[:, 1], 4).tolist(),
        }
        print(f'{mv["name"]}: {n_frames} frames @ {fps} fps', file=sys.stderr)

    gltf['animations'] = anims
    write_glb(out_path, gltf, blob)
    with open(rm_path, 'w') as f:
        json.dump(rootmotion, f)
    print(f'{len(moves)} moves, {len(keyed)} bones keyed, scaleRoot {rootmotion["scaleRoot"]}', file=sys.stderr)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.exit('usage: python add_moves.py RIG.glb BAKED_DIR OUT.glb ROOTMOTION.json')
    try:
        main(*sys.argv[1:])
    except (ValueError, KeyError, OSError) as e:
        sys.exit(f'add-moves: {type(e).__name__}: {e}')
