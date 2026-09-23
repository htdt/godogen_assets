"""
What is inside a GLB (plain Python, no dependencies): meshes / primitives / morph targets / attributes, skins, bones
matching a pattern, animations with the nodes and paths they drive, images, and where the bytes go.

    python3 tools/glb_info.py file.glb [--bones Jaw] [--channels]

Used to check lipsync output (lipsync/mouth_rig.py, lipsync/lipsync.py): the jaw bone is there, `speak` has exactly 2 channels
(mixamorig:Jaw rotation + mesh weights), morph targets are sparse, teeth carry COLOR_0, no dense morph normals.
"""
import collections
import json
import struct
import sys

argv = sys.argv[1:]
path = argv[0]
bones = argv[argv.index('--bones') + 1] if '--bones' in argv else 'Jaw'
data = open(path, 'rb').read()
n = struct.unpack('<I', data[12:16])[0]
g = json.loads(data[20:20 + n])
acc, bv, nodes = g['accessors'], g['bufferViews'], g['nodes']
use = collections.Counter()


def add(ai, tag):
    a = acc[ai]
    if 'bufferView' in a:
        use[tag] += bv[a['bufferView']]['byteLength']
    if 'sparse' in a:
        sp = a['sparse']
        use[tag + ' (sparse)'] += bv[sp['indices']['bufferView']]['byteLength'] + bv[sp['values']['bufferView']]['byteLength']


print(f'{path}: {len(data) / 1e6:.1f} MB')
for m in g['meshes']:
    names = m.get('extras', {}).get('targetNames')
    tris = sum(acc[p['indices']]['count'] // 3 for p in m['primitives'] if 'indices' in p)
    print(f"mesh {m['name']!r}: {len(m['primitives'])} primitives, {tris} triangles, morph targets {names}")
    for p in m['primitives']:
        mat = g['materials'][p['material']]['name'] if 'material' in p else None
        print(f"  material {mat!r}: {acc[p['attributes']['POSITION']]['count']} verts, {sorted(p['attributes'])}")
        for k, v in p['attributes'].items():
            add(v, k)
        if 'indices' in p:
            add(p['indices'], 'indices')
        for t in p.get('targets', []):
            for k, v in t.items():
                add(v, 'morph ' + k)
for s in g.get('skins', []):
    print(f"skin: {len(s['joints'])} joints; matching {bones!r}: "
          f"{[nodes[j]['name'] for j in s['joints'] if bones in nodes[j].get('name', '')]}")
for a in g.get('animations', []):
    ch = collections.Counter(nodes[c['target']['node']]['name'] for c in a['channels'])
    paths = sorted({c['target']['path'] for c in a['channels']})
    detail = dict(ch) if '--channels' in argv or len(ch) <= 4 else f'{len(ch)} nodes'
    print(f"animation {a['name']!r}: {len(a['channels'])} channels {paths}: {detail}")
for i in g.get('images', []):
    use['image ' + i.get('name', '?')] += bv[i['bufferView']]['byteLength']
print('bytes:', {k: f'{v / 1e6:.2f} MB' for k, v in sorted(use.items(), key=lambda kv: -kv[1])})
