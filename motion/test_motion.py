"""CPU regression checks: motion/kimenv/bin/python -P motion/test_motion.py."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock
import numpy as np


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


am, strikes = load('add_moves'), load('strikes')


def accessor(g, blob, index):
    a = g['accessors'][index]
    v = g['bufferViews'][a['bufferView']]
    width = {'SCALAR': 1, 'VEC3': 3, 'VEC4': 4}[a['type']]
    return np.frombuffer(blob, '<f4', count=a['count'] * width,
                         offset=v.get('byteOffset', 0) + a.get('byteOffset', 0)).reshape(a['count'], width)


def fixture():
    nodes = [{'name': 'Armature', 'children': [1]}, {'name': 'mixamorig:Hips', 'translation': [0, 1, 0]}]
    by = {'Hips': 1}
    def bone(name, parent, offset):
        by[name] = len(nodes)
        nodes[by[parent]].setdefault('children', []).append(len(nodes))
        nodes.append({'name': 'mixamorig:' + name, 'translation': offset})
    bone('Spine', 'Hips', [0, .15, 0])
    bone('Spine1', 'Spine', [0, .15, 0])
    bone('Spine2', 'Spine1', [0, .15, 0])
    bone('Neck', 'Spine2', [0, .1, 0])
    bone('Head', 'Neck', [0, .1, 0])
    for s, sign in [('Left', 1), ('Right', -1)]:
        bone(s + 'Shoulder', 'Spine2', [.1 * sign, 0, 0])
        bone(s + 'Arm', s + 'Shoulder', [.1 * sign, 0, 0])
        bone(s + 'ForeArm', s + 'Arm', [.3 * sign, 0, 0])
        bone(s + 'Hand', s + 'ForeArm', [.25 * sign, 0, 0])
        bone(s + 'UpLeg', 'Hips', [.1 * sign, -.1, 0])
        bone(s + 'Leg', s + 'UpLeg', [0, -.4, 0])
        bone(s + 'Foot', s + 'Leg', [0, -.4, 0])
        bone(s + 'ToeBase', s + 'Foot', [0, 0, .1])
    g = {'asset': {'version': '2.0'}, 'buffers': [{'byteLength': 4}], 'nodes': nodes,
         'skins': [{'joints': list(by.values())}], 'scenes': [{'nodes': [0]}], 'scene': 0}
    world = {0: np.eye(4)}
    for i, n in enumerate(nodes):
        for c in n.get('children', []):
            world[c] = world[i] @ am.node_matrix(nodes[c])
    rest = np.array([world[i][:3, 3] for i in by.values()])
    clip = {'names': [am.MAP[n] for n in by], 'rest': rest.tolist(), 'restQuat': [[0, 0, 0, 1]] * len(by),
            'quat': np.tile([0, 0, 0, 1], (4, len(by), 1)).tolist(),
            'pos': np.tile(rest, (4, 1, 1)).tolist(), 'fps': 30}
    return g, by, clip


class RetargetTests(unittest.TestCase):
    def transfer(self, g, clip):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            am.write_glb(d / 'rig.glb', g, bytearray(4))
            (d / 'manifest.json').write_text(json.dumps({'moves': [{'name': 'test', 'file': 'test.json'}]}))
            (d / 'test.json').write_text(json.dumps(clip))
            am.main(d / 'rig.glb', d / 'out.glb', d / 'root.json', d)
            result, blob = am.read_glb(d / 'out.glb')
            return result, blob, json.loads((d / 'root.json').read_text())

    def test_half_turns_and_random_rotations(self):
        rng = np.random.default_rng(4)
        for q in [np.array([1, -1, 0, 0]), *rng.normal(size=(200, 4))]:
            q = q / np.linalg.norm(q)
            np.testing.assert_allclose(am.qto_mat(am.qfrom_mat(am.qto_mat(q))), am.qto_mat(q), atol=1e-12)

    def test_helper_hierarchy_first_skin_and_matrix_hand(self):
        g, by, clip = fixture()
        hand, forearm = by['RightHand'], by['RightForeArm']
        helper = len(g['nodes'])
        g['nodes'].append({'children': [hand], 'name': 'helper'})
        g['nodes'][forearm]['children'] = [helper]
        g['skins'].insert(0, {'joints': [by['Head']]})
        m = am.node_matrix(g['nodes'][hand])
        g['nodes'][hand] = {'name': 'mixamorig:RightHand', 'matrix': m.T.ravel().tolist()}
        angle = np.pi / 3
        q = [0, 0, np.sin(angle / 2), np.cos(angle / 2)]
        for frame in clip['quat']:
            frame[clip['names'].index('RightHand')] = q
        out, blob, _ = self.transfer(g, clip)
        self.assertNotIn('matrix', out['nodes'][hand])
        tracks = {c['target']['node']: accessor(out, blob, out['animations'][0]['samplers'][c['sampler']]['output'])
                  for c in out['animations'][0]['channels'] if c['target']['path'] == 'rotation'}
        np.testing.assert_allclose(am.qto_mat(tracks[hand][0]), am.qto_mat(q), atol=1e-6)
        self.assertNotIn(helper, tracks)

    def test_root_path_starts_at_zero_and_old_clip_survives(self):
        g, _, clip = fixture()
        g['animations'] = [{'name': 'speak', 'samplers': [], 'channels': []}]
        pos = np.asarray(clip['pos'])
        pos[:, :, 0] += np.arange(4)[:, None] + 7
        clip['pos'] = pos.tolist()
        out, _, rm = self.transfer(g, clip)
        self.assertEqual(out['animations'][0]['name'], 'speak')
        self.assertEqual(rm['clips']['test']['pelvisXZ'], [[0, 0], [1, 0], [2, 0], [3, 0]])

    def test_rejects_bad_input_before_writing(self):
        g, _, clip = fixture()
        for scale in ([1, 2, 1], [-1, 1, 1]):
            broken = copy.deepcopy(g)
            broken['nodes'][0]['scale'] = scale
            with self.assertRaises(ValueError):
                self.transfer(broken, clip)
        for bad in (float('nan'), 0):
            broken = copy.deepcopy(clip)
            broken['fps'] = bad
            with self.assertRaises(ValueError):
                self.transfer(g, broken)

    def test_prop_and_unmapped_jaw_are_not_keyed(self):
        g, by, clip = fixture()
        for name, parent in [('Jaw', by['Head']), ('sword', by['RightHand'])]:
            i = len(g['nodes'])
            g['nodes'][parent].setdefault('children', []).append(i)
            g['nodes'].append({'name': name, 'scale': [1, 1, 1] if name == 'Jaw' else [.1, 2, .1]})
            if name == 'Jaw':
                g['skins'][0]['joints'].append(i)
        out, _, _ = self.transfer(g, clip)
        keyed = {c['target']['node'] for c in out['animations'][0]['channels']}
        self.assertTrue({len(g['nodes']) - 1, len(g['nodes']) - 2}.isdisjoint(keyed))


class StrikeTests(unittest.TestCase):
    def setUp(self):
        self.idx = {'LeftHand': 0, 'RightHand': 1}
        self.j = np.zeros((60, 2, 3))
        self.root = np.zeros((60, 3))
        self.mv = {'name': 'slash', 'strike': 'hand', 'strike_gate': {'side': 'right'}}

    def test_idle_translation_and_offhand_do_not_pass(self):
        self.root[:, 0] = np.arange(60) / 10
        self.j[:] = self.root[:, None]
        self.j[:, 0, 1] = np.sin(np.arange(60) / 3)
        metrics, _ = strikes.measure(self.j, self.root, self.idx, self.mv, 30)
        self.assertFalse(metrics['strike_ok'])
        self.assertEqual(metrics['strike_tip'], 'RightHand')

    def test_swing_passes_and_timing_uses_authored_hand(self):
        self.j[:, 1, 1] = .5 * np.sin(np.arange(60) / 5)
        self.j[:, 0, 1] = 2 * np.sin(np.arange(60) / 3)
        metrics, _ = strikes.measure(self.j, self.root, self.idx, self.mv, 30)
        self.assertTrue(metrics['strike_ok'])
        data = strikes.frame_data(self.j, self.root, self.idx, self.mv, 30)
        self.assertEqual(data['strike_tip'], 'RightHand')
        self.assertTrue(data['active'][0] <= data['contact'] <= data['active'][1])
        self.assertTrue(data['contact_estimate'])

    def test_single_frame_spike_does_not_pass(self):
        self.j[30, 1, 1] = .5
        metrics, _ = strikes.measure(self.j, self.root, self.idx, self.mv, 30)
        self.assertFalse(metrics['strike_ok'])

    def test_invalid_gate_rejected(self):
        for gate in ({'side': 'rigth'}, {'min_speed': float('nan')}, {'min_excursion': -1}, {'speed': 3}):
            with self.assertRaises(ValueError):
                strikes.validate({**self.mv, 'strike_gate': gate})


class GenerationTests(unittest.TestCase):
    def test_all_rejected_clears_previous_baked_set(self):
        gen = load('gen_moves')
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            spec = d / 'spec.json'
            spec.write_text(json.dumps({'fps': 30, 'moves': [{'name': 'slash', 'prompt': 'A person strikes.', 'duration': 2}]}))
            (d / 'manifest.json').write_text(json.dumps({'moves': [{'name': 'slash', 'file': 'slash.json'}]}))
            (d / 'slash.json').write_text('{}')

            def reject(module, *args):
                if args[0] == 'gen':
                    report = d / 'gen/moves/slash.json'
                    report.write_text(json.dumps({'accepted': False, 'all_gates': [{'strike_ok': False}]}))
                    later = time.time() + 1  # coarse inode clocks may lag gen_moves' t_call
                    os.utime(report, (later, later))
                    return 1
                return 0

            with mock.patch.object(gen, 'check'), mock.patch.object(gen, 'encoder_up', return_value=True), mock.patch.object(gen, 'run', side_effect=reject):
                result = gen.main(str(spec), str(d))
            self.assertEqual(result['rejected'], ['slash'])
            self.assertEqual(json.loads((d / 'manifest.json').read_text())['moves'], [])
            self.assertFalse((d / 'slash.json').exists())


if __name__ == '__main__':
    unittest.main()
