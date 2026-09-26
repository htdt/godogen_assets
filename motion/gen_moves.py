"""
gen_moves.py: generate a move set from a spec with Kimodo and bake it for add-moves. Run by bin/gen-moves.

    python gen_moves.py SPEC.json OUT_DIR [--json]

Generation is kimodo-practical's kimogen.py (best-of-8 per move, numeric gates), baking its bake_kimodo.py; both run
in-process with their work folder redirected to OUT_DIR/gen (NPZ + gate report per move, stance_pose.json,
state.json, encoder.log). OUT_DIR gets manifest.json and one clip JSON per move.

Incremental: a move whose spec entry is unchanged since its last accepted generation is reused (state.json holds a
hash per move), so adding or rewording one move regenerates only that one. A move with no passing sample is left out
and the others are baked; the run still fails, naming it.

Handled here, on top of kimogen's spec: with stance_bookend moves, the move named by the spec's "stance" (default
idle_stance, the name kimogen hardcodes) goes first and its medoid frame becomes the stance they start and end in; a
new stance regenerates them. Per move, "seed" (default 42) draws another best-of-8, and "jitter_max" replaces
kimogen's jitter gate (mean joint acceleration, 0.015 m/frame^2) for fast moves such as runs and jumps, whose natural
swing exceeds it. The Llama-3 text encoder service (CPU, 16 GB) is started only when something is generated, and
stopped afterwards unless it was already running.

Additional gates (gate_sample): a loop move's cycle is searched in every sample (loops.py) and gated, and the
winner is cut to it (cut_loop) in place of kimogen's trim, which picks the stillest stretch of a repeated action and
cuts at whole frames; the contact gate also accepts feet at rest above the floor (stairs, a ladder, a seat), where
Kimodo's contact labels see none. strikes.py gates the authored striking limb's amplitude and supplies its timing.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from cli_args import ArgumentParser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'kimodo-practical', 'kimodo'))
import kimogen  # noqa: E402
import kimoconstraints  # noqa: E402
import bake_kimodo  # noqa: E402


def _load(name):  # by path: on sys.path, this folder's kimodo/ checkout would shadow the installed package
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


loops = _load('loops')
strikes = _load('strikes')
ENCODER_URL = 'http://127.0.0.1:9550/'
JITTER_MAX = kimogen.JITTER_MAX
CONTACT_FRAC = 0.05  # kimogen's contact gate: feet carry the body in at least this share of the frames
_gate_sample = kimogen.gate_sample


def gate_sample(j, r, fc, idx, mv, stance, fps):
    g = _gate_sample(j, r, fc, idx, mv, stance, fps)
    if g.get('malformed'):
        return g
    if not g['contact_ok']:
        g['support_frac'] = round(loops.support(j, idx, fps), 4)
        g['contact_ok'] = g['support_frac'] >= CONTACT_FRAC
    if mv.get('loop'):
        found = loops.find(j, r, idx, fps)
        g['loop_ok'] = found is not None and found['seam'] <= loops.LOOP_ERR_MAX
        if found:
            g.update(loop_seam=found['seam'], loop_motion=found['motion'],
                     loop_window=[found['start'], found['period']])
            g['score'] += found['seam'] * 10
    if mv.get('strike'):
        g.update(strikes.measure(j, r, idx, mv, fps)[0])
    g['pass'] = (not g['nonfinite'] and g['contact_ok'] and g['foot_skate_ok'] and g['jitter_ok']
                 and g['travel_ok'] and g['apex_ok'] and g['stance_ok'] and g.get('loop_ok', True)
                 and g.get('strike_ok', True))
    return g


kimogen.gate_sample = gate_sample
kimogen.frame_data = strikes.frame_data
kimogen.best_loop = lambda j, r, *_, **__: ((0, len(j) - 1), 0.0)  # keep the whole take: cut_loop cuts it


def run(module, *args):
    """module.main() with args; returns its exit code (argparse errors are already on stderr)."""
    old_argv = sys.argv
    sys.argv = [module.__file__, *args]
    try:
        module.main()
    except SystemExit as e:
        if isinstance(e.code, str):
            raise RuntimeError(e.code) from None
        return e.code or 0
    finally:
        sys.argv = old_argv
    return 0


class SpecOK(Exception):
    pass


def check(spec_path, names):
    """kimogen gen up to its model load, by which point it has validated the spec and the constraints."""
    def stop(_model):
        raise SpecOK
    load, kimogen.load_kimodo = kimogen.load_kimodo, stop
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            run(kimogen, 'gen', '--spec', spec_path, '--only', ','.join(names))
    except SpecOK:
        return
    finally:
        kimogen.load_kimodo = load
    reason = (err.getvalue().strip().splitlines() or ['rejected by kimogen'])[-1]
    raise ValueError(f'{spec_path}: {reason.removeprefix("kimogen.py: error: ")}')


def gates(mv):
    """The per-move settings kimogen takes per call: one gen call per distinct pair."""
    return mv.get('seed', 42), mv.get('jitter_max', JITTER_MAX)


def cut_loop(mv):
    """Cut an accepted loop move's whole take to the cycle its gates found (loops.cut); raises ValueError when the
    cycle leaves out a required constraint."""
    from kimodo.skeleton.definitions import SOMASkeleton77
    name = mv['name']
    npz_path, rep_path = (os.path.join(kimogen.MOVES_OUT, name + ext) for ext in ('.npz', '.json'))
    rep = json.load(open(rep_path))
    g = rep['gates']
    s, P = g.pop('loop_window')
    with np.load(npz_path) as f:
        z = {k: f[k] for k in f.files}
    skeleton = SOMASkeleton77()
    idx = {n: i for i, n in enumerate(skeleton.bone_order_names)}
    out, M, R2, p02 = loops.cut(z, s, P, skeleton, kimogen.canonicalize, idx)
    if mv.get('strike'):
        strike_metrics = strikes.measure(out['posed_joints'], out['root_positions'], idx, mv, int(z['fps']))[0]
        if not strike_metrics['strike_ok']:
            raise ValueError(f'{name}: the closed cycle no longer passes its strike gate')
        g.update(strike_metrics)
    resolved_path = os.path.join(kimogen.MOVES_OUT, name + '.resolved_constraints.json')
    if os.path.exists(resolved_path):
        resolved = json.load(open(resolved_path))
        kept = []
        for rec in resolved['records']:
            f = (rec['frame'] - s) * M / P
            if -0.5 <= f <= M + 0.5:
                kept.append({**rec, 'frame': int(round(min(max(f, 0), M)))})
            elif rec['required']:
                raise ValueError(f'{name}: its cycle (frames {s}-{s + P:.0f}) leaves out the required '
                                 f'{rec["type"]} constraint at frame {rec["frame"]}: author it inside one cycle, mark '
                                 'it "required": false, or drop "loop"')
        resolved.update(records=kimoconstraints.transform_records(kept, *kimoconstraints.compose_canonical(R2, p02)),
                        frames=M + 1)
        json.dump(resolved, open(resolved_path, 'w'), indent=1, allow_nan=False)
    np.savez_compressed(npz_path, **out)
    g['loop_err'], g['loop_trim'] = g.pop('loop_seam'), [s, round(s + P, 2)]
    rep['frames'] = M + 1
    rep['frame_data'] = kimogen.frame_data(out['posed_joints'], out['root_positions'], idx, mv, int(z['fps']))
    json.dump(rep, open(rep_path, 'w'), indent=1, allow_nan=False)
    print(f'[loop] {name}: frames {s}-{s + P:.2f} of the take -> a closed cycle of {M} frames, seam '
          f'{g["loop_err"]} m, {g["loop_motion"]}x the take\'s motion', file=sys.stderr)


def why(name):
    """The gates that failed across a rejected move's samples, e.g. 'loop_ok 8/8, jitter_ok 2/8'."""
    rep_path = os.path.join(kimogen.MOVES_OUT, name + '.json')
    if not os.path.exists(rep_path):
        return 'no report'
    samples = json.load(open(rep_path)).get('all_gates') or []
    counts = {}
    for g in samples:
        for k, v in g.items():
            if k.endswith('_ok') and v is False:
                counts[k] = counts.get(k, 0) + 1
    return ', '.join(f'{k} {n}/{len(samples)}' for k, n in sorted(counts.items(), key=lambda kv: -kv[1])) or 'none'


def encoder_up():
    try:
        urllib.request.urlopen(ENCODER_URL, timeout=5)
        return True
    except OSError:
        return False


def start_encoder(log_path):
    print('starting the text encoder service (loads 16 GB, 1-3 min)', file=sys.stderr)
    log = open(log_path, 'w')
    proc = subprocess.Popen([sys.executable, '-P', '-m', 'kimodo.scripts.run_text_encoder_server'], stdout=log,
                            stderr=subprocess.STDOUT, env={**os.environ, 'GRADIO_SERVER_NAME': '127.0.0.1'})
    log.close()
    deadline = time.monotonic() + 600
    try:
        while not encoder_up():
            if proc.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('text encoder service failed or timed out:\n' + ''.join(open(log_path).readlines()[-20:]))
            time.sleep(5)
    except BaseException:
        stop_encoder(proc)
        raise
    return proc


def stop_encoder(proc):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main(spec_path, out_dir):
    t0 = time.time()
    spec_path, out_dir = os.path.abspath(spec_path), os.path.abspath(out_dir)
    spec = json.load(open(spec_path))
    moves = spec.get('moves') if isinstance(spec, dict) else None
    if not isinstance(moves, list) or not moves or not all(isinstance(m, dict) for m in moves):
        raise ValueError(f'{spec_path}: expected {{"moves": [{{"name", "prompt", "duration", ...}}, ...]}}')
    for mv in moves:
        if not isinstance(mv.get('name'), str) or not kimogen.SAFE_MOVE_NAME.fullmatch(mv['name']):
            raise ValueError('move names must contain only letters, numbers, _ and -')
        strikes.validate(mv)
    by_name = {m.get('name'): m for m in moves}
    if len(by_name) != len(moves):
        raise ValueError('move names must be unique')
    bookended = [m for m in moves if m.get('stance_bookend')]
    stance = spec.get('stance', 'idle_stance')
    if bookended and (stance not in by_name or by_name[stance].get('stance_bookend')):
        raise ValueError(f'stance_bookend moves start and end in the stance of the spec\'s "stance" move ({stance}): '
                         'add it, not bookended itself')

    work = os.path.join(out_dir, 'gen')
    kimogen.OUT, kimogen.MOVES_OUT = work, os.path.join(work, 'moves')
    kimogen.STANCE_PATH = os.path.join(work, 'stance_pose.json')
    os.makedirs(kimogen.MOVES_OUT, exist_ok=True)
    state_path = os.path.join(work, 'state.json')
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}

    def key(mv):
        k = {'move': mv, 'fps': spec.get('fps')}
        if mv.get('loop'):
            k['loop_cut'] = loops.VERSION
        if mv.get('strike'):
            k['strike_gate'] = strikes.VERSION
        if mv.get('constraints_file'):
            k['constraints_file'] = hashlib.sha1(
                open(os.path.join(os.path.dirname(spec_path), mv['constraints_file']), 'rb').read()).hexdigest()
        if mv.get('stance_bookend'):
            k['stance'] = open(kimogen.STANCE_PATH).read() if os.path.exists(kimogen.STANCE_PATH) else None
        return hashlib.sha1(json.dumps(k, sort_keys=True).encode()).hexdigest()

    def done(mv):
        return state.get(mv['name']) == key(mv) and os.path.exists(os.path.join(kimogen.MOVES_OUT, mv['name'] + '.npz'))

    encoder, generated, rejected = None, [], {}  # rejected: name -> reason

    def generate(batch):
        nonlocal encoder
        check(spec_path, [mv['name'] for mv in batch])  # before the 16 GB encoder load
        if encoder is None and not encoder_up():
            encoder = start_encoder(os.path.join(work, 'encoder.log'))
        for seed, jitter_max in sorted({gates(mv) for mv in batch}):
            names = [mv['name'] for mv in batch if gates(mv) == (seed, jitter_max)]
            kimogen.JITTER_MAX = jitter_max  # read by kimogen.gate_sample
            t_call = time.time()
            try:
                code = run(kimogen, 'gen', '--spec', spec_path, '--only', ','.join(names), '--seed', str(seed))
            finally:  # keep what was accepted, even when a later move crashes
                for name in names:
                    npz, rep = (os.path.join(kimogen.MOVES_OUT, name + ext) for ext in ('.npz', '.json'))
                    state.pop(name, None)
                    if not os.path.exists(rep) or os.path.getmtime(rep) < t_call:
                        continue  # not reached (a crash): generated again next time
                    if not json.load(open(rep)).get('accepted') or not os.path.exists(npz):
                        rejected[name] = f'no sample passed ({why(name)})'
                        continue
                    try:
                        if by_name[name].get('loop'):
                            cut_loop(by_name[name])
                    except ValueError as e:
                        os.remove(npz)
                        rejected[name] = str(e)
                        continue
                    state[name] = key(by_name[name])
                    generated.append(name)
                json.dump(state, open(state_path, 'w'), indent=1)
            if code not in (0, 1):  # 1 = some move had no passing sample
                raise RuntimeError(f'kimogen gen failed for {", ".join(names)} (see stderr)')

    def extract_stance():  # kimogen reads the stance from <moves>/idle_stance.npz
        src = os.path.join(work, 'stance_source')
        os.makedirs(src, exist_ok=True)
        shutil.copyfile(os.path.join(kimogen.MOVES_OUT, stance + '.npz'), os.path.join(src, 'idle_stance.npz'))
        moves_out, kimogen.MOVES_OUT = kimogen.MOVES_OUT, src
        try:
            kimogen.extract_stance()
        finally:
            kimogen.MOVES_OUT = moves_out

    try:
        if bookended and not done(by_name[stance]):
            if os.path.exists(kimogen.STANCE_PATH):
                os.remove(kimogen.STANCE_PATH)
            generate([by_name[stance]])
        if bookended and stance in rejected:
            rejected.update((mv['name'], f'no stance: {stance} was rejected') for mv in bookended)
        elif bookended and not os.path.exists(kimogen.STANCE_PATH):
            extract_stance()
        todo = [mv for mv in moves if not done(mv) and mv['name'] not in rejected]
        if todo:
            generate(todo)
    finally:
        if encoder is not None:
            stop_encoder(encoder)
    if generated or rejected:
        run(kimogen, 'report')
        for mv in moves:
            report_path = os.path.join(kimogen.MOVES_OUT, mv['name'] + '.json')
            if mv.get('strike') and os.path.isfile(report_path):
                with open(report_path) as f:
                    g = json.load(f).get('gates', {})
                print(f'[strike] {mv["name"]}: {g.get("strike_tip", "?")}, '
                      f'speed {g.get("strike_speed", "?")} m/s, excursion {g.get("strike_excursion", "?")} m, '
                      f'pass={g.get("strike_ok", False)}', file=sys.stderr)

    for name in rejected:  # an older accepted clip of a now rejected move is not baked
        npz = os.path.join(kimogen.MOVES_OUT, name + '.npz')
        if os.path.exists(npz):
            os.remove(npz)
    baked = [mv['name'] for mv in moves if done(mv)]
    result = {'output': out_dir, 'moves': baked, 'generated': generated, 'rejected': list(rejected)}
    manifest_path = os.path.join(out_dir, 'manifest.json')
    old = json.load(open(manifest_path))['moves'] if os.path.exists(manifest_path) else []
    if baked:
        if run(bake_kimodo, '--spec', spec_path, '--in', kimogen.MOVES_OUT, '--web', out_dir,
               *(['--allow-missing'] if rejected else [])):
            raise RuntimeError('bake failed (see stderr)')
    else:
        # A fully rejected replacement must not expose the previous accepted set to add-moves.
        json.dump({'moves': [], 'source': 'kimodo'}, open(manifest_path, 'w'), indent=1)
    for mv in old:
        if mv['name'] not in baked and os.path.exists(os.path.join(out_dir, mv['file'])):
            os.remove(os.path.join(out_dir, mv['file']))
    if rejected:
        for name, reason in rejected.items():
            print(f'[reject] {name}: {reason}', file=sys.stderr)
        result['error'] = (f'rejected: {", ".join(rejected)}' + (f' (the other {len(baked)} moves are baked)' if baked
                           else '') + '; reword the prompt, lengthen the duration, relax the gate or set another '
                           '"seed" (the gate table is on stderr)')
    result['seconds'] = round(time.time() - t0, 1)
    return result


if __name__ == '__main__':
    ap = ArgumentParser(description='Generate and bake a Kimodo move set from a spec.')
    ap.add_argument('spec')
    ap.add_argument('out_dir')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    stdout = os.fdopen(os.dup(1), 'w')
    os.dup2(2, 1)  # everything the libraries print is progress: stderr
    try:
        result = main(a.spec, a.out_dir)
    except (ValueError, KeyError, OSError, RuntimeError) as e:
        print(f'gen-moves: {e}', file=sys.stderr)
        if a.json:
            print(json.dumps({'error': str(e)}), file=stdout)
        sys.exit(1)
    if 'error' in result:
        print(f'gen-moves: {result["error"]}', file=sys.stderr)
    if a.json:
        print(json.dumps(result), file=stdout)
    elif result['moves']:
        print(result['output'], file=stdout)
    sys.exit(1 if 'error' in result else 0)
