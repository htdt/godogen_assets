"""
gen_moves.py: generate a move set from a spec with Kimodo and bake it for add-moves. Run by bin/gen-moves.

    python gen_moves.py SPEC.json OUT_DIR [--json]

Generation is kimodo-practical's kimogen.py (best-of-8 per move, numeric gates), baking its bake_kimodo.py; both run
in-process with their work folder redirected to OUT_DIR/gen (NPZ + gate report per move, stance_pose.json,
state.json, encoder.log). OUT_DIR gets manifest.json and one clip JSON per move.

Incremental: a move whose spec entry is unchanged since its last accepted generation is reused (state.json holds a
hash per move), so adding or rewording one move regenerates only that one.

Handled here, on top of kimogen's spec: with stance_bookend moves, the move named by the spec's "stance" (default
idle_stance, the name kimogen hardcodes) goes first and its medoid frame becomes the stance they start and end in; a
new stance regenerates them. Per move, "seed" (default 42) draws another best-of-8, and "jitter_max" replaces
kimogen's jitter gate (mean joint acceleration, 0.015 m/frame^2) for fast moves such as runs and jumps, whose natural
swing exceeds it. The Llama-3 text encoder service (CPU, 16 GB) is started only when something is generated, and
stopped afterwards unless it was already running.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'kimodo-practical', 'kimodo'))
import kimogen  # noqa: E402
import bake_kimodo  # noqa: E402

ENCODER_URL = 'http://127.0.0.1:9550/'
JITTER_MAX = kimogen.JITTER_MAX


def run(module, *args):
    """module.main() with args; returns its exit code (argparse errors are already on stderr)."""
    sys.argv = [module.__file__, *args]
    try:
        module.main()
    except SystemExit as e:
        if isinstance(e.code, str):
            raise RuntimeError(e.code) from None
        return e.code or 0
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
    while not encoder_up():
        if proc.poll() is not None:
            raise RuntimeError('text encoder service failed:\n' + ''.join(open(log_path).readlines()[-20:]))
        time.sleep(5)
    return proc


def main(spec_path, out_dir):
    t0 = time.time()
    spec_path, out_dir = os.path.abspath(spec_path), os.path.abspath(out_dir)
    spec = json.load(open(spec_path))
    moves = spec.get('moves') if isinstance(spec, dict) else None
    if not isinstance(moves, list) or not moves or not all(isinstance(m, dict) for m in moves):
        raise ValueError(f'{spec_path}: expected {{"moves": [{{"name", "prompt", "duration", ...}}, ...]}}')
    by_name = {m.get('name'): m for m in moves}
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
        if mv.get('constraints_file'):
            k['constraints_file'] = hashlib.sha1(
                open(os.path.join(os.path.dirname(spec_path), mv['constraints_file']), 'rb').read()).hexdigest()
        if mv.get('stance_bookend'):
            k['stance'] = open(kimogen.STANCE_PATH).read() if os.path.exists(kimogen.STANCE_PATH) else None
        return hashlib.sha1(json.dumps(k, sort_keys=True).encode()).hexdigest()

    def done(mv):
        return state.get(mv['name']) == key(mv) and os.path.exists(os.path.join(kimogen.MOVES_OUT, mv['name'] + '.npz'))

    encoder, generated, rejected = None, [], []

    def generate(batch):
        nonlocal encoder
        check(spec_path, [mv['name'] for mv in batch])  # before the 16 GB encoder load
        if encoder is None and not encoder_up():
            encoder = start_encoder(os.path.join(work, 'encoder.log'))
        for seed, jitter_max in sorted({gates(mv) for mv in batch}):
            names = [mv['name'] for mv in batch if gates(mv) == (seed, jitter_max)]
            kimogen.JITTER_MAX = jitter_max  # read by kimogen.gate_sample
            try:
                code = run(kimogen, 'gen', '--spec', spec_path, '--only', ','.join(names), '--seed', str(seed))
            finally:  # keep what was accepted, even when a later move crashes
                for name in names:
                    if os.path.exists(os.path.join(kimogen.MOVES_OUT, name + '.npz')):
                        state[name] = key(by_name[name])
                        generated.append(name)
                    else:
                        state.pop(name, None)
                json.dump(state, open(state_path, 'w'), indent=1)
            if code not in (0, 1):  # 1 = some move had no passing sample
                raise RuntimeError(f'kimogen gen failed for {", ".join(names)} (see stderr)')
            rejected.extend(name for name in names if name not in state)

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
            if stance in rejected:
                raise RuntimeError(f'no stance: no sample of {stance} passed the gates')
        if bookended and not os.path.exists(kimogen.STANCE_PATH):
            extract_stance()
        todo = [mv for mv in moves if not done(mv)]
        if todo:
            generate(todo)
    finally:
        if encoder is not None:
            encoder.terminate()
            encoder.wait()
    if generated or rejected:
        run(kimogen, 'report')
    if rejected:
        raise RuntimeError(f'no sample passed the gates for: {", ".join(rejected)} (reword the prompt, lengthen the '
                           'duration, relax the gate or set another "seed"; the gate table is on stderr)')

    manifest_path = os.path.join(out_dir, 'manifest.json')
    if os.path.exists(manifest_path):  # clips of moves dropped from the spec
        for mv in json.load(open(manifest_path))['moves']:
            if mv['name'] not in by_name and os.path.exists(os.path.join(out_dir, mv['file'])):
                os.remove(os.path.join(out_dir, mv['file']))
    if run(bake_kimodo, '--spec', spec_path, '--in', kimogen.MOVES_OUT, '--web', out_dir):
        raise RuntimeError('bake failed (see stderr)')
    return {'output': out_dir, 'moves': list(by_name), 'generated': generated, 'seconds': round(time.time() - t0, 1)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Generate and bake a Kimodo move set from a spec.')
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
    print(json.dumps(result) if a.json else result['output'], file=stdout)
