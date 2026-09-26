"""asset-blender's script runner: Blender diagnostics on stderr, explicit results on inherited fd 3."""
import json
import os
import runpy
import sys

result_out = os.fdopen(os.dup(3), 'w')
args = sys.argv[sys.argv.index('--') + 1:]
script, args = args[0], args[1:]
as_json = '--json' in args
args = [a for a in args if a != '--json']
if args[:1] == ['--']:
    args = args[1:]


def asset_result(result):
    if as_json:
        print(json.dumps(result), file=result_out, flush=True)
    else:
        for path in result.get('outputs', [result.get('output')]):
            if path:
                print(path, file=result_out, flush=True)


sys.argv = [script, '--', *args]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
try:
    runpy.run_path(script, run_name='__main__', init_globals={'asset_result': asset_result})
except BaseException as error:
    if isinstance(error, SystemExit) and error.code in (None, 0):
        pass
    else:
        if as_json:
            print(json.dumps({'error': str(error)}), file=result_out, flush=True)
        if isinstance(error, SystemExit):
            print(str(error), file=sys.stderr)
            raise SystemExit(1)
        raise
