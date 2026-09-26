"""
mia-rig: auto-rig a humanoid GLB with Make-It-Animatable (Mixamo skeleton + skin weights), headless, on the CPU.

    mia-rig hero.glb --fingers --anim none -o out/      # rig only (the input for add-moves / lipsync)
    mia-rig hero.glb --anim Walking.fbx -o out/         # + a Mixamo clip ("FBX, Without Skin")
    mia-rig --json hero.glb --fingers --anim none       # one JSON result line on stdout

Outputs in the output folder (default: rig/outputs/<name>/):
    <name>_rigged.glb   T-pose bind, skinned, Mixamo bone names, no animation, grounded (soles at y=0, origin
                        between the feet), human scale, facing +Z: the character to animate
    <name>_anim.glb     the same rig with the Mixamo clip (--anim) baked in (MIA's bundled "Standard Run" by default)
    <name>_anim.fbx     MIA's FBX of the animated rig (MIA's own frame, not grounded)
stdout carries only the rigged GLB path (or the JSON result with --json), progress goes to stderr; exit 0 / 1.

It drives the functions behind MIA's Gradio demo (app.py) directly, so the results match the demo, with one addition:
right after MIA resets the character to its T-pose rest, the torso is stood upright (hips -> chest vertical) and that
pose becomes the rest (--no-upright to skip). A torso that leans at bind (~10 deg) carries that lean
into every transferred move; MIA straightens the limbs but not the torso's lean against the hips.
"""
import os
import sys
import json
import shutil
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from cli_args import ArgumentParser

HERE = os.path.dirname(os.path.abspath(__file__))
MIA_DIR = os.path.join(HERE, 'Make-It-Animatable')
BLENDER = os.environ.get('BLENDER') or os.path.join(os.path.dirname(HERE), 'deps', 'blender', 'blender')


def main():
    p = ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input')
    p.add_argument('-o', '--out', default=None)
    p.add_argument('--anim', default=os.path.join(MIA_DIR, 'data', 'Standard Run.fbx'),
                   help='Mixamo animation FBX ("none" for a rig without animation)')
    p.add_argument('--fingers', action='store_true', help='keep finger bones (default: hands only)')
    p.add_argument('--rest-pose', default='No', choices=['No', 'T-pose', 'A-pose'],
                   help='pose of the input; "No" lets MIA predict it and reset the model to T-pose')
    p.add_argument('--normal', action='store_true', help='use the normal-aware skinning model')
    p.add_argument('--no-inplace', action='store_true', help='keep root motion of the animation')
    p.add_argument('--no-upright', action='store_true', help="keep the torso lean MIA's rest pose inherits")
    p.add_argument('--keep-work', action='store_true',
                   help="keep MIA's intermediates (<out>/<name>/: predicted joints, weights, rest pose) for debugging")
    p.add_argument('--json', action='store_true', help='print a JSON result on stdout')
    args = p.parse_args()

    # Keep stdout for the result only: MIA, bpy and Gradio print progress to fd 1.
    result_out = os.fdopen(os.dup(1), 'w')
    os.dup2(2, 1)
    code = 0
    try:
        result = rig(args)
        print(json.dumps(result) if args.json else result['output'], file=result_out, flush=True)
    except Exception as e:  # noqa: BLE001 - report any failure uniformly for callers
        import traceback
        traceback.print_exc()
        if args.json:
            print(json.dumps({'error': f'{type(e).__name__}: {e}'}), file=result_out, flush=True)
        code = 1
    # bpy (Blender as a module) segfaults during interpreter teardown; everything is written by now, so skip it.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def rig(args):
    src = os.path.abspath(args.input)
    name = os.path.splitext(os.path.basename(src))[0]
    out = os.path.abspath(args.out or os.path.join(HERE, 'outputs', name))
    anim = None if args.anim.lower() == 'none' else os.path.abspath(args.anim)
    os.makedirs(out, exist_ok=True)
    # MIA writes its outputs next to the input file, so work on a copy inside the output folder.
    work = os.path.join(out, f'{name}{os.path.splitext(src)[1]}')
    if os.path.abspath(work) != src:
        shutil.copy(src, work)

    sys.argv = sys.argv[:1]
    sys.path.insert(0, MIA_DIR)
    import gradio as gr
    # gr.Info/Warning/Success need a live Gradio request; print instead (patched before app's decorators bind them)
    gr.Info = gr.Warning = gr.Success = lambda message, *a, **k: print(message)
    import app  # chdirs into MIA_DIR in init_models()
    if not args.no_upright:
        patch_upright_rest()
    app.init_models()
    app.init_blocks()  # defines the Gradio components that the pipeline's return values refer to
    for _ in app._pipeline(
        work, is_gs=False, opacity_threshold=0.01, no_fingers=not args.fingers,
        rest_pose_type=args.rest_pose, ignore_pose_parts=[], input_normal=args.normal, bw_fix=True,
        bw_vis_bone='LeftArm', reset_to_rest=True, animation_file=anim, retarget=True,
        inplace=not args.no_inplace,
    ):
        pass

    # Collect the deliverables. MIA's own animated GLB (FBX2glTF) loses the metallic/roughness map, so the animated
    # GLB is rebuilt from rest.glb + the FBX's action with Blender (merge_anim.py).
    import subprocess
    mia_out = os.path.join(out, name)
    rigged = os.path.join(out, f'{name}_rigged.glb')

    def blender(script, *script_args):
        subprocess.run([BLENDER, '-b', '--factory-startup', '--python-exit-code', '1', '-P', os.path.join(HERE, script), '--', *script_args],
                       check=True, stdout=subprocess.DEVNULL)

    blender('normalize_rig.py', os.path.join(mia_out, 'rest.glb'), rigged)  # ground + centre (translation only)
    result = {'input': src, 'output': rigged, 'out_dir': out, 'fingers': args.fingers}
    if anim:
        result['anim_fbx'] = os.path.join(out, f'{name}_anim.fbx')
        result['anim_glb'] = os.path.join(out, f'{name}_anim.glb')
        shutil.copy(os.path.join(mia_out, f'{name}.fbx'), result['anim_fbx'])
        blender('merge_anim.py', rigged, os.path.join(mia_out, f'{name}.fbx'), result['anim_glb'])
    if not args.keep_work:
        shutil.rmtree(mia_out, ignore_errors=True)
        if work != src:
            os.remove(work)
    return result


def patch_upright_rest():
    """Wrap MIA's rest reset: after it applies its T-pose as the rest, stand the torso upright and apply again."""
    sys.path.insert(0, HERE)
    import rigops
    import bpy
    from util import blender_utils

    orig = blender_utils.set_rest_bones

    def set_rest_bones(armature_obj, *a, reset_as_rest=False, **k):
        result = orig(armature_obj, *a, reset_as_rest=reset_as_rest, **k)
        if reset_as_rest:
            lean = rigops.upright_spine(armature_obj, bpy.context.view_layer)
            if lean:
                print(f'Torso stood upright at bind (was leaning {lean:.1f} deg)')
                orig(armature_obj, reset_as_rest=True)
        return result

    blender_utils.set_rest_bones = set_rest_bones


if __name__ == '__main__':
    main()
