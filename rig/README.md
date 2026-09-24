# mia-rig — Make-It-Animatable auto-rigging

Humanoid GLB → rigged GLB with [Make-It-Animatable](https://github.com/jasongzy/Make-It-Animatable) v1 (CVPR 2025):
a learned Mixamo skeleton, skin weights and pose reset, predicted from the mesh alone. Runs on the CPU (~1 min,
~5 GB RAM), so it can run while the GPU is busy.

```bash
mia-rig out/hero.glb --fingers --anim none -o out/rig/   # rig only -> out/rig/hero_rigged.glb
mia-rig out/hero.glb --anim Walking.fbx -o out/rig/      # + a Mixamo clip -> hero_anim.glb, hero_anim.fbx
```

| Option | Default | |
|---|---|---|
| `-o DIR` | `rig/outputs/<name>/` | |
| `--anim FBX\|none` | MIA's bundled "Standard Run" | a Mixamo clip (mixamo.com, "FBX, Without Skin"), retargeted onto the rig |
| `--fingers` | off (22 bones, single-bone hands) | 52 bones with fingers; use it for `add-moves` characters |
| `--rest-pose No\|T-pose\|A-pose` | `No` = predicted | pose of the input |
| `--no-inplace` | in place | keep the clip's root motion |
| `--no-upright` | upright | keep the torso lean the rest pose inherits (see below) |
| `--normal` | off | MIA's normal-aware skinning model |
| `--keep-work` | off | keep MIA's intermediates in `<out>/<name>/` (predicted joints, weights, rest pose) |
| `--json` | | `{input, output, out_dir, fingers[, anim_glb, anim_fbx]}` or `{"error": ...}` |

## What you get

- `<name>_rigged.glb`: Mixamo bone names (`mixamorig:*`), T-pose bind, skinned, no animation, human scale
  (~1.8-2.1 m), facing +Z, grounded (soles at y=0, origin between the feet). The input for `lipsync` and
  `add-moves`; Mixamo names also retarget in Godot (BoneMap), Unity (Humanoid) and Unreal (IK Retargeter).
  three.js's GLTFLoader strips the `:` (`mixamorigHips`).
- `<name>_anim.glb` / `_anim.fbx` with `--anim`: the rig plus the clip (the GLB is rebuilt with Blender from MIA's
  rest pose and FBX action, because MIA's own FBX2glTF export loses the metallic/roughness map).
- Two steps on top of MIA's demo pipeline: the torso is stood upright at bind (MIA straightens the limbs but keeps
  the torso leaning ~10° against the hips, a lean every transferred move would carry) and the rig is grounded (MIA
  keeps TRELLIS's frame with the hips at the origin).

Limits: humanoids only (two arms, two legs, one spine); anything held near the hands gets skinned to the forearm.
Poses other than T/A work (MIA resets them), but arms away from the body rig best.

Check a rig: `asset-blender tools/pose_test.py -- out/rig/hero_rigged.glb poses.png` (rest, arms down, arms up,
guard, squat, twist + kick) and, for a clip, `asset-blender tools/render_anim.py -- out/rig/hero_anim.glb anim.png`.

## How it works

`mia_rig.py` drives the functions behind MIA's Gradio demo headless, so results match the demo. `rigops.py`
(upright spine, shared with MIA's bpy 4.3), `normalize_rig.py` (grounding; optional `--upright-spine`,
`--straighten-legs`, `--feet-forward` for rigs from elsewhere) and `merge_anim.py` run in Blender 4.5.
`make_templates.py` rebuilds MIA's Mixamo skeleton templates.

## Setup

`../setup.sh rig` (after `../setup.sh blender`): MIA checkout (pinned) with its submodules; micromamba env `.conda`
with Python 3.11 and MIA's requirements (torch 2.1.2 cu121, PyG, pytorch3d, bpy 4.3); the v1 weights and data from
`jasongzy/Make-It-Animatable` (~2.5 GB, into the checkout); FBX2glTF. MIA's skeleton templates ship in the gated
`jasongzy/Mixamo` dataset, so `make_templates.py` rebuilds them from the Mixamo skeleton inside the bundled
"Standard Run.fbx".

Hardware: none beyond the CPU. MIA takes the GPU when it sees one, at no gain in speed, so `bin/mia-rig` hides it
(`CUDA_VISIBLE_DEVICES=`): the cu121 torch works whatever the GPU, and the GPU stays free for the GPU tools.
