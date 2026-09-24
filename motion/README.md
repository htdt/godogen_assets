# Kimodo moves — add-moves

Humanoid move sets generated with [NVIDIA Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/)
(text- and constraint-conditioned motion diffusion, trained on studio mocap that covers videogame combat and
locomotion), generated and gated with [kimodo-practical](https://github.com/htdt/kimodo-practical)'s `kimogen.py`,
then put onto a Mixamo rig as ordinary glTF clips. This folder is `KIMODO_HOME`.

## Moves onto a character

```bash
add-moves out/rig/hero_rigged.glb                                 # -> out/rig/hero_moves.glb + hero_rootmotion.json
add-moves out/talk/hero_mouth.glb --speak line.wav -t line.txt    # mouth rig: moves + the "speak" clip
add-moves hero_rigged.glb --baked path/to/web_moves -o hero.glb   # another baked move set
```

The input is a `mia-rig` rig (`--fingers --anim none`) or its `lipsync` mouth rig: Mixamo bone names, a bind close
to a T-pose with flat feet, facing +Z. `add_moves.py` (numpy only, ~1 s) writes one glTF animation per move and keeps
the rest of the GLB as it is; a rig without the Mixamo core bones is an error.

Kimodo's SOMA skeleton has nearly the Mixamo layout and the T-pose as its zero pose, so the transfer is direct:
each bone takes its source joint's world rotation from rest. Two static rest corrections keep it faithful: SOMA's
T-pose hand bends ~18° off the forearm where rigs bind straight (the baked clips carry the straightened rest), and
thighs, shins, upper arms and forearms are aimed along the source T-pose once, so a bind that is not an exact T-pose
does not skew every pose. The root is scaled by the leg-length ratio. Measured on three MIA characters over the 17
MK moves: median foot pitch on planted frames 0.8-0.9°, wrist bend within 1.3° of the source; fingers curl as in the
source.

- Clips are in place: hips X/Z stay at bind, height follows the move. `<name>_rootmotion.json` carries the travel
  for the game to move the entity by: `scaleRoot`, and per clip `fps`, `numFrames`, `loop`, `frameData`, `hipY` and
  `pelvisXZ`, the hips' horizontal offset from the first frame, in metres in the character's frame (+z forward,
  +x the character's left).
- Only the skeleton's Mixamo bones are keyed: a mouth rig's jaw stays free, so speech layers over any move.
  Other animations in the input are kept; one with a move's name is replaced.
- Proportions differ from Kimodo's body, so planted feet can slide a little (toe speed p90 ~0.1 m/s on planted
  frames; walks ~0.4) and hands that reach the face in the source can touch a big-headed character's face. Look at a
  filmstrip per character: `asset-blender tools/render_anim.py -- hero_moves.glb kick.png --action kick_high`.

`--json`: `{output, rootmotion, baked}`. In the engine, play the clips by driving animation time with weight
crossfades (Godot and Bevy import glTF animations natively; Babylon: paused `AnimationGroup` + `goToFrame`) and move
the entity by `pelvisXZ`. godogen's `asset-gen/motion.md` has the engine-side rules (impact timing, gates against
props).

## The default move set

`add-moves` uses the 17-move MK set (kimodo-practical's validated `kimodo/moveset_mk.json`): `idle_stance`,
`walk_fwd`, `walk_back` (loops), `jump_up`, `crouch`, `block_high`, `jab`, `punch_heavy`, `uppercut`, `kick_front`,
`kick_high`, `kick_side`, `sweep`, `hit_head`, `hit_heavy`, `knockdown`, `victory`. Every move starts and ends in
the idle's stance, so clips chain. `bake_mk.sh` (run by setup) generates it into
`kimodo-practical/kimodo/out/web_mk`: GPU ~2.5 GB VRAM plus the text encoder on the CPU (~16 GB RAM), ~15 min, as
one GPU job. `bake_mk.sh --only sweep,jab` regenerates single moves and re-bakes the set.

## Custom moves

A game's own move set is a spec of prompts and constraints for `kimogen.py`, which generates best-of-N per move and
gates it (foot skate, jitter, stance bookends, apex heights, constraint adherence). Write the spec from
kimodo-practical's `BAKE.md` (spec and gates), `KIMODO.md` (running the generator) and `ANIMATION_AGENT.md` (which
control to use when); `kimodo/moveset_mk.json` is a worked example. `kimogen.py` writes into its own checkout, so
author in a per-project clone:

```bash
git clone "$KIMODO_HOME/kimodo-practical" moves && cd moves/kimodo
export TEXT_ENCODERS_DIR=$KIMODO_HOME/text_encoders TEXT_ENCODER_DEVICE=cpu TEXT_ENCODER_MODE=api
PY=$KIMODO_HOME/kimenv/bin/python
GRADIO_SERVER_NAME=127.0.0.1 $PY -P -m kimodo.scripts.run_text_encoder_server &  # once per session, loads 16 GB
$PY kimogen.py gen --spec my_moves.json && $PY kimogen.py report
$PY bake_kimodo.py --spec my_moves.json --web out/web_mine
add-moves hero_rigged.glb --baked out/web_mine
```

Run the venv's Python (`kimenv/bin/python -P -m ...`), not its console scripts (`kimodo_gen`, `kimodo_textencoder`):
they carry absolute shebangs. `-P` keeps the current directory off `sys.path`: started from `$KIMODO_HOME`, the
`kimodo/` checkout would shadow the installed package. `TEXT_ENCODER_MODE=api` makes a missing service an error; the
default `auto` silently loads the 16 GB encoder into every process instead. The service binds all interfaces unless
`GRADIO_SERVER_NAME` is set.

`add-moves` does not land authored hand/foot targets exactly on a character with other proportions (it transfers
rotations, like any retargeter). A move whose hand must hit a prop point needs kimodo-practical's own Stage 1/3
(`certify.mjs`, `prebake.mjs` with its constraint IK; README "For agents"), which is also the path for rigs without
Mixamo names.

## Setup

`../setup.sh motion`, into this folder:

- `kimodo/`: upstream NVIDIA Kimodo (pinned), editable-installed into `kimenv/` (Python 3.12, torch 2.6.0 cu124,
  the `[all]` extras; pip's cmake builds the MotionCorrection extension);
- `kimodo-practical/` (pinned): the generator wrapper, bake, the MK spec;
- `text_encoders/`: the Llama-3-8B base is gated, so `setup_text_encoder.py` assembles the same layout from the
  public byte-identical `NousResearch/Meta-Llama-3-8B-Instruct` mirror and the McGill-NLP LLM2Vec adapters
  (symlinks into the Hugging Face cache, ~16 GB);
- the Kimodo weights (`nvidia/Kimodo-SOMA-RP-v1.1`, ~1 GB) download on the first generation, then `bake_mk.sh` runs.

`export KIMODO_HOME=<repo>/motion` in the shell profile: godogen's motion docs key on it. The venv is not relocatable
(editable install, absolute paths in the encoder adapter configs): after moving the repo, delete `kimenv/` and
`kimodo-practical/text_encoders`, `text_encoders`, then re-run setup.
