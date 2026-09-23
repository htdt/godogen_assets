# Kimodo moves — add-moves

Humanoid move sets generated with [NVIDIA Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/)
(text- and constraint-conditioned motion diffusion, trained on studio mocap that covers videogame combat and
locomotion) through [kimodo-practical](https://github.com/htdt/kimodo-practical): moves are generated and gated once
on Kimodo's skeleton, then retargeted onto any certified rig and shipped as ordinary glTF clips. This folder is
`KIMODO_HOME`.

## Moves onto a character

```bash
add-moves out/rig/hero_rigged.glb                                 # -> out/rig/hero_moves.glb + hero_rootmotion.json
add-moves out/talk/hero_mouth.glb --speak line.wav -t line.txt    # mouth rig: moves + the "speak" clip
add-moves hero_rigged.glb --baked path/to/web_moves -o hero.glb   # another baked move set
```

The input is a `mia-rig` rig (`--fingers --anim none`) or its `lipsync` mouth rig. Steps, all on the CPU (~1 min):

1. `certify.mjs`: the rig retargets cleanly (bone roles resolved, round-trip and absolute gates);
2. `qa_endeffectors.mjs --gate`: feet stay flat, hands track, on every move;
3. `prebake.mjs`: one glTF animation per move, in place (hips X/Z zeroed, Y kept), and `rootmotion.json`
   (`scaleRoot`, per clip `fps`, `numFrames`, `loop`, per-frame `pelvisXZ` in character metres) for the game to
   integrate at the entity layer;
4. mouth rigs: `--speak` adds the lip-sync clip; the moves never key the jaw, so speech layers over any move.

Exit 1 means certification or the gate failed: regenerate the character (new seed or image), never patch the rig.
Measured on MIA rigs: footFlat 1.7-1.9° (limit 6), footGround ≤ 0.055 (0.10), hand end-effector 11-14° (15, the
tightest; `--fingers` gives it the most margin). `--json`: `{output, rootmotion, certificate, baked}`.

In the engine, play the clips by driving animation time with weight crossfades (Godot and Bevy import glTF
animations natively; Babylon: paused `AnimationGroup` + `goToFrame`), and move the entity by `rootmotion.json`.
godogen's `asset-gen/motion.md` has the engine-side rules (impact timing, gates against props).

## The default move set

`add-moves` uses the 17-move MK set (kimodo-practical's validated `kimodo/moveset_mk.json`): `idle_stance`,
`walk_fwd`, `walk_back` (loops), `jump_up`, `crouch`, `block_high`, `jab`, `punch_heavy`, `uppercut`, `kick_front`,
`kick_high`, `kick_side`, `sweep`, `hit_head`, `hit_heavy`, `knockdown`, `victory`. Every move starts and ends in
the idle's stance, so clips chain. `bake_mk.sh` (run by setup) generates it into
`kimodo-practical/kimodo/out/web_mk`: GPU ~2.5 GB VRAM plus the text encoder on the CPU (~16 GB RAM), ~15 min, as
one GPU job. `bake_mk.sh --only sweep,jab` regenerates single moves and re-bakes the set.

## Custom moves

A game's own move set (state machines, root-motion locomotion, poses authored in the game) follows kimodo-practical's
docs: README "For agents", then KIMODO.md → ALIGN.md → BAKE.md → INTEGRATE.md, and ANIMATION_AGENT.md while writing
move specs. `kimogen.py` writes into its own checkout, so author in a per-project clone:

```bash
git clone "$KIMODO_HOME/kimodo-practical" moves && cd moves && npm install
export TEXT_ENCODERS_DIR=$KIMODO_HOME/text_encoders TEXT_ENCODER_DEVICE=cpu TEXT_ENCODER_MODE=api
PY=$KIMODO_HOME/kimenv/bin/python
GRADIO_SERVER_NAME=127.0.0.1 $PY -P -m kimodo.scripts.run_text_encoder_server &  # once per session, loads 16 GB
cd kimodo && $PY kimogen.py gen --spec my_moves.json && $PY kimogen.py report
$PY bake_kimodo.py --spec my_moves.json --web out/web_mine
add-moves hero_rigged.glb --baked out/web_mine
```

Run the venv's Python (`kimenv/bin/python -P -m ...`), not its console scripts (`kimodo_gen`, `kimodo_textencoder`):
they carry absolute shebangs. `-P` keeps the current directory off `sys.path`: started from `$KIMODO_HOME`, the
`kimodo/` checkout would shadow the installed package. `TEXT_ENCODER_MODE=api` makes a missing service an error; the default `auto` silently
loads the 16 GB encoder into every process instead. The service binds all interfaces unless `GRADIO_SERVER_NAME`
is set.

## Characters

The lib is rig-agnostic: its bone mapping resolves Mixamo, Tripo, UE, VRoid and Meshy names and falls back to
topology, so `mia-rig` rigs certify as they are.

- Characters come from `gen3d` → `mia-rig --fingers --anim none`. Certify + gate still run per character: they take
  seconds and catch skeleton problems such as a torso leaning at bind or feet below the origin (`mia-rig` fixes both
  at bind).
- Stock clips (walk, idle, wave) are Mixamo FBX through `mia-rig --anim`; Kimodo is for custom or chained move sets.
- Applying a baked set needs no clone of the lib (`add-moves` uses this folder's copy); clones are for authoring.

## Setup

`../setup.sh motion`, into this folder:

- `kimodo/`: upstream NVIDIA Kimodo (pinned), editable-installed into `kimenv/` (Python 3.12, torch 2.6.0 cu124,
  the `[all]` extras; pip's cmake builds the MotionCorrection extension);
- `kimodo-practical/` (pinned) + `npm install` (three, gltf-transform);
- `text_encoders/`: the Llama-3-8B base is gated, so `setup_text_encoder.py` assembles the same layout from the
  public byte-identical `NousResearch/Meta-Llama-3-8B-Instruct` mirror and the McGill-NLP LLM2Vec adapters
  (symlinks into the Hugging Face cache, ~16 GB);
- the Kimodo weights (`nvidia/Kimodo-SOMA-RP-v1.1`, ~1 GB) download on the first generation, then `bake_mk.sh` runs.

`export KIMODO_HOME=<repo>/motion` in the shell profile: godogen's motion docs key on it. The venv is not relocatable
(editable install, absolute paths in the encoder adapter configs): after moving the repo, delete `kimenv/` and
`kimodo-practical/text_encoders`, `text_encoders`, then re-run setup.
