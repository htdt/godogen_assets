# Kimodo moves — gen-moves, add-moves

Humanoid moves generated from text prompts with [NVIDIA Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/)
(text- and constraint-conditioned motion diffusion, trained on studio mocap that covers locomotion, gestures,
everyday actions, videogame combat and dance). `gen-moves` generates and gates a move set with
[kimodo-practical](https://github.com/htdt/kimodo-practical)'s `kimogen.py` and bakes it; `add-moves` puts baked
moves onto a Mixamo rig as ordinary glTF clips.

## Moves onto a character

```bash
add-moves out/rig/hero_rigged.glb                                 # -> out/rig/hero_moves.glb + hero_rootmotion.json
add-moves out/talk/hero_mouth.glb                                 # a mouth rig: speech goes on with lipsync --add
add-moves hero_rigged.glb --baked motion/basic --baked out/moves/knight          # the basic set + custom moves
```

Without `--baked`, the moves are the basic set (below); `--baked` takes `gen-moves` outputs, repeatable, a later set
winning a name clash, and a relative path missing from the current directory is taken from the repo root
(`motion/basic`). The input is a `mia-rig` rig (`--fingers --anim none`) or its `lipsync` mouth rig: Mixamo bone
names, a bind close to a T-pose with flat feet, facing +Z. `add_moves.py` (numpy only, ~1 s) writes one glTF
animation per move and keeps the rest of the GLB as it is; a rig without the Mixamo core bones is an error.

- Clips are in place: hips X/Z stay at bind, height follows the move. `<name>_rootmotion.json` carries the travel
  for the game to move the entity by: `scaleRoot`, and per clip `fps`, `numFrames`, `loop`, `frameData`, `hipY`,
  `pelvisXZ`, the hips' horizontal offset from the first frame, in metres in the character's frame (+z forward,
  +x the character's left), and `footContact`, per frame `[left, right]`, 1 while that foot is planted (on the floor
  or on anything else): footstep sounds go where a 0 turns 1, foot IK holds while it stays 1.
- Loop clips are closed cycles: the last frame repeats the first, so they loop as imported (Godot: `loop_mode =
  LINEAR`, length and tracks as imported; three.js / Bevy: repeat). One cycle lasts `(numFrames - 1) / fps` and
  travels `pelvisXZ[-1]`: at each wrap, continue the entity's path from there.
- Only the skeleton's Mixamo bones are keyed: a mouth rig's jaw stays free, so speech layers over any move. The
  mouth rig's `.face.json` is copied next to the output, where `lipsync --add` finds it.
  Other animations in the input are kept; one with a move's name is replaced.
- Proportions differ from Kimodo's body, so planted feet can slide a little (toe speed p90 ~0.1 m/s on planted
  frames; walks ~0.4) and hands that reach the face in the source can touch a big-headed character's face. Look at a
  filmstrip per character: `asset-blender tools/render_anim.py -- hero_moves.glb jump.png --action jump`.

`--json`: `{output, rootmotion, baked}` (`baked`: the move sets used). In the engine, play the clips by driving
animation time with weight crossfades (Godot and Bevy import glTF animations natively; Babylon: paused
`AnimationGroup` + `goToFrame`) and move the entity by `pelvisXZ`. godogen's `asset-gen/motion.md` has the
engine-side rules (impact timing, gates against props).

## The basic set

`basic.json`, baked by setup into `basic/`: `idle`, `walk`, `run` (loops) and `jump` (starts and ends in the idle's
stance). It covers a character that stands and moves around; every other move a game needs is a custom one.

## Custom moves

A move set is a JSON spec of prompts. `gen-moves` generates each move best-of-8, gates the samples numerically and
bakes the winners into a folder for `add-moves --baked`:

```json
{"fps": 30, "moves": [
  {"name": "wave", "prompt": "A person waves hello with their right hand, then lowers it.", "duration": 3.0,
   "travel": "in_place"},
  {"name": "sneak", "prompt": "A person sneaks forward slowly, crouched low.", "duration": 4.0, "loop": true,
   "travel": "fwd"},
  {"name": "fall", "prompt": "A person collapses and falls backward onto the ground.", "duration": 3.0,
   "travel": null, "apex": {"kind": "root_floor", "max": 0.4}}
]}
```

```bash
gen-moves knight.json -o out/moves/knight                         # -> out/moves/knight/ (manifest.json + clips)
add-moves out/rig/knight_rigged.glb --baked out/moves/knight
```

| key | |
|---|---|
| `name`, `prompt`, `duration` | clip name; one action as "A person ..." with plain physical verbs; seconds (≤ 10) |
| `loop` | cycles (idles, walks, a repeated action): the take's most seamless cycle (≥ 1 s) among those that keep the take's motion. Without it, the whole take is baked |
| `travel` | `"fwd"`, `"back"`, `"in_place"` or `null`: gate on the net root travel |
| `apex` | gate on the move's defining moment, in metres: `root_rise` (jumps), `root_dip` (crouches), `root_floor` (falls, `max`), `ankle_height` (kicks), `foot_excursion` (sweeps, lunges), with `min` and/or `max` |
| `stance_bookend` | start and end in the stance of the move named by the spec's top-level `"stance"` (default `idle_stance`), which is generated first: one-shots chain with that idle, and their gates measure from standing |
| `strike`, `height` | `"hand"`/`"foot"`, `"low"`/`"mid"`/`"high"`: attack frame data (startup, active, contact) in `rootmotion.json` |
| `constraints` | Kimodo keyframes, hand/foot targets and root paths: kimodo-practical's `BAKE.md` §3 and `ANIMATION_AGENT.md` |
| `jitter_max` | the jitter gate, mean joint acceleration (default 0.015 m/frame²): fast moves exceed it by nature (the basic `run` and `jump`: 0.03) |
| `seed` | another best-of-8 draw (default 42) |

- Samples are gated on foot contact (feet at rest, at any height: stairs and seats count), foot skate, jitter,
  travel, apex, stance, constraint adherence and, for a loop, its cycle (`loop_err`: the seam in metres, at most 0.06;
  `loop_motion`: the cycle's share of the take's motion, at least 0.75); the best passing one wins. A move with no
  passing sample is left out and the others are baked; `gen-moves` then exits 1 naming it and the gates that failed,
  and the gate table on stderr has the numbers. Kimodo undershoots amplitude: say the height or distance plainly ("at
  head height"), gate what defines the move (a jump gated only on smoothness never leaves the ground; one without a
  bookend can start in a squat and pass its rise by standing up), and when every sample fails, reword stronger or
  lengthen the duration.
- One action per move: split combos into moves. Kimodo also makes an action once: "repeatedly hammers" comes back as
  one or two strikes, so its loop fails the motion gate. For a repeated action, prompt one repetition that ends where
  it started ("raises the hammer, strikes down, and raises it again") with `loop`: the cycle runs from rest to rest.
  Generate the natural tempo and play it faster in the engine rather than squeezing the duration.
- Details that fix a position (a seat height, which shoulder carries the load, how far a hand reaches) are
  suggestions to the model. Measure the clip and fit the scene to it: `hipY` in `rootmotion.json`, or `pos` in the
  baked clip JSON (per frame, the joint positions in metres, joint order in `names`). To force a position, pin the
  hand or foot with `constraints`.
- Incremental: a move whose spec entry is unchanged is reused, so adding or rewording a move regenerates only that
  one. `<DIR>/gen/` holds the work (per-move NPZ and gate report); delete it to regenerate everything.
- Cost: ~1 min GPU per move (~2.5 GB VRAM, one GPU job) plus the Llama-3 text encoder, a CPU service (~16 GB RAM) that
  `gen-moves` starts when something needs generating (1-3 min) and stops afterwards. For a session of runs, start it
  once from the repo root and `gen-moves` reuses it:
  `GRADIO_SERVER_NAME=127.0.0.1 TEXT_ENCODERS_DIR=$PWD/motion/text_encoders TEXT_ENCODER_DEVICE=cpu
  motion/kimenv/bin/python -P -m kimodo.scripts.run_text_encoder_server &` (without `GRADIO_SERVER_NAME` it binds all
  interfaces).

`--json`: `{output, moves, generated, rejected, seconds}` (`moves`: the baked ones; with a rejected move also
`error`, exit 1) or `{error}`. Look at every new move on a character before using it.

`add-moves` transfers rotations, like any retargeter, so hand and foot targets authored in `constraints` do not land
exactly on a character with other proportions.

## How it works

`gen_moves.py` runs kimodo-practical's `kimogen.py` (best-of-8, numeric gates) and `bake_kimodo.py` in-process,
starting the text encoder service only when something needs generating. It replaces two of kimogen's gates.
Contact also accepts feet at rest above the floor, where Kimodo's contact labels see none. Loops use `loops.py`
instead of kimogen's trim, which picks the stillest stretch of a repeated action and cuts at whole frames. In
every sample, `loops.py` searches for the cycle whose ends match best in pose and velocity, counted in frames of the
cycle's own motion, among the cycles that keep the take's motion. It refines the period below a frame, then cuts the
winner: resampled over the exact period, the remaining mismatch spread over the cycle, positions rebuilt by forward
kinematics.

`add_moves.py` does the transfer. Kimodo's SOMA skeleton has nearly the Mixamo layout and the T-pose as its zero
pose, so the transfer is direct: each bone takes its source joint's world rotation from rest. Two static rest
corrections keep it faithful: SOMA's T-pose hand bends ~18° off the forearm where rigs bind straight (the baked clips
carry the straightened rest), and thighs, shins, upper arms and forearms are aimed along the source T-pose once, so a
bind that is not an exact T-pose does not skew every pose. The root is scaled by the leg-length ratio. Planted feet
stay within ~1° of flat, wrists within ~1.3° of the source bend, and fingers curl as in the source.

## Setup

`../setup.sh motion`, into this folder:

- `kimodo/`: upstream NVIDIA Kimodo (pinned), editable-installed into `kimenv/` (Python 3.12, torch 2.6.0 cu124,
  the `[all]` extras; pip's cmake builds the MotionCorrection extension);
- `kimodo-practical/` (pinned): the generator, gates and bake that `gen-moves` runs;
- `text_encoders/`: the Llama-3-8B base is gated, so `setup_text_encoder.py` assembles the same layout from the
  public byte-identical `NousResearch/Meta-Llama-3-8B-Instruct` mirror and the McGill-NLP LLM2Vec adapters
  (symlinks into the Hugging Face cache, ~16 GB);
- the Kimodo weights (`nvidia/Kimodo-SOMA-RP-v1.1`, ~1 GB) download on the first generation, when setup bakes
  `basic.json` into `basic/` (~5 min, one GPU job).

The venv is not relocatable (editable install, absolute paths in the encoder adapter configs): after moving the repo,
delete `kimenv/` and `kimodo-practical/text_encoders`, `text_encoders`, then re-run setup.

Hardware: torch 2.6.0 cu124 covers GPUs before Blackwell; Blackwell (sm_120) needs a CUDA ≥ 12.8 torch build.
The text encoder runs on the CPU (`TEXT_ENCODER_DEVICE=cpu` in `bin/gen-moves`, ~16 GB RAM) so that the GPU holds
only Kimodo (~2.5 GB); with ~16 GB of VRAM to spare, `TEXT_ENCODER_DEVICE=cuda` runs it there.
