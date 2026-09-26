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
Use one character per GLB and apply mirrored or nonuniform skeleton scales before rigging. Shared body/mouth skins
and helper nodes between joints are supported; ambiguous duplicate humanoid bones are rejected.

- Clips are in place: hips X/Z stay at bind, height follows the move. `<name>_rootmotion.json` carries the travel
  for the game to move the entity by: `scaleRoot`, and per clip `scaleRoot`, `fps`, `numFrames`, `loop`, `frameData`, `hipY`,
  `pelvisXZ`, the hips' horizontal offset from the first frame, in metres in the character's frame (+z forward,
  +x the character's left), and `footContact`, per frame `[left, right]`, 1 while that foot is planted (on the floor
  or on anything else): footstep sounds go where a 0 turns 1, foot IK holds while it stays 1.
- Loop clips are closed cycles: the last frame repeats the first, so they loop as imported (Godot: `loop_mode =
  LINEAR`, length and tracks as imported; three.js / Bevy: repeat). One cycle lasts `(numFrames - 1) / fps` and
  travels `pelvisXZ[-1]`: at each wrap, continue the entity's path from there.
- Only the skeleton's Mixamo bones are keyed: a mouth rig's jaw stays free, so speech layers over any move. The
  mouth rig's `.face.json` is copied next to the output, where `lipsync --add` finds it.
  Other animations in the input are kept; one with a move's name is replaced.
- Proportions differ from Kimodo's body, so planted feet can slide and hands that reach the face in the source can
  touch a big-headed character's face. Look at a
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
| `strike`, `height` | `"hand"`/`"foot"` enables strike amplitude gates and attack frame data; optional `"low"`/`"mid"`/`"high"` is a timing label, not a target-height gate |
| `strike_gate` | `{side, min_speed, min_excursion}`: `"left"`, `"right"` or `"either"` (default); minimum limb speed (default 1 m/s, sustained for ~0.1 s, at least 3 intervals) and excursion (default 0.25 m), relative to the pelvis |
| `constraints` | Kimodo keyframes, hand/foot targets and root paths: kimodo-practical's `BAKE.md` §3 and `ANIMATION_AGENT.md` |
| `jitter_max` | the jitter gate, mean joint acceleration (default 0.015 m/frame²): fast moves exceed it by nature (the basic `run` and `jump`: 0.03) |
| `seed` | another best-of-8 draw (default 42) |

- Samples are gated on foot contact (feet at rest, at any height: stairs and seats count), foot skate, jitter,
  travel, apex, stance, strike amplitude, constraint adherence and, for a loop, its cycle (`loop_err`: the seam in metres, at most 0.06;
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
  one. Strike/loop algorithm versions also invalidate their cached moves. `<DIR>/gen/` holds the work (per-move NPZ
  and gate report); delete it to regenerate everything. A fully rejected run writes an empty manifest so the old
  accepted set cannot be mistaken for its replacement.
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

## Combat and held props

A smooth hand gesture can pass locomotion checks while making a poor sword attack. Specify the weapon, striking
hand, wind-up, direction and follow-through. Use one strike per clip, and set `strike_gate.side` to the hand carrying
the weapon: a fast off-hand gesture must not win selection or determine hit timing.
For repeatable guard/impact/follow-through poses, author keyframe or hand position/rotation `constraints`; a prompt
alone cannot enforce a blade path. Recheck those targets after transfer to the final character's proportions.

```json
{"fps": 30, "moves": [{
  "name": "sword_slash",
  "prompt": "A person firmly grips a one-handed sword in their right hand, winds up above their right shoulder, then makes one forceful diagonal downward slash across the front of their body toward an opponent at chest height, follows through to their left hip and returns to guard. The left hand stays near the chest.",
  "duration": 3.0, "travel": "in_place", "strike": "hand",
  "strike_gate": {"side": "right", "min_speed": 1.5, "min_excursion": 0.35},
  "jitter_max": 0.03
}]}
```

The gates reject weak movement, idle translation and the wrong hand. They do not measure blade orientation, grip,
collision or acting quality. Preview the transferred character with its weapon:

```bash
asset-blender tools/render_anim.py -- hero_moves.glb slash.png --action sword_slash \
  --prop sword.glb --bone RightHand --grip 0,-0.4,0 --scale 1 --rotation 0,0,0 --frames 12
```

`--grip X,Y,Z` is the handle point in the prop's original glTF scene coordinates (default origin). `--scale` sizes it;
`--rotation` is XYZ Euler degrees; `--offset` moves the grip from the wrist in metres. Both use the character's glTF
bind axes (+Y up, +Z forward), so Blender's imported bone rolls do not change their meaning. Fit these example grip
coordinates to the actual prop. The preview includes small meshes and keeps the prop rigid through the clip; it
does not write a combined GLB. Check for long pauses and trim the holds or reword the prompt for gameplay timing.

In Godot, use `Skeleton3D → BoneAttachment3D → Grip (Node3D) → weapon`. Set `bone_name` to the imported hand name
(usually `mixamorig_RightHand`), check that `find_bone` returns a valid index, and leave `override_pose` off. Fit the
handle and blade direction on `Grip` once; its transform is local to the imported bone, so do not paste the preview's
bind-axis offsets there unchanged. Inspect guard, wind-up, impact and follow-through, and match finger curl to the
handle. Two-handed weapons need supporting-hand constraint/IK fitted to the final character and weapon.
Godot's [BoneAttachment3D](https://docs.godotengine.org/en/stable/classes/class_boneattachment3d.html) follows the
evaluated bone transform; a skinning matrix including the inverse bind is not an attachment transform.

`frameData.active` and `contact` are source-frame estimates from the selected limb's speed (`contact_estimate: true`),
not verified weapon contact. Convert frames to seconds with the clip's `fps`, account for playback speed, and sweep
the blade between its previous and current positions during the active window. Trigger damage and impact effects
on the actual target intersection. If the visual swing is wrong, regenerate/reword it or use an authored Mixamo
clip; changing the prop offset or playing a weak gesture faster cannot supply missing wind-up and follow-through.

## How it works

`gen_moves.py` runs kimodo-practical's `kimogen.py` (best-of-8, numeric gates) and `bake_kimodo.py` in-process,
starting the text encoder service only when something needs generating. It replaces two of kimogen's gates.
Contact also accepts feet at rest above the floor, where Kimodo's contact labels see none. Loops use `loops.py`
instead of kimogen's trim, which picks the stillest stretch of a repeated action and cuts at whole frames. `strikes.py`
adds gates and timing for the authored striking limb before best-of-N selection; closed loops are checked again. In
every sample, `loops.py` searches for the cycle whose ends match best in pose and velocity, counted in frames of the
cycle's own motion, among the cycles that keep the take's motion. It refines the period below a frame, then cuts the
winner: resampled over the exact period, the remaining mismatch spread over the cycle, positions rebuilt by forward
kinematics.

`add_moves.py` does the transfer. Kimodo's SOMA skeleton has nearly the Mixamo layout and the T-pose as its zero
pose, so the transfer is direct: each bone takes its source joint's world rotation from rest. Two static rest
corrections keep it faithful: SOMA's T-pose hand bends ~18° off the forearm where rigs bind straight (the baked clips
carry the straightened rest), and thighs, shins, upper arms and forearms are aimed along the source T-pose once, so a
bind that is not an exact T-pose does not skew every pose. The root is scaled by the leg-length ratio. Hand and foot
rotations follow source deltas; fingers curl as in the source. This is rotation transfer, not target-aware IK.

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
The text encoder defaults to the CPU (`TEXT_ENCODER_DEVICE=cpu`, ~16 GB RAM) so that the GPU holds
only Kimodo (~2.5 GB); with ~16 GB of VRAM to spare, `TEXT_ENCODER_DEVICE=cuda` runs it there.
