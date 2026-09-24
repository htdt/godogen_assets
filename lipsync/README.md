# lipsync — talking characters

Gives a `mia-rig` character a mouth that opens (jaw bone, cut lips, teeth, tongue, dark cavity, two lip shapes) and
lip-syncs it to a voice line. Automatic, no per-character tuning; faces from TRELLIS.2 (realistic and cartoon) both
work. Looks right at game distances (head-and-shoulders and wider); a full-screen close-up shows blurry baked lips and
generic teeth.

```bash
lipsync out/rig/hero_rigged.glb line.wav -t line.txt -o out/talk/        # mouth rig + "speak" clip, ~2 min
lipsync --add line_02 out/talk/hero_speak.glb line2.wav -t line2.txt     # one more clip on a mouth-rigged GLB
add-moves out/talk/hero_mouth.glb --speak line.wav -t line.txt           # Kimodo moves + "speak" in one GLB
```

Voice lines come from `qwen-tts` (keep the transcript: `-t` helps Rhubarb). For several lines of one character, clone
the voice (`qwen-tts clone`), then `lipsync --add` per line.

| Option | |
|---|---|
| `-o DIR` | default `lipsync/outputs/<name>/`; with `--add`: output GLB, default in place |
| `-t LINE.txt` | transcript (recognition hint) |
| `--phonetic` | Rhubarb's language-independent recognizer, for non-English lines |
| `--no-render` | skip the video (~1 min instead of ~2) |
| `--engine eevee\|cycles` | video renderer |
| `--hinge-depth F` | jaw hinge depth behind the lips in face widths (0.62) |
| `--face FACE.json` | `--add` only; default `<GLB without .glb>.face.json`, else `<name>_mouth.face.json` beside it |
| `--json` | `{output, mouth, face, video, check_sheet, seconds}` (`--add`: `{output, clip, face, seconds}`) |

## What you get

| File | |
|---|---|
| `<name>_mouth.glb` | the rig with a mouth, T-pose bind, no animation: skeleton + `mixamorig:Jaw` (child of Head), body mesh with morph targets `wide` / `round`, skinned `mouth_parts` (teeth + tongue) |
| `<name>_mouth.face.json` | jaw bone and axis, full opening `jaw_open_max_rad`, face frame, lip centre, mouth width, the rig report |
| `<name>_speak.glb` | the same + one glTF animation `speak` with exactly two channels: jaw rotation and the body's morph weights |
| `<name>_speak.mp4` | face close-up \| head and shoulders, with the audio |
| `poses.png` | check sheet: rest, jaw 40 / 100 %, wide, round, combinations; rows front, 3/4, sagittal cut |
| `work/` | landmark render + detections (`face_landmarks.png`), logs, Rhubarb cues |

Cost: ~1 % more triangles and ≤ 1.2 MB on a 1024 mesh, +7 % on a 30k mesh; body skinning is unchanged, so
`add-moves` treats the mouth rig like the plain rig and leaves the jaw to speech.

## In a game

- **Jaw**: bone `mixamorig:Jaw`; open = rotation about its local +X by `jaw × jaw_open_max_rad` on top of the rest
  rotation.
- **Lips**: morph targets `wide` and `round` (0..1) on every primitive of the body mesh.
- **Teeth + tongue**: mesh `mouth_parts`, base colour = vertex colour `COLOR_0` (baked occlusion, since real-time
  renderers do not darken a mouth). three.js and glTFast use it automatically; in Godot make sure vertex colour is
  used as albedo.
- **Clip**: `speak` (and each `--add` clip) touches only the jaw and the morph weights: play it on its own layer over
  any body clip, one clip per voice line, time-synced to the audio (`mixer.setTime(audio.currentTime)`). Clips are
  baked offline; for speech generated at runtime, drive the jaw from the audio's amplitude envelope instead.
- If a body clip from elsewhere keys the jaw at rest, strip that track or apply the speech layer after the body layer.

## Checks

The first three come with every run:

1. `work/face_landmarks.png`: the red polyline sits on the lip line from corner to corner. If it does not, nothing
   downstream can be right: regenerate the image (front-facing, visible face, closed mouth).
2. The report (last line of `work/mouth_rig.log`, `report` in the face JSON): `landmarks_on_mesh` ≥ 470 of 478,
   `lip_fit_residual` < 0.01, `resurface` not `skipped` with `rim_gaps` 0, `seam_loops` 1, `jaw_open_max_rad`
   0.22-0.42.
3. `poses.png`: the jaw opens without tearing, teeth visible, `wide` parts the lips, `round` pushes them forward; the
   sagittal cut shows teeth right behind the lips, tongue low, a dark cavity, nothing poking out of the chin.
4. Holes (magenta = see-through): `asset-blender lipsync/hole_check.py -- x_mouth.glb hc.png --jaw 0.6 --zoom`.
5. Jaw weights, if the chin or cheeks deform oddly: `asset-blender lipsync/face_test.py -- x_mouth.glb w.png --weights
   --views front,side` (red = follows the jaw; smooth falloff to the cheeks).
6. Contents: `python3 tools/glb_info.py x_speak.glb`: `speak` has 2 channels, morph targets are sparse, `mouth_parts`
   carries `COLOR_0`.

`landmark detection failed` means no face in the front render (helmet, mask, face turned away): regenerate the image.

## How it works

`mouth_rig.py` (Blender) welds the glTF seam duplicates, renders the head from the front and finds 478 face
landmarks with MediaPipe (`face_landmarks.py`, own venv), ray-casts them onto the mesh and fits the lip line. TRELLIS
closes mouths with folds (a second skin layer behind the lips), so the mouth area is deleted and rebuilt as one clean
layer re-sampled from the visible skin, then cut along the lip line. Jaw weights are a harmonic function on the cut
surface; the jaw hinge sits in front of the ears and the full opening scales with the mouth width. Teeth, tongue and
cavity are generated from the face frame, and `wide` / `round` are analytic displacement fields. `lipsync.py` runs
Rhubarb Lip Sync (mouth shapes A-H, X) on the audio, maps each shape to jaw / wide / round targets (`VISEMES`),
smooths them and keys one action. The code comments carry the details and every constant.

## Setup

`../setup.sh lipsync` (after `../setup.sh blender`): a uv venv with MediaPipe (`.venv`), the Face Landmarker model
(`models/`), Rhubarb Lip Sync 1.14 (`rhubarb/`). Rendering the video needs `ffmpeg`.
