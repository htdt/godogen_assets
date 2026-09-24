# godogen_assets — local asset generation

Free, local generators for [godogen](https://github.com/htdt/godogen) games: images, textured 3D models, rigged and
animated characters with talking mouths, sound effects and voice lines. Each tool is a one-shot command in `bin/`
with its own isolated environment; this repo holds the tools, their interfaces and their docs in one place.

| Command | Does | Runs on | Doc |
|---|---|---|---|
| `qwen-image` | text → PNG, `rgba` → transparent PNG, `edit` with reference images (Qwen-Image-2.1) | GPU | [image/](image/README.md) |
| `gen3d` | image → textured GLB (TRELLIS.2-4B) | GPU | [mesh/](mesh/README.md) |
| `mia-rig` | humanoid GLB → Mixamo-rigged GLB, optionally + a Mixamo clip (Make-It-Animatable) | CPU | [rig/](rig/README.md) |
| `add-moves` | rigged GLB → GLB with a Kimodo move set (17 fighting moves by default) + `rootmotion.json` | CPU | [motion/](motion/README.md) |
| `lipsync` | rigged GLB + voice line → mouth rig (jaw, lips, teeth) + lip-synced `speak` clip | CPU | [lipsync/](lipsync/README.md) |
| `stable-audio` | text → sound effect or seamless ambience loop (Stable Audio 3 small-sfx) | GPU | [sfx/](sfx/README.md) |
| `qwen-tts` | text → voice line, from a voice description or a reference clip (Qwen3-TTS 1.7B) | GPU | [voice/](voice/README.md) |
| `asset-blender` | runs a check script (`tools/`, `lipsync/`) in headless Blender 4.5 LTS | CPU | [below](#checking-results) |

## A character, start to finish

```bash
qwen-image rgba "a knight in T-pose, arms straight out, legs apart, face visible, mouth closed, neutral expression, \
single isolated 3D character model, full body from head to feet in frame, centered, front view, soft even studio \
lighting, no cast shadows, highly detailed, clean" -o hero.png
gen3d hero.png --faces 30000 --tex 1024 -o out/                    # -> out/hero.glb (game budget)
mia-rig out/hero.glb --fingers --anim none -o out/rig/             # -> out/rig/hero_rigged.glb
add-moves out/rig/hero_rigged.glb                                  # -> out/rig/hero_moves.glb + hero_rootmotion.json

# talking: voice line -> mouth rig -> moves + speech in one GLB
qwen-tts design "Halt, traveller." --voice "Male, around 40, stern castle guard" -o line.wav
echo "Halt, traveller." > line.txt
lipsync out/rig/hero_rigged.glb line.wav -t line.txt -o out/talk/  # -> hero_mouth.glb, hero_speak.glb
add-moves out/talk/hero_mouth.glb --speak line.wav -t line.txt     # -> out/talk/hero_moves.glb
```

Props stop after `gen3d`. Stock animation instead of Kimodo moves: `mia-rig --anim clip.fbx` with a Mixamo clip.

## Conventions

- **Output contract**, every command: stdout carries only the output path(s), or with `--json` one JSON object per
  result (`{"error": ...}` on failure); progress goes to stderr; exit 0 on success, 1 on failure. Keep stderr in a
  file and read it only on failure.
- **One GPU job at a time.** `qwen-image`, `gen3d`, `stable-audio`, `qwen-tts` and `motion/bake_mk.sh` each load a
  model onto the GPU; two at once run out of memory. `mia-rig`, `add-moves` and `lipsync` run on the CPU (lipsync
  renders its check images briefly on the GPU) and can overlap with a GPU job.
- **Minutes, not seconds.** Each call loads its model and exits. `qwen-image` (1024², 40 steps) and `gen3d` take
  minutes per asset, longer than common shell-tool timeouts: set a long timeout or run in the background, and time
  the first call (`seconds` / `gen_s` in the JSON) before planning a batch.
- **Frames and units.** `gen3d` GLBs are glTF Y-up, centred, height normalised to 1.0 (no metric scale: size them in
  the engine). `mia-rig` rigs are human scale (~1.8 m), soles at y=0, facing +Z, Mixamo bone names.
- **Isolation.** Every tool has its own environment inside its folder; the launchers drop the caller's venv,
  `PYTHONPATH` and `LD_LIBRARY_PATH`, so they behave the same from any shell or project.
- Without `-o`, results go to the tool's `outputs/` folder.

## Setup

Needs: Linux x86-64; NVIDIA GPU with ≥ 12 GB VRAM and driver ≥ 580 (CUDA 13 wheels in image/voice, CUDA 12.x
elsewhere); ≥ 24 GB RAM; ~130 GB free disk during setup; `git curl unzip gcc g++ ffmpeg`,
[uv](https://docs.astral.sh/uv/) and [micromamba](https://mamba.readthedocs.io/) on `PATH`; a Hugging Face token
(`hf auth login`) whose account accepted the [Stable Audio 3 small-sfx](https://huggingface.co/stabilityai/stable-audio-3-small-sfx)
licence (the only gated model; the other gated repos are replaced by public mirrors).

```bash
git clone <this repo> godogen_assets && cd godogen_assets
./setup.sh                     # all parts; or pick: blender image mesh rig motion lipsync sfx voice
./setup.sh link                # bin/* -> ~/.local/bin (or add bin/ to PATH)
echo "export KIMODO_HOME=$PWD/motion" >> ~/.bashrc   # (and ~/.zshrc) godogen's motion docs key on it
```

`setup.sh` is idempotent (re-run after a failure) and pins the tested upstream commits and package versions; each
part's README says what it installs. Versions are for Ampere/Ada GPUs: a newer GPU generation
(sm_120+) needs newer torch/CUDA pins in `mesh/` and `motion/`. Envs are not relocatable: after moving the repo,
delete the `.venv` / `.conda` / `motion/kimenv` folders and re-run `setup.sh`.

Check the install:

```bash
qwen-image info && stable-audio info && qwen-tts info
qwen-image rgba --resolution 512 --steps 10 "a red apple" -o /tmp/apple.png
gen3d --type 512 --no-preview /tmp/apple.png -o /tmp/gen3d && mia-rig --help >/dev/null && add-moves --help
```

## Checking results

You can't look at a GLB directly, so render it:

```bash
asset-blender tools/render_glb.py -- model.glb renders/            # 4 textured + 4 clay views, mesh stats JSON
asset-blender tools/pose_test.py -- hero_rigged.glb poses.png      # 6 stress poses of a rig (skinning check)
asset-blender tools/render_anim.py -- hero_moves.glb kick.png --action kick_high   # frames of one clip
python3 tools/glb_info.py hero_moves.glb                           # meshes, morph targets, skins, clips, bytes
```

`gen3d` also writes `<name>_preview.png`, and `lipsync` writes a check sheet and a video.

## Instead of Tripo (godogen asset-gen)

| Tripo CLI | Local |
|---|---|
| `tripo make ref.png -p face_limit=30000` | `gen3d ref.png --faces 30000 --tex 1024` (RGBA input from `qwen-image rgba`, no solid background; no `auto_size`: scale in the engine) |
| `--then rig-check,rig` | `mia-rig x.glb --fingers --anim none` (humanoids only) |
| `tripo anim retarget --animation preset:biped:*` | `mia-rig --anim clip.fbx` (Mixamo clips) or `add-moves` (Kimodo move set) |

## Layout

```
bin/       the commands (thin launchers into each part's environment)
image/     qwen-image    cli.py, generate.py (quantized pipeline), quantize.py
mesh/      gen3d         gen3d.py, lowmem.py (the 12 GB patches for TRELLIS.2), test_lowmem.py
rig/       mia-rig       mia_rig.py, Blender rig scripts (normalize_rig, merge_anim, rigops), make_templates.py
motion/    add-moves     KIMODO_HOME: add_moves.py (the transfer), setup.sh, bake_mk.sh (Kimodo, kimodo-practical)
lipsync/   lipsync       mouth_rig.py, lipsync.py, face_landmarks.py, check renders (face_test, hole_check)
sfx/       stable-audio  cli.py
voice/     qwen-tts      cli.py
tools/     render_glb.py, pose_test.py, render_anim.py, glb_info.py
```

Everything `setup.sh` builds or downloads (environments, upstream checkouts, weights, `deps/blender`) and every
`outputs/` folder is git-ignored.
