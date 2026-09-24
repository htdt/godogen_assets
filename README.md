# godogen_assets — local asset generation

Free, local generators for [godogen](https://github.com/htdt/godogen) games: images, textured 3D models, rigged and
animated characters with talking mouths, sound effects and voice lines. Each tool is a one-shot command in `bin/`
with its own isolated environment; this repo holds the tools, their interfaces and their docs in one place.

## Principles

1. **Simple.** One tool per task, a minimal setup. Exploration happens elsewhere; only its result lands here.
2. **Open and local, on a home GPU.** Open-source code and open-weight models, run on the local machine. Everything
   is tested on an RTX 3060 12 GB, ~200 € second-hand.
3. **Set up by an agent.** A small GPU needs quantization, a large one has other priorities: no single `setup.sh`
   can capture every machine and OS. Instead, the setup is a clean manual (the scripts are the tested recipe for the
   reference machine, the docs say what depends on the hardware), so a Claude Code or Codex agent sets it up
   quickly on the machine at hand.
4. **Assets for godogen.** This is godogen's asset part. Its only purpose is to give the gamedev agent the assets a
   game needs.
5. **Open parts, lean docs.** Not a monolithic product: when something is missing or broken, the gamedev agent can
   tune the part. Yet the internals stay out of its way, so it uses the tools quickly without polluting its context:
   this README and the top of a part's README are enough to use a tool.

## Tools

| Command | Does | Runs on | Doc |
|---|---|---|---|
| `qwen-image` | text → PNG, `rgba` → transparent PNG, `edit` with reference images (Qwen-Image-2.1) | GPU | [image/](image/README.md) |
| `gen3d` | image → textured GLB (TRELLIS.2-4B) | GPU | [mesh/](mesh/README.md) |
| `mia-rig` | humanoid GLB → Mixamo-rigged GLB, optionally + a Mixamo clip (Make-It-Animatable) | CPU | [rig/](rig/README.md) |
| `gen-moves` | move spec (text prompts) → baked humanoid move set, best-of-8 per move with numeric gates (Kimodo) | GPU | [motion/](motion/README.md) |
| `add-moves` | rigged GLB → GLB with Kimodo moves (idle, walk, run, jump, or `gen-moves` sets) + `rootmotion.json` | CPU | [motion/](motion/README.md) |
| `lipsync` | rigged GLB + voice line → mouth rig (jaw, lips, teeth) + lip-synced `speak` clip | CPU | [lipsync/](lipsync/README.md) |
| `stable-audio` | text → sound effect or seamless ambience loop (Stable Audio 3 small-sfx) | GPU | [sfx/](sfx/README.md) |
| `qwen-tts` | text → voice line, from a voice description or a reference clip (Qwen3-TTS 1.7B) | GPU | [voice/](voice/README.md) |
| `asset-blender` | runs a check script (`tools/`, `lipsync/`) in headless Blender 4.5 LTS | CPU | [below](#checking-results) |

A part's README starts with what using the tool needs: options, what comes out, limits. Its last sections, `How it
works` and `Setup`, are for tuning, fixing or installing the part.

## A character, start to finish

```bash
qwen-image rgba "a knight in T-pose, arms straight out, legs apart, face visible, mouth closed, neutral expression, \
single isolated 3D character model, full body from head to feet in frame, centered, front view, soft even studio \
lighting, no cast shadows, highly detailed, clean" -o hero.png
gen3d hero.png --faces 30000 --tex 1024 -o out/                    # -> out/hero.glb (game budget)
mia-rig out/hero.glb --fingers --anim none -o out/rig/             # -> out/rig/hero_rigged.glb
add-moves out/rig/hero_rigged.glb                                  # -> out/rig/hero_moves.glb + hero_rootmotion.json

# custom moves: a spec of prompts (motion/README.md) -> a move set, on top of the basic one
gen-moves hero_moves.json -o out/moves/hero                        # -> out/moves/hero/
add-moves out/rig/hero_rigged.glb --baked "$KIMODO_HOME/basic" --baked out/moves/hero

# talking: voice line -> mouth rig -> moves + speech in one GLB
qwen-tts design "Halt, traveller." --voice "Male, around 40, stern castle guard" -o line.wav
echo "Halt, traveller." > line.txt
lipsync out/rig/hero_rigged.glb line.wav -t line.txt -o out/talk/  # -> hero_mouth.glb, hero_speak.glb
add-moves out/talk/hero_mouth.glb                                  # -> out/talk/hero_moves.glb (jaw left free)
lipsync --add speak out/talk/hero_moves.glb line.wav -t line.txt   # + the "speak" clip, in place
```

Props stop after `gen3d`. Stock animation instead of Kimodo moves: `mia-rig --anim clip.fbx` with a Mixamo clip.

## Conventions

- **Output contract**, every command: stdout carries only the output path(s), or with `--json` one JSON object per
  result (`{"error": ...}` on failure); progress goes to stderr; exit 0 on success, 1 on failure. Keep stderr in a
  file and read it only on failure.
- **One GPU job at a time.** `qwen-image`, `gen3d`, `stable-audio`, `qwen-tts` and `gen-moves` each load a model
  onto the GPU; two at once run out of memory. `mia-rig`, `add-moves` and `lipsync` run on the CPU (lipsync
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

For the agent that sets this up. `setup.sh` is the tested recipe for the reference machine; on other hardware,
follow it part by part and adapt what the hardware changes (below). Parts are independent: install the ones the
games need (`blender` serves `rig`, `lipsync` and the checks in `tools/`).

Reference machine: Linux x86-64; RTX 3060 12 GB, driver ≥ 580; 24 GB RAM; ~130 GB free disk during setup;
`git curl unzip gcc g++ ffmpeg`, [uv](https://docs.astral.sh/uv/) and [micromamba](https://mamba.readthedocs.io/)
on `PATH`; a Hugging Face token (`hf auth login`) whose account accepted the
[Stable Audio 3 small-sfx](https://huggingface.co/stabilityai/stable-audio-3-small-sfx) licence (the only gated
model; the other gated repos are replaced by public mirrors).

```bash
git clone <this repo> godogen_assets && cd godogen_assets
./setup.sh                     # all parts; or pick: blender image mesh rig motion lipsync sfx voice
./setup.sh link                # bin/* -> ~/.local/bin (or add bin/ to PATH)
echo "export KIMODO_HOME=$PWD/motion" >> ~/.bashrc   # (and ~/.zshrc) godogen's motion docs key on it
```

`setup.sh` is idempotent (re-run after a failure) and pins the tested upstream commits and package versions; each
part's README (Setup) says what it installs and which choices are tied to the hardware. Envs are not relocatable:
after moving the repo, delete the `.venv` / `.conda` / `motion/kimenv` folders and re-run `setup.sh`.

What depends on the hardware:

- **GPU generation, driver.** `image` and `voice` use torch 2.14's CUDA 13 build (driver ≥ 580; an older driver
  needs a CUDA 12.x build). `mesh` (CUDA 12.4 toolkit, torch 2.6, a flash-attn wheel), `motion` (torch 2.6 cu124)
  and `sfx` (torch 2.7.1 cu126) are pinned for GPUs before Blackwell; Blackwell (sm_120) needs CUDA ≥ 12.8 builds
  there. `mesh` compiles its CUDA extensions for the GPU it finds. `rig` runs on the CPU whatever the GPU.
- **VRAM.** 12 GB is the target: `image` quantizes (int8 DiT, NF4 text encoder, CPU offload) and `mesh` runs
  `lowmem.py`'s exact patches; their READMEs give the knobs for less or more. `gen-moves` needs ~2.5 GB, `qwen-tts`
  ~4.6 GB.
- **RAM.** 24 GB fits the largest users one at a time: `gen-moves`' Llama-3 text encoder on the CPU (~16 GB) and
  `gen3d` (peaks ~16 GB). With ~16 GB of VRAM to spare, the encoder can run on the GPU (motion README, Setup).

Check the install:

```bash
qwen-image info && stable-audio info && qwen-tts info
qwen-image rgba --resolution 512 --steps 10 "a red apple" -o /tmp/apple.png
gen3d --type 512 --no-preview /tmp/apple.png -o /tmp/gen3d && mia-rig --help >/dev/null && add-moves --help
ls "$KIMODO_HOME/basic/manifest.json"                              # the basic move set, baked by setup
```

## Checking results

You can't look at a GLB directly, so render it:

```bash
asset-blender tools/render_glb.py -- model.glb renders/            # 4 textured + 4 clay views, mesh stats JSON
asset-blender tools/pose_test.py -- hero_rigged.glb poses.png      # 6 stress poses of a rig (skinning check)
asset-blender tools/render_anim.py -- hero_moves.glb jump.png --action jump        # frames of one clip
python3 tools/glb_info.py hero_moves.glb                           # meshes, morph targets, skins, clips, bytes
```

`gen3d` also writes `<name>_preview.png`, and `lipsync` writes a check sheet and a video.

## Layout

```
bin/       the commands (thin launchers into each part's environment)
image/     qwen-image    cli.py, generate.py (quantized pipeline), quantize.py
mesh/      gen3d         gen3d.py, lowmem.py (the 12 GB patches for TRELLIS.2), test_lowmem.py
rig/       mia-rig       mia_rig.py, Blender rig scripts (normalize_rig, merge_anim, rigops), make_templates.py
motion/    gen-moves     KIMODO_HOME: gen_moves.py (Kimodo via kimodo-practical), basic.json (the default set)
           add-moves     add_moves.py (the transfer)
lipsync/   lipsync       mouth_rig.py, lipsync.py, face_landmarks.py, check renders (face_test, hole_check)
sfx/       stable-audio  cli.py
voice/     qwen-tts      cli.py
tools/     render_glb.py, pose_test.py, render_anim.py, glb_info.py
```

Everything `setup.sh` builds or downloads (environments, upstream checkouts, weights, `deps/blender`) and every
`outputs/` folder is git-ignored.
