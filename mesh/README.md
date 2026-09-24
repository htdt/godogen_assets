# gen3d — TRELLIS.2 image → GLB

Image → textured GLB (PBR base colour + metallic/roughness) with [TRELLIS.2-4B](https://github.com/microsoft/TRELLIS.2)
(Microsoft, MIT), patched (`lowmem.py`) to run on a 12 GB GPU.

```bash
gen3d hero.png -o out/                              # 1024_cascade: best detail, ~480k triangles, 2048² textures
gen3d --faces 30000 --tex 1024 crate.png -o out/    # game budget (= Tripo's face_limit=30000)
gen3d --type 512 --no-preview *.png -o drafts/      # 3-7x faster, for prompt trials and background props
gen3d --json a.png b.png -o out/                    # images in one call share the model load; one JSON line each
```

Writes `<out>/<name>.glb`, `<name>_preview.png` (front/right/back/left, PBR-shaded on top, normals below) and
`<name>.json` (settings, timings, peak VRAM / RAM, token counts, face count).

| Option | Default | |
|---|---|---|
| `-o DIR` | `mesh/outputs/` | |
| `--type` | `1024_cascade` | `512`, `1024`, `1024_cascade`, `1536_cascade` (slowest, most detail) |
| `--faces N` | 500000 | decimation target of the exported mesh |
| `--tex N` | 2048 | baked texture size |
| `--seed N` | 42 | a new seed is the fix for a bad result |
| `--steps N` | 12 per stage | |
| `--max-tokens N` | 49152 | cascade token cap; lower = less VRAM on very dense shapes |
| `--suffix S` | | appended to the output names |
| `--no-preview` | | |
| `--save-latent` / `--from-latent x_latent.pt` | | re-export (other `--faces` / `--tex`) without re-sampling |
| `--json` | | per image the `<name>.json` stats plus `output` and `preview`, or `{"input", "error"}` |

## Input images

- RGBA with a clean alpha is best (`qwen-image rgba`): the alpha is the mask. RGB inputs get BiRefNet background
  removal. No solid background needed.
- One isolated subject, whole object in frame, centred, soft even light, no cast shadows. Suffix that works:
  `single isolated 3D game asset, full object in frame, centered, three-quarter front view, soft even studio
  lighting, no cast shadows, highly detailed, clean`; characters: `single isolated 3D character model, full body
  from head to feet in frame, centered, front view, soft even studio lighting, no cast shadows, highly detailed, clean`.
- Characters for `mia-rig`: T-pose or wide A-pose, arms clearly away from the body, legs apart, nothing held near
  the hands (it gets skinned to the forearm). For `lipsync` also: face visible, mouth closed, neutral expression, no
  helmet or mask.
- Objects come out re-centred to face front, which suits game assets (the input's camera angle is not kept).

## What comes out

- glTF Y-up, centred at the origin, height normalised to 1.0: no metric scale, give each asset its real size in the
  engine (rigged characters get human scale from `mia-rig`).
- Hard-surface props and furniture are the strongest case (thin parts such as a wire handle survive). Characters are
  complete and rig-ready; faces are decent at 1024 and soft at 512, hands mitten-like, backs invented. Foliage
  becomes clumps (fine for stylised trees).
- Typical defects: small holes, inner shells, glass and transparent parts export opaque.
- At a 30k budget the texture carries the detail; geometry loses small folds and facial relief, but rigging and
  animation behave the same as at full resolution.
- Time grows with surface detail: at 1024, a character takes a few minutes and a leafy tree up to ~10; export adds
  seconds. Host RAM peaks around 16 GB, so keep other RAM-heavy jobs off while it runs.

## How it works

`gen3d.py` runs TRELLIS.2's image-to-3D pipeline and GLB export. Upstream asks for a 24 GB GPU; `lowmem.py` patches
it to fit 12 GB. Sampling fits in 3-6 GB with TRELLIS.2's `low_vram` offloading; decoding and export did not:

1. The last 512³ → 1024³ upsampling block of both sparse VAE decoders runs on x-axis slabs with a 2-voxel halo, and
   the ConvNeXt MLPs run in chunks. The result is bit-identical (`test_lowmem.py` checks it on a saved latent).
2. PyTorch's cached VRAM is released before CuMesh (hole filling, simplification, remeshing) allocates its own.
3. The export's dual-contouring remesh retries on a 768 / 512 grid when the 1024 grid does not fit.
4. The cascade may pick a resolution below 1024 for extremely dense shapes (`--max-tokens`).
5. Host RAM: only the sub-models of the chosen pipeline load, freed heap goes back to the OS, and the launcher sets
   `MALLOC_MMAP_THRESHOLD_` against glibc fragmentation.

## Setup

`../setup.sh mesh`:

- TRELLIS.2 checkout (pinned commit) in `TRELLIS.2/`;
- micromamba env `.conda`: Python 3.10, the CUDA 12.4 toolkit from conda-forge (the extensions are compiled with it,
  not with a system nvcc), torch 2.6.0 cu124, the flash-attn 2.7.3 wheel;
- CUDA extensions (nvdiffrast, nvdiffrec, CuMesh, FlexGEMM, o-voxel; pinned commits in `extensions/`) compiled for
  the local GPU's architecture, several minutes;
- weights: `microsoft/TRELLIS.2-4B` into `models/` (~14 GB), the rest into the Hugging Face cache.

The official DINOv3 and RMBG-2.0 repos are gated, so gen3d uses the ungated byte-identical mirror
`camenduru/dinov3-vitl16-pretrain-lvd1689m` and the MIT `ZhengPeng7/BiRefNet` (same architecture as RMBG-2.0)
instead; override with `TRELLIS_DINOV3=` / `TRELLIS_REMBG=`. Two pins matter: `transformers` 4.57 (5.x changed
DINOv3's layer layout) and `opencv-python-headless` < 5 (5.x cannot read the EXR environment map of the preview).

Hardware: the CUDA 12.4 toolkit, torch 2.6.0 cu124 and the flash-attn wheel (CUDA 12, torch 2.6, Python 3.10) go
together and cover GPUs before Blackwell; Blackwell (sm_120) needs a CUDA ≥ 12.8 toolkit, a matching torch and
flash-attn, and the extensions rebuilt. The extensions compile for the GPU `nvidia-smi` reports
(`TORCH_CUDA_ARCH_LIST`); `MAX_JOBS=4` limits the compile's RAM. The patches are exact, so they stay on with more
VRAM; with less, `--type 512` and a lower `--max-tokens` are the knobs.
