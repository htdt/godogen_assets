# qwen-image — Qwen-Image-2.1

Text-to-image, transparent (RGBA) generation and image editing with
[Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (diffusers `QwenImage21Pipeline`), quantized to fit a
12 GB GPU.

```bash
qwen-image generate "a red apple on a wooden table" -o apple.png
qwen-image rgba "a wooden shield, game icon" -o shield.png                # real alpha, no matting needed
qwen-image edit -i apple.png "change the background to a sunset beach" -o beach.png
qwen-image edit -i a.png -i b.png "put the character from image 1 into scene 2" -o out.png   # up to 10 references
qwen-image generate --size 2048x2048 --seed 7 "..."                        # 2K; VAE tiling switches on by itself
echo "prompt" | qwen-image generate - -o out.png                           # prompt from stdin
qwen-image info                                                             # GPU, versions, model status
```

Options for `generate` / `rgba` / `edit`:

| Option | Default | |
|---|---|---|
| `-o PATH` | `image/outputs/<timestamp>.png` | PNG |
| `--size WxH` | square at `--resolution`; `edit`: the last input's aspect ratio | multiples of 32 |
| `--resolution N` | 1024 | side length when `--size` is not given |
| `--steps N` | 40 | |
| `--seed N` | random | |
| `--cfg F` / `--negative TEXT` | 1.0 | cfg > 1 enables the negative prompt and doubles compute; the model is meant to run at 1.0 |
| `--quant int8\|nf4` | int8 | transformer precision; nf4 is smaller, a bit faster, slightly worse |
| `--json` | | `{output, width, height, mode, prompt, seed, steps, cfg, inputs, dit_quant, te_quant, seconds, peak_vram_gb}` or `{"error": ...}` |

2K sizes from the model card: 2048x2048, 2400x1792, 1792x2400, 2528x1696, 1696x2528, 2752x1536, 1536x2752.

## Notes

- One call at a time: each call loads the model (~10 s), generates and exits; two in parallel run out of memory.
  A 1024² image at 40 steps takes minutes; `--resolution 512 --steps 10` is ~8× faster for checking how a prompt
  reads.
- `rgba` adds the model's RGBA phrasing to the prompt itself. Judge the alpha on a contrasting colour
  (`magick out.png -background magenta -flatten qa.png`), not the raw PNG.
- Out of memory: lower `--resolution` (edits use the most, the reference image doubles the token count) or
  `--quant nf4`.
- Text inside the image (signs, labels, UI captions) renders legibly.

## How it fits in 12 GB

In bf16 the pipeline needs ~40 GB VRAM (7B DiT 14 GB, Qwen3-VL 8B text encoder 17.5 GB, VAE 1.4 GB). `generate.py`
loads the DiT as int8 weight-only (torchao, bf16 compute) and the text encoder as NF4 (bitsandbytes), with
model-level CPU offload (one component on the GPU at a time). `quantize.py` does this once into
`models/qwen-image-2.1-int8` (~14.5 GB), which every call then loads directly.

Never quantize the DiT with bitsandbytes LLM.int8 (`load_in_8bit`): its kernel casts activations to fp16, which
overflows on this DiT and returns the same noise for every prompt. NF4 and torchao int8 are clean.

With more VRAM, the underlying script runs other precisions straight from the Hugging Face weights:
`image/.venv/bin/python image/generate.py "..." --model Qwen/Qwen-Image-2.1 --te-quant bf16 --dit-quant bf16 --offload none`
(`--help` lists the rest: `--offload sequential`, `--vae-tiling`, `--no-kv-cache`).

## Setup

`../setup.sh image`: uv venv (Python 3.12, torch 2.14 with CUDA 13, diffusers pinned from git because the 2.1
pipeline is not in a release yet), the ~33 GB download into the Hugging Face cache, and the one-time quantization.
Once `models/qwen-image-2.1-int8` exists, the bf16 download can be deleted from the cache
(`image/.venv/bin/hf cache rm model/Qwen/Qwen-Image-2.1`), unless you want the `--model Qwen/Qwen-Image-2.1` route above.
