#!/usr/bin/env python
"""Qwen-Image-2.1 inference (text-to-image, RGBA generation, image editing).

Tuned for a 12 GB consumer GPU: by default the DiT transformer is loaded in int8
weight-only via torchao (bf16 compute) and the Qwen3-VL text encoder in 4-bit NF4 via
bitsandbytes, and the pipeline uses model-level CPU offload, so only one component
lives on the GPU at a time.

Examples:
  # text-to-image (official blog prompt)
  python generate.py "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement"

  # transparent (RGBA) sticker
  python generate.py "This is an RGBA image with transparency. A cute cartoon dragon sticker. The image has alpha channel and the background is transparent." --out outputs/dragon.png

  # edit an existing image
  python generate.py "Change the background to a sunset beach" --image outputs/t2i.png --out outputs/edit.png

  # full-precision run (needs ~40 GB VRAM)
  python generate.py "..." --te-quant bf16 --dit-quant bf16 --offload none
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _drop_foreign_cudnn_from_ld_library_path() -> None:
    """Re-exec without other environments' NVIDIA lib dirs in LD_LIBRARY_PATH.

    A shell that exports another venv's cuDNN lib dir breaks this one: cuDNN loads its engine sub-libraries
    by soname at runtime, LD_LIBRARY_PATH wins over this venv's copies, two cuDNN builds get mixed and every
    convolution fails with CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED. The dynamic loader reads the variable at
    process start, so the process has to be restarted for the change to take effect. (bin/qwen-image already
    starts from a clean environment; this covers running the script directly.)
    """
    value = os.environ.get("LD_LIBRARY_PATH", "")
    if not value or os.environ.get("_QWEN_LD_FIXED"):
        return
    venv_lib = os.path.abspath(os.path.join(os.path.dirname(sys.executable), "..", "lib"))
    kept = [p for p in value.split(":") if p and not ("site-packages/nvidia" in p and not p.startswith(venv_lib))]
    if kept == [p for p in value.split(":") if p]:
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join(kept)
    os.environ["_QWEN_LD_FIXED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_drop_foreign_cudnn_from_ld_library_path()
# Less fragmentation when large activations (2K generation, VAE decode) come and go on a 12 GB card.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

MODEL_ID = "Qwen/Qwen-Image-2.1"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VAE_TILE, VAE_TILE_STRIDE = 768, 512  # --vae-tiling: tile size and step in pixels
ALPHA_FLOOR, ALPHA_CEIL = 6, 250  # decoded alpha at or below the floor becomes 0, at or above the ceiling 255


def local_quantized(dit_quant: str) -> str:
    """The pre-quantized folder quantize.py writes for this transformer precision."""
    return os.path.join(PROJECT_DIR, "models", f"qwen-image-2.1-{dit_quant}")


def log(*args, **kwargs) -> None:
    """Progress output goes to stderr so stdout can carry a machine-readable result (--json)."""
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt", help="Text prompt.")
    p.add_argument("--model", default=None,
                   help="Model repo id or local folder. Default: the pre-quantized folder quantize.py wrote for "
                        f"--dit-quant ({local_quantized('<dit-quant>')}) if it exists, else {MODEL_ID} quantized "
                        "at load time.")
    p.add_argument("--negative-prompt", default=None, help="Negative prompt (only used when --cfg > 1).")
    p.add_argument("--cfg", type=float, default=1.0,
                   help="true_cfg_scale. Qwen-Image-2.1 is designed to run without guidance (1.0). "
                        "Values > 1 double the compute per step.")
    p.add_argument("--image", action="append", default=None,
                   help="Condition image(s) for editing. Repeat the flag for multiple references (up to 10).")
    p.add_argument("--width", type=int, default=None, help="Output width (multiple of 32).")
    p.add_argument("--height", type=int, default=None, help="Output height (multiple of 32).")
    p.add_argument("--resolution", type=int, default=1024,
                   help="Target side length when --width/--height are not given (also used to resize condition "
                        "images). The blog examples use 2048; 1024 is ~4x faster and fits comfortably in 12 GB.")
    p.add_argument("--steps", type=int, default=40, help="Denoising steps (official default: 40).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="Output PNG path (default: outputs/<timestamp>.png).")
    p.add_argument("--te-quant", choices=["nf4", "int8", "bf16"], default="nf4",
                   help="Text encoder (Qwen3-VL 8B, 17.5 GB in bf16) quantization.")
    p.add_argument("--dit-quant", choices=["nf4", "int8", "bf16"], default="int8",
                   help="Transformer (7B DiT, 14.2 GB in bf16) quantization. int8 = torchao int8 weight-only "
                        "(~7.1 GB, bf16 compute, default). nf4 = bitsandbytes 4-bit (~4.5 GB, lower quality).")
    p.add_argument("--offload", choices=["model", "sequential", "none"], default="model",
                   help="CPU offload strategy. 'model' keeps one component on GPU at a time.")
    p.add_argument("--no-kv-cache", action="store_true",
                   help="Disable the prefix KV cache for text/condition tokens (slower, changes the sample).")
    p.add_argument("--vae-tiling", action="store_true",
                   help=f"Decode in {VAE_TILE} px VAE tiles (needed at 2K on 12 GB).")
    p.add_argument("--rgb", action="store_true",
                   help="Save RGB. The VAE always decodes an alpha channel, and on opaque images it is not 255 "
                        "everywhere (down to ~180 on a 2K painting).")
    p.add_argument("--json", action="store_true",
                   help="Print a JSON result (output path, size, seed, timing, peak VRAM) on stdout; logs go to stderr.")
    return p.parse_args(argv)


def clean_alpha(image: Image.Image) -> Image.Image:
    """Transparent to 0, opaque to 255: the decoded alpha leaves the background at 1-6 (a haze over bright
    backgrounds, with the VAE tile grid in it) and the subject at 250-254. A linear map keeps soft edges soft."""
    import numpy as np

    a = np.asarray(image.getchannel("A"), dtype=np.float32)
    a = np.clip((a - ALPHA_FLOOR) * 255 / (ALPHA_CEIL - ALPHA_FLOOR), 0, 255)
    image = image.copy()
    image.putalpha(Image.fromarray(np.round(a).astype(np.uint8)))
    return image


def make_tf_bnb_config(quant: str, dtype: torch.dtype):
    from transformers import BitsAndBytesConfig

    if quant == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    if quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def make_transformer_quant_config(quant: str, dtype: torch.dtype):
    """Quantization config for the DiT.

    int8 uses torchao int8 weight-only (int8 weights, bfloat16 compute). bitsandbytes' LLM.int8 is NOT
    used for the transformer: it casts activations to float16 inside the matmul, which overflows on this
    DiT and yields pure noise (verified: identical garbage for every prompt, while nf4 is clean).
    """
    if quant == "nf4":
        from diffusers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    if quant == "int8":
        from diffusers import TorchAoConfig
        from torchao.quantization import Int8WeightOnlyConfig

        return TorchAoConfig(Int8WeightOnlyConfig())
    return None


def patch_bnb_int8_module_moves() -> None:
    """Make bitsandbytes int8 layers survive whole-model .to() moves (needed for CPU offload).

    `Linear8bitLt` keeps its packed int8 matrix (`CB`) and row scales (`SCB`) as plain attributes on the
    weight parameter and on `self.state`. bitsandbytes only relocates them in its own `Linear8bitLt.to()`
    override, but PyTorch moves a parent model through `Module._apply`, which never calls child `.to()`
    overrides. The result is that `.to("cpu")` on the transformer leaves the full 7 GB of int8 weights on
    the GPU, and the next move back allocates a second copy. This hooks `_apply` so the side tensors follow
    the parameter (`CB` is by definition the int8 weight data, so it aliases it instead of copying).
    """
    import bitsandbytes as bnb

    cls = bnb.nn.Linear8bitLt
    if getattr(cls, "_offload_patched", False):
        return
    orig_apply = cls._apply

    def _apply(self, fn, recurse=True):
        result = orig_apply(self, fn, recurse)
        w = self.weight
        if w.data.dtype == torch.int8:
            if getattr(w, "CB", None) is not None:
                w.CB = w.data
            if self.state.CB is not None:
                self.state.CB = w.data
        if getattr(w, "SCB", None) is not None:
            w.SCB = fn(w.SCB)
        if self.state.SCB is not None:
            self.state.SCB = fn(self.state.SCB)
        return result

    cls._apply = _apply
    cls._offload_patched = True


def is_prequantized(model_path: str) -> bool:
    """True if the folder was written by quantize.py (its component configs carry quantization_config)."""
    cfg = os.path.join(model_path, "transformer", "config.json")
    if not os.path.isfile(cfg):
        return False
    with open(cfg) as f:
        return "quantization_config" in f.read()


def load_pipeline(args: argparse.Namespace):
    from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
    from transformers import Qwen3VLForConditionalGeneration

    dtype = torch.bfloat16
    t0 = time.time()
    patch_bnb_int8_module_moves()  # harmless when no int8 layers are present

    local = local_quantized(args.dit_quant)
    model = args.model or (local if os.path.isdir(local) else MODEL_ID)

    if os.path.isdir(model) and is_prequantized(model):
        # Pre-quantized folder: the bitsandbytes config is stored with each component. bitsandbytes places
        # quantized weights on the GPU as they load, so load one component at a time and park it on the CPU
        # before the next one, otherwise text encoder + transformer land on the GPU together and OOM on 12 GB.
        log(f"[load] pre-quantized text_encoder from {model} ...")
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(model, subfolder="text_encoder", dtype=dtype)
        if args.offload != "none":
            text_encoder.to("cpu")
            torch.cuda.empty_cache()
        log("[load] pre-quantized transformer ...")
        transformer = QwenImage21Transformer2DModel.from_pretrained(
            model, subfolder="transformer", torch_dtype=dtype, device_map="cuda"
        )
        if args.offload != "none":
            transformer.to("cpu")
            torch.cuda.empty_cache()
        log("[load] pipeline (vae, scheduler, processor) ...")
        pipe = QwenImage21Pipeline.from_pretrained(
            model, torch_dtype=dtype, text_encoder=text_encoder, transformer=transformer
        )
        return finish_pipeline(pipe, args, t0)

    components = {}
    te_cfg = make_tf_bnb_config(args.te_quant, dtype)
    if te_cfg is not None:
        log(f"[load] text_encoder ({args.te_quant}) from {model} ...")
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            model, subfolder="text_encoder", quantization_config=te_cfg, dtype=dtype
        )
        if args.offload != "none":
            # Park it on the CPU so the transformer can be quantized on the GPU next.
            text_encoder.to("cpu")
            torch.cuda.empty_cache()
        components["text_encoder"] = text_encoder

    dit_cfg = make_transformer_quant_config(args.dit_quant, dtype)
    if dit_cfg is not None:
        log(f"[load] transformer ({args.dit_quant}) from {model} ...")
        # Quantize on the GPU: torchao needs an explicit device_map, and bitsandbytes lands there anyway.
        # The text encoder is already parked on the CPU at this point, so the GPU is free.
        transformer = QwenImage21Transformer2DModel.from_pretrained(
            model, subfolder="transformer", quantization_config=dit_cfg, torch_dtype=dtype, device_map="cuda"
        )
        if args.offload != "none":
            transformer.to("cpu")
            torch.cuda.empty_cache()
        components["transformer"] = transformer

    log("[load] pipeline (vae, scheduler, processor" + (", remaining bf16 components" if len(components) < 2 else "") + ") ...")
    pipe = QwenImage21Pipeline.from_pretrained(model, torch_dtype=dtype, **components)
    return finish_pipeline(pipe, args, t0)


def finish_pipeline(pipe, args: argparse.Namespace, t0: float):
    if args.vae_tiling and hasattr(pipe.vae, "enable_tiling"):
        # The default 256 px tiles blended over 64 px leave a stripe every 192 px on smooth gradients (skies);
        # 768 px tiles blended over 256 px show none and decode 2752x1536 in ~4.4 GB (untiled needs more than 12 GB).
        pipe.vae.enable_tiling(tile_sample_min_height=VAE_TILE, tile_sample_min_width=VAE_TILE,
                               tile_sample_stride_height=VAE_TILE_STRIDE, tile_sample_stride_width=VAE_TILE_STRIDE)

    if args.offload == "model":
        pipe.enable_model_cpu_offload()
    elif args.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")

    log(f"[load] done in {time.time() - t0:.1f}s")
    return pipe


def run(args: argparse.Namespace) -> dict:
    """Generate one image according to `args`. Returns a metadata dict; raises on failure."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    out_path = args.out or os.path.join(PROJECT_DIR, "outputs", time.strftime("%Y%m%d-%H%M%S") + ".png")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    condition_images = None
    if args.image:
        condition_images = [Image.open(p) for p in args.image]
        log(f"[input] {len(condition_images)} condition image(s): {[im.size for im in condition_images]}")

    pipe = load_pipeline(args)

    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator("cuda").manual_seed(args.seed)

    call_kwargs = dict(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        generator=generator,
        output_resolution=args.resolution,
        use_kv_cache=not args.no_kv_cache,
    )
    if args.width and args.height:
        call_kwargs.update(width=args.width, height=args.height)
    if condition_images is not None:
        call_kwargs["image"] = condition_images
    if args.cfg > 1.0:
        call_kwargs.update(true_cfg_scale=args.cfg, negative_prompt=args.negative_prompt or " ")

    log(f"[run] steps={args.steps} seed={args.seed} cfg={args.cfg} "
        f"size={call_kwargs.get('width', '?')}x{call_kwargs.get('height', '?')} (resolution={args.resolution})")
    t0 = time.time()
    image = pipe(**call_kwargs).images[0]
    elapsed = time.time() - t0

    if args.rgb:
        image = image.convert("RGB")
    elif image.mode == "RGBA":
        image = clean_alpha(image)
    image.save(out_path)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    log(f"[done] {out_path}  mode={image.mode} size={image.size}  {elapsed:.1f}s  peak VRAM {peak_gb:.2f} GB")
    return {
        "output": out_path,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "prompt": args.prompt,
        "seed": args.seed,
        "steps": args.steps,
        "cfg": args.cfg,
        "inputs": [os.path.abspath(p) for p in (args.image or [])],
        "dit_quant": args.dit_quant,
        "te_quant": args.te_quant,
        "seconds": round(elapsed, 1),
        "peak_vram_gb": round(peak_gb, 2),
    }


def main(argv: list[str] | None = None) -> int:
    import json

    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as e:  # noqa: BLE001 - report any failure uniformly for callers
        if args.json:
            print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        else:
            log(f"error: {type(e).__name__}: {e}")
        return 1
    if args.json:
        print(json.dumps(result))
    else:
        print(result["output"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
