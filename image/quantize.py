#!/usr/bin/env python
"""Quantize Qwen-Image-2.1 once with bitsandbytes and save a local pre-quantized pipeline.

Afterwards `generate.py --model <out dir>` (or the default local path) loads the quantized
weights directly, with no per-run quantization and roughly a third of the disk reads.

    .venv/bin/python quantize.py               # int8 transformer, nf4 text encoder -> image/models/qwen-image-2.1-int8
    .venv/bin/python quantize.py --dit-quant nf4 --out models/qwen-image-2.1-nf4
"""

from __future__ import annotations

import argparse
import os
import shutil
import time

import sys


def _drop_foreign_cudnn_from_ld_library_path() -> None:
    """Re-exec without other environments' NVIDIA lib dirs in LD_LIBRARY_PATH (see generate.py)."""
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

import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

MODEL_ID = "Qwen/Qwen-Image-2.1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=MODEL_ID, help="Source repo id or local diffusers folder.")
    p.add_argument("--out", default=None, help="Output folder (default: models/qwen-image-2.1-<dit-quant>).")
    p.add_argument("--te-quant", choices=["nf4", "int8"], default="nf4", help="Text encoder quantization.")
    p.add_argument("--dit-quant", choices=["nf4", "int8"], default="int8", help="Transformer quantization.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", f"qwen-image-2.1-{args.dit_quant}")
    os.makedirs(out, exist_ok=True)
    dtype = torch.bfloat16

    from diffusers import BitsAndBytesConfig as DiffusersBnB
    from diffusers import QwenImage21Transformer2DModel, TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import BitsAndBytesConfig as TransformersBnB
    from transformers import Qwen3VLForConditionalGeneration

    def tf_cfg(q):
        if q == "nf4":
            return TransformersBnB(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        return TransformersBnB(load_in_8bit=True)

    def df_cfg(q):
        if q == "nf4":
            return DiffusersBnB(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        # int8: torchao weight-only (bf16 compute). bitsandbytes LLM.int8 casts activations to float16 and
        # produces pure noise with this DiT, so it is deliberately not offered for the transformer.
        return TorchAoConfig(Int8WeightOnlyConfig())

    # 1. Text encoder
    t0 = time.time()
    print(f"[quantize] text_encoder -> {args.te_quant}", flush=True)
    te = Qwen3VLForConditionalGeneration.from_pretrained(
        args.source, subfolder="text_encoder", quantization_config=tf_cfg(args.te_quant), dtype=dtype
    )
    te.save_pretrained(os.path.join(out, "text_encoder"), safe_serialization=True)
    del te
    torch.cuda.empty_cache()
    print(f"[quantize] text_encoder saved ({time.time() - t0:.0f}s)", flush=True)

    # 2. Transformer
    t0 = time.time()
    print(f"[quantize] transformer -> {args.dit_quant}", flush=True)
    dit = QwenImage21Transformer2DModel.from_pretrained(
        args.source, subfolder="transformer", quantization_config=df_cfg(args.dit_quant), torch_dtype=dtype,
        device_map="cuda",
    )
    dit.to("cpu")  # saving a torchao model from the GPU needs more than 12 GB; from the CPU it takes ~3 s
    torch.cuda.empty_cache()
    dit.save_pretrained(os.path.join(out, "transformer"), safe_serialization=True)
    del dit
    torch.cuda.empty_cache()
    print(f"[quantize] transformer saved ({time.time() - t0:.0f}s)", flush=True)

    # 3. Copy the small unquantized parts (vae, scheduler, processor, model_index.json) as-is.
    src = args.source
    if not os.path.isdir(src):
        src = snapshot_download(args.source, allow_patterns=["model_index.json", "vae/*", "scheduler/*", "processor/*"])
    # Hugging Face cache blobs are read-only; copy contents without permissions and replace any previous copy.
    for name in ("vae", "scheduler", "processor"):
        shutil.rmtree(os.path.join(out, name), ignore_errors=True)
        shutil.copytree(os.path.join(src, name), os.path.join(out, name), copy_function=shutil.copyfile)
    shutil.copyfile(os.path.join(src, "model_index.json"), os.path.join(out, "model_index.json"))

    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out) for f in fs) / 1e9
    print(f"[done] {out}  ({total:.1f} GB on disk)")


if __name__ == "__main__":
    main()
