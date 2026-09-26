#!/usr/bin/env python
"""qwen-image: one-command Qwen-Image-2.1 generation and editing for scripts and other apps.

    qwen-image generate "a red apple on a wooden table" -o apple.png
    qwen-image rgba "a cute cartoon dragon sticker" -o dragon.png          # transparent PNG
    qwen-image edit -i apple.png "change the background to a sunset beach" -o beach.png
    qwen-image generate --json --size 2048x2048 --seed 7 "..."             # JSON result on stdout
    qwen-image info                                                        # environment / model check

Every subcommand prints the output path on stdout (or a JSON object with --json) and its progress on
stderr, exits 0 on success and 1 on failure. Read the prompt from stdin by passing "-".
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import generate  # noqa: E402  (also scrubs LD_LIBRARY_PATH and sets the CUDA allocator before importing torch)

RGBA_PREFIX = "This is an RGBA image with transparency. "
RGBA_SUFFIX = " The image has alpha channel and the background is transparent."


def add_common(p: argparse.ArgumentParser, editing: bool = False) -> None:
    p.add_argument("prompt", help='Text prompt ("-" reads it from stdin).')
    p.add_argument("-o", "--out", default=None, help="Output PNG path (default: outputs/<timestamp>.png in the project).")
    p.add_argument("--size", default=None, metavar="WxH",
                   help="Output size, e.g. 1024x1024 or 2752x1536 (multiples of 32)."
                        + (" Default: derived from the input image's aspect ratio at --resolution." if editing
                           else " Default: --resolution square."))
    p.add_argument("--resolution", type=int, default=1024,
                   help="Target side length when --size is not given (1024 default, 2048 for the blog-quality 2K).")
    p.add_argument("--steps", type=int, default=40, help="Denoising steps (default 40).")
    p.add_argument("--seed", type=int, default=None, help="Seed (default: random).")
    p.add_argument("--cfg", type=float, default=1.0, help="true_cfg_scale; >1 enables the negative prompt (2x compute).")
    p.add_argument("--negative", default=None, help="Negative prompt (with --cfg > 1).")
    p.add_argument("--quant", choices=["int8", "nf4"], default="int8",
                   help="Transformer precision: int8 (default, best quality) or nf4 (smaller, a bit faster).")
    p.add_argument("--json", action="store_true", help="Print a JSON result on stdout instead of the path.")


def parse_size(size: str | None) -> tuple[int, int] | None:
    if not size:
        return None
    try:
        w, h = size.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise SystemExit(f"--size must look like 1024x1024, got {size!r}")


def read_prompt(prompt: str) -> str:
    if prompt == "-":
        prompt = sys.stdin.read()
    prompt = prompt.strip()
    if not prompt:
        raise SystemExit("empty prompt")
    return prompt


def transparent(path: str) -> bool:
    """Whether an input image has transparent pixels (an edit of it keeps the alpha channel)."""
    from PIL import Image

    with Image.open(path) as im:
        if im.mode not in ("RGBA", "LA", "PA") and "transparency" not in im.info:
            return False
        return im.convert("RGBA").getchannel("A").getextrema()[0] < 128


def build_generate_argv(ns: argparse.Namespace, prompt: str, images: list[str] | None) -> list[str]:
    import random

    argv = [prompt, "--steps", str(ns.steps), "--resolution", str(ns.resolution),
            "--seed", str(ns.seed if ns.seed is not None else random.randint(0, 2**31 - 1)),
            "--dit-quant", ns.quant, "--cfg", str(ns.cfg)]
    if ns.command == "generate" or (ns.command == "edit" and not any(map(transparent, images))):
        argv.append("--rgb")
    size = parse_size(ns.size)
    if size:
        argv += ["--width", str(size[0]), "--height", str(size[1])]
    longest_side = max(size) if size else ns.resolution
    if longest_side >= 1536:
        argv.append("--vae-tiling")  # keeps the VAE decode inside 12 GB at 2K
    if ns.out:
        argv += ["--out", ns.out]
    if ns.negative:
        argv += ["--negative-prompt", ns.negative]
    for img in images or []:
        argv += ["--image", img]
    if ns.json:
        argv.append("--json")
    return argv


def cmd_info(ns: argparse.Namespace) -> int:
    import torch

    info = {
        "project": HERE,
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
        "prequantized_models": [p for p in map(generate.local_quantized, ("int8", "nf4")) if os.path.isdir(p)],
        "hf_model": generate.MODEL_ID,
    }
    try:
        import diffusers, transformers, bitsandbytes, torchao  # noqa: E401

        info.update(diffusers=diffusers.__version__, transformers=transformers.__version__,
                    bitsandbytes=bitsandbytes.__version__, torchao=torchao.__version__)
    except Exception as e:  # noqa: BLE001
        info["import_error"] = f"{type(e).__name__}: {e}"
    if ns.json:
        print(json.dumps(info, indent=2))
    else:
        for k, v in info.items():
            print(f"{k:20s} {v}")
    return 0 if info["cuda_available"] and "import_error" not in info else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="qwen-image", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Text-to-image.")
    add_common(g)

    r = sub.add_parser("rgba", help="Text-to-image with a transparent background (RGBA PNG).")
    add_common(r)

    e = sub.add_parser("edit", help="Edit / compose with one or more reference images.")
    e.add_argument("-i", "--image", action="append", required=True,
                   help="Reference image (repeat for up to 10; the last one sets the default aspect ratio).")
    add_common(e, editing=True)

    i = sub.add_parser("info", help="Show environment and model status.")
    i.add_argument("--json", action="store_true")

    ns = p.parse_args(argv)
    if ns.command == "info":
        return cmd_info(ns)

    prompt = read_prompt(ns.prompt)
    images = None
    if ns.command == "rgba":
        if "rgba" not in prompt.lower():
            prompt = RGBA_PREFIX + prompt + RGBA_SUFFIX
    elif ns.command == "edit":
        images = ns.image
        for img in images:
            if not os.path.isfile(img):
                raise SystemExit(f"input image not found: {img}")

    return generate.main(build_generate_argv(ns, prompt, images))


if __name__ == "__main__":
    sys.exit(main())
