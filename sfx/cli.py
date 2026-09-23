#!/usr/bin/env python
"""stable-audio: one-command sound-effect generation with Stable Audio 3 small-sfx.

    stable-audio generate "two steel swords clashing, sharp metallic hit with a short ring" -o clash.wav
    stable-audio generate "coin pickup chime" --count 4 -o coin.wav          # coin_1.wav .. coin_4.wav
    stable-audio generate "steady rain on leaves" --loop --duration 20 -o rain.ogg
    stable-audio generate --json "..."                                         # JSON result on stdout
    stable-audio info                                                          # environment / model check

Prints the output path(s) on stdout (or a JSON object with --json), progress on stderr, exits 0 on
success and 1 on failure. Read the prompt from stdin by passing "-". Output format follows the
extension: .wav (16-bit), .ogg (Vorbis), .flac, .mp3.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "small-sfx"
HF_REPO = "stabilityai/stable-audio-3-small-sfx"
PREFIX = "TrackType: SFX, "
# Below ~3 s the model emits full-scale noise (0.8 s -> ~80% of samples clipped, 1.5 s -> ~15%, 3 s clean;
# same in fp16 and fp32). Short effects are generated at this length and trimmed to the sound.
MIN_GEN_S = 3.0
TRIM_FLOOR_DB = -55.0  # trailing/leading audio this far below the peak is cut
FADE_S = 0.03


def log(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def read_prompt(prompt: str) -> str:
    if prompt == "-":
        prompt = sys.stdin.read()
    prompt = " ".join(prompt.split())
    if not prompt:
        raise SystemExit("empty prompt")
    return prompt if prompt.lower().startswith("tracktype:") else PREFIX + prompt


def output_paths(out: str | None, count: int) -> list[str]:
    if not out:
        out = os.path.join(HERE, "outputs", datetime.now().strftime("%Y%m%d-%H%M%S") + ".wav")
    base, ext = os.path.splitext(os.path.abspath(out))
    ext = ext or ".wav"
    if ext.lower() not in (".wav", ".ogg", ".flac", ".mp3"):
        raise SystemExit(f"unsupported output extension {ext!r}: use .wav, .ogg, .flac or .mp3")
    if count == 1:
        return [base + ext]
    return [f"{base}_{i}{ext}" for i in range(1, count + 1)]


def trim(audio, sr: int):
    """Cut audio quieter than TRIM_FLOOR_DB below the peak at both ends, then fade out."""
    import torch

    env = audio.abs().amax(dim=0)
    idx = torch.nonzero(env > env.max() * 10 ** (TRIM_FLOOR_DB / 20)).flatten()
    if len(idx) == 0:
        return audio
    start = max(0, int(idx[0]) - int(0.005 * sr))
    audio = audio[:, start:int(idx[-1]) + 1].clone()
    n = min(int(FADE_S * sr), audio.shape[1])
    audio[:, -n:] *= torch.linspace(1, 0, n)
    return audio


def make_loop(audio, sr: int, length_s: float, xfade_s: float):
    """Equal-power crossfade of the audio past `length_s` into the start, so the clip loops seamlessly."""
    import torch

    n, x = int(length_s * sr), int(xfade_s * sr)
    out = audio[:, :n].clone()
    t = torch.linspace(0, 1, x)
    out[:, :x] = audio[:, :x] * torch.sqrt(t) + audio[:, n:n + x] * torch.sqrt(1 - t)
    return out


def save(path: str, audio, sr: int) -> None:
    import soundfile as sf

    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = audio.t().numpy()
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        sf.write(path, data, sr, subtype="PCM_16")
    elif ext == ".ogg":
        sf.write(path, data, sr, format="OGG", subtype="VORBIS")
    elif ext == ".mp3":
        sf.write(path, data, sr, format="MP3", subtype="MPEG_LAYER_III")
    else:
        sf.write(path, data, sr)


def cmd_generate(ns: argparse.Namespace) -> int:
    prompt = read_prompt(ns.prompt)
    if ns.count < 1:
        raise SystemExit("--count must be >= 1")
    if ns.loop and ns.duration < 2:
        raise SystemExit("--loop needs --duration >= 2")
    paths = output_paths(ns.out, ns.count)
    xfade = min(1.0, ns.duration / 4) if ns.loop else 0.0
    gen_s = max(ns.duration + xfade, MIN_GEN_S)
    if gen_s > 120:
        raise SystemExit("--duration is limited to 120 s by the model")
    seed0 = ns.seed if ns.seed is not None else random.randint(0, 2**31 - 1 - ns.count)

    t0 = time.time()
    with contextlib.redirect_stdout(sys.stderr):  # the library prints; keep stdout for the result
        import torch
        from stable_audio_3 import StableAudioModel

        log(f"loading {MODEL}...")
        model = StableAudioModel.from_pretrained(MODEL, device=ns.device)
        sr = model.model.sample_rate
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        results = []
        for i, path in enumerate(paths):
            seed = seed0 + i
            log(f"generating {path} (seed {seed}, {gen_s:.1f}s)...")
            audio = model.generate(prompt=prompt, negative_prompt=ns.negative, duration=gen_s,
                                   steps=ns.steps, cfg_scale=ns.cfg, seed=seed)
            audio = audio[0].float().cpu().clamp(-1, 1)
            if ns.loop:
                audio = make_loop(audio, sr, ns.duration, xfade)
            elif not ns.no_trim:
                audio = trim(audio, sr)
            if ns.mono:
                audio = audio.mean(dim=0, keepdim=True)
            if not ns.no_normalize:
                peak = float(audio.abs().max())
                if peak > 0:
                    audio = audio * (10 ** (-1 / 20) / peak)
            save(path, audio, sr)
            results.append({"output": path, "seed": seed, "length_s": round(audio.shape[1] / sr, 3)})
        peak_vram = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None
        device = str(model.device)

    if ns.json:
        print(json.dumps({"output": results[0]["output"], "outputs": results, "prompt": prompt,
                          "duration": ns.duration, "loop": ns.loop, "sample_rate": sr,
                          "channels": 1 if ns.mono else 2, "steps": ns.steps, "cfg": ns.cfg, "device": device,
                          "seconds": round(time.time() - t0, 1), "peak_vram_gb": peak_vram}))
    else:
        for r in results:
            print(r["output"])
    return 0


def cmd_info(ns: argparse.Namespace) -> int:
    info: dict = {"project": HERE, "python": sys.executable, "model": MODEL, "hf_repo": HF_REPO}
    with contextlib.redirect_stdout(sys.stderr):  # the library prints flash-attn notices on import
        try:
            import torch

            info.update(torch=torch.__version__, cuda_available=torch.cuda.is_available(),
                        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(HF_REPO, "model.safetensors")
            info["weights_cached"] = isinstance(cached, str)
            import stable_audio_3  # noqa: F401
        except Exception as e:  # noqa: BLE001
            info["import_error"] = f"{type(e).__name__}: {e}"
    if ns.json:
        print(json.dumps(info, indent=2))
    else:
        for k, v in info.items():
            print(f"{k:16s} {v}")
    return 0 if "import_error" not in info else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="stable-audio", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Text-to-sound-effect.")
    g.add_argument("prompt", help='Sound description ("-" reads stdin). "TrackType: SFX, " is prepended unless present.')
    g.add_argument("-o", "--out", default=None,
                   help="Output path, .wav/.ogg/.flac/.mp3 (default: outputs/<timestamp>.wav in the project).")
    g.add_argument("--duration", type=float, default=3.0,
                   help="Seconds to generate (default 3; values under 3 are raised to 3). One-shots are then "
                        "trimmed to the sound; with --loop this is the exact loop length.")
    g.add_argument("--count", type=int, default=1, help="Number of variations (files get _1.._N suffixes).")
    g.add_argument("--loop", action="store_true", help="Seamless loop of exactly --duration seconds (ambience).")
    g.add_argument("--mono", action="store_true", help="Downmix to mono (e.g. for 3D positional sounds).")
    g.add_argument("--no-trim", action="store_true", help="Keep the full generated length.")
    g.add_argument("--no-normalize", action="store_true", help="Skip peak normalisation to -1 dBFS.")
    g.add_argument("--seed", type=int, default=None, help="Seed (default random; variation i uses seed+i).")
    g.add_argument("--steps", type=int, default=8, help="Diffusion steps (default 8).")
    g.add_argument("--cfg", type=float, default=1.0, help="CFG scale (default 1.0).")
    g.add_argument("--negative", default=None, help="Negative prompt.")
    g.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available; CPU takes ~4 s per sound).")
    g.add_argument("--json", action="store_true", help="Print a JSON result on stdout instead of the path(s).")

    i = sub.add_parser("info", help="Show environment and model status.")
    i.add_argument("--json", action="store_true")

    ns = p.parse_args(argv)
    try:
        return cmd_info(ns) if ns.command == "info" else cmd_generate(ns)
    except SystemExit as e:
        if isinstance(e.code, str):
            if getattr(ns, "json", False):
                print(json.dumps({"error": e.code}))
            else:
                log(f"stable-audio: {e.code}")
            return 1
        raise
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        if getattr(ns, "json", False):
            print(json.dumps({"error": msg}))
        log(f"stable-audio: {msg}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
