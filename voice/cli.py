#!/usr/bin/env python
"""qwen-tts: one-command speech with Qwen3-TTS 1.7B (voice design + voice clone).

    qwen-tts design "Aye, that blade'll hold." --voice "gruff old dwarf, deep gravelly voice, Scottish accent" -o dwarf_ref.wav
    qwen-tts clone "Mind the anvil, lad." --ref dwarf_ref.wav --ref-text "Aye, that blade'll hold." -o dwarf_02.wav
    qwen-tts batch lines.json                      # many lines, each model loaded once
    qwen-tts info                                  # environment / model check

`design` voices a line from a text description of the speaker; the same description gives a
noticeably different voice on each call. For a character with several lines, design one reference
line and `clone` it for the rest (same voice every time).

Prints the output path (or a JSON object with --json) on stdout, progress on stderr, exits 0 on
success and 1 on failure. TEXT of "-" reads stdin. Output format follows the extension:
.wav (16-bit), .ogg (Vorbis), .flac, .mp3 — 24 kHz mono.

Batch manifest: a JSON list; each item has "text" and "out", plus either "voice" (design) or "ref"
(clone, with optional "ref_text"), and optional "language" / "seed". Design items run first, so a
clone item may use a ref that a design item in the same file produces (its text is filled in).
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
LANGUAGES = ["Auto", "English", "Chinese", "Japanese", "Korean", "German", "French", "Russian",
             "Portuguese", "Spanish", "Italian"]
TRIM_FLOOR_DB = -50.0


def log(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def read_text(text: str) -> str:
    if text == "-":
        text = sys.stdin.read()
    text = text.strip()
    if not text:
        raise SystemExit("empty text")
    return text


def resolve_out(out: str | None) -> str:
    if not out:
        out = os.path.join(HERE, "outputs", datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".wav")
    out = os.path.abspath(out)
    ext = os.path.splitext(out)[1].lower()
    if ext not in (".wav", ".ogg", ".flac", ".mp3"):
        raise SystemExit(f"unsupported output extension {ext!r}: use .wav, .ogg, .flac or .mp3")
    return out


def trim(wav, sr: int):
    """Cut leading/trailing audio quieter than TRIM_FLOOR_DB below the peak, keeping 50 ms of air."""
    import numpy as np

    env = np.abs(wav)
    idx = np.nonzero(env > env.max() * 10 ** (TRIM_FLOOR_DB / 20))[0]
    if len(idx) == 0:
        return wav
    pad = int(0.05 * sr)
    return wav[max(0, idx[0] - pad): min(len(wav), idx[-1] + pad)]


def save(path: str, wav, sr: int) -> None:
    import soundfile as sf

    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        sf.write(path, wav, sr, subtype="PCM_16")
    elif ext == ".ogg":
        sf.write(path, wav, sr, format="OGG", subtype="VORBIS")
    elif ext == ".mp3":
        sf.write(path, wav, sr, format="MP3", subtype="MPEG_LAYER_III")
    else:
        sf.write(path, wav, sr)


def check_language(lang: str) -> str:
    match = next((l for l in LANGUAGES if l.lower() == lang.lower()), None)
    if not match:
        raise SystemExit(f"unsupported language {lang!r}: use one of {', '.join(LANGUAGES)}")
    return match


class Engine:
    """Loads one Qwen3-TTS model at a time on the GPU and runs design / clone jobs."""

    def __init__(self, trim_silence: bool = True):
        import torch

        self.torch = torch
        self.trim = trim_silence
        self.name = None
        self.model: Any = None
        self.clone_prompts: dict = {}

    def load(self, repo: str):
        if self.name == repo:
            return
        from qwen_tts import Qwen3TTSModel

        if self.model is not None:
            del self.model
            self.clone_prompts.clear()
            gc.collect()
            self.torch.cuda.empty_cache()
        log(f"loading {repo}...")
        self.model = Qwen3TTSModel.from_pretrained(repo, device_map="cuda:0", dtype=self.torch.bfloat16,
                                                   attn_implementation="sdpa")
        self.name = repo

    def _finish(self, wavs, sr, job: dict, t0: float) -> dict:
        wav = wavs[0]
        if self.trim:
            wav = trim(wav, sr)
        save(job["out"], wav, sr)
        return {"output": job["out"], "text": job["text"], "mode": job["mode"], "language": job["language"],
                "seed": job["seed"], "audio_seconds": round(len(wav) / sr, 2), "seconds": round(time.time() - t0, 1),
                **({"voice": job["voice"]} if job["mode"] == "design" else
                   {"ref": job["ref"], "ref_text": job.get("ref_text")})}

    def design(self, job: dict) -> dict:
        self.load(DESIGN_MODEL)
        t0 = time.time()
        self.torch.manual_seed(job["seed"])
        log(f"design -> {job['out']}")
        wavs, sr = self.model.generate_voice_design(text=job["text"], instruct=job["voice"], language=job["language"])
        return self._finish(wavs, sr, job, t0)

    def clone(self, job: dict) -> dict:
        self.load(CLONE_MODEL)
        t0 = time.time()
        key = (job["ref"], job.get("ref_text"))
        if key not in self.clone_prompts:
            if not os.path.isfile(job["ref"]):
                raise SystemExit(f"reference audio not found: {job['ref']}")
            if not job.get("ref_text"):
                log("no --ref-text: cloning from the speaker embedding only (lower fidelity)")
            self.clone_prompts[key] = self.model.create_voice_clone_prompt(
                ref_audio=job["ref"], ref_text=job.get("ref_text"), x_vector_only_mode=not job.get("ref_text"))
        self.torch.manual_seed(job["seed"])
        log(f"clone -> {job['out']}")
        wavs, sr = self.model.generate_voice_clone(text=job["text"], language=job["language"],
                                                   voice_clone_prompt=self.clone_prompts[key])
        return self._finish(wavs, sr, job, t0)

    def peak_vram(self):
        return round(self.torch.cuda.max_memory_allocated() / 1e9, 2)


def make_job(item: dict, base_dir: str) -> dict:
    if not item.get("text") or not item.get("out"):
        raise SystemExit(f"batch item needs 'text' and 'out': {item}")
    if bool(item.get("voice")) == bool(item.get("ref")):
        raise SystemExit(f"batch item needs exactly one of 'voice' (design) or 'ref' (clone): {item}")
    job = {"text": item["text"].strip(), "out": resolve_out(os.path.join(base_dir, item["out"])),
           "language": check_language(item.get("language", "Auto")),
           "seed": item.get("seed", random.randint(0, 2**31 - 1))}
    if item.get("voice"):
        job.update(mode="design", voice=item["voice"])
    else:
        job.update(mode="clone", ref=os.path.abspath(os.path.join(base_dir, item["ref"])), ref_text=item.get("ref_text"))
    return job


def run(jobs: list[dict], trim_silence: bool) -> tuple[list[dict], float]:
    engine = Engine(trim_silence)
    if engine.torch.cuda.is_available():
        engine.torch.cuda.reset_peak_memory_stats()
    else:
        raise SystemExit("CUDA GPU required")
    results, designed = [], {}
    for job in [j for j in jobs if j["mode"] == "design"]:
        results.append(engine.design(job))
        designed[job["out"]] = job["text"]
    for job in [j for j in jobs if j["mode"] == "clone"]:
        if not job.get("ref_text") and job["ref"] in designed:
            job["ref_text"] = designed[job["ref"]]
        results.append(engine.clone(job))
    return results, engine.peak_vram()


def cmd_generate(ns: argparse.Namespace) -> int:
    t0 = time.time()
    if ns.command == "batch":
        with open(ns.manifest) as f:
            items = json.load(f)
        if not isinstance(items, list) or not items:
            raise SystemExit("manifest must be a non-empty JSON list")
        base = os.path.dirname(os.path.abspath(ns.manifest)) if ns.relative_to_manifest else os.getcwd()
        jobs = [make_job(it, base) for it in items]
    else:
        item = {"text": read_text(ns.text), "out": ns.out or resolve_out(None), "language": ns.language}
        if ns.seed is not None:
            item["seed"] = ns.seed
        if ns.command == "design":
            item["voice"] = ns.voice
        else:
            item.update(ref=ns.ref, ref_text=ns.ref_text)
        jobs = [make_job(item, os.getcwd())]

    with contextlib.redirect_stdout(sys.stderr):
        results, peak = run(jobs, not ns.no_trim)

    if ns.json:
        payload: dict = results[0] if ns.command != "batch" else {"outputs": results}
        payload.update(seconds_total=round(time.time() - t0, 1), peak_vram_gb=peak)
        print(json.dumps(payload))
    else:
        for r in results:
            print(r["output"])
    return 0


def cmd_info(ns: argparse.Namespace) -> int:
    info: dict = {"project": HERE, "python": sys.executable, "design_model": DESIGN_MODEL, "clone_model": CLONE_MODEL}
    with contextlib.redirect_stdout(sys.stderr):
        try:
            import torch
            from huggingface_hub import try_to_load_from_cache

            info.update(torch=torch.__version__, cuda_available=torch.cuda.is_available(),
                        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
            for key, repo in (("design_cached", DESIGN_MODEL), ("clone_cached", CLONE_MODEL)):
                info[key] = isinstance(try_to_load_from_cache(repo, "model.safetensors"), str)
            import qwen_tts  # noqa: F401
        except Exception as e:  # noqa: BLE001
            info["import_error"] = f"{type(e).__name__}: {e}"
    if ns.json:
        print(json.dumps(info, indent=2))
    else:
        for k, v in info.items():
            print(f"{k:16s} {v}")
    return 0 if "import_error" not in info and info.get("cuda_available") else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="qwen-tts", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--no-trim", action="store_true", help="Keep leading/trailing silence.")
        sp.add_argument("--json", action="store_true", help="Print a JSON result on stdout instead of the path.")

    def single(sp):
        sp.add_argument("text", help='Line to speak ("-" reads stdin).')
        sp.add_argument("-o", "--out", default=None, help="Output path, .wav/.ogg/.flac/.mp3 (default: outputs/<timestamp>.wav).")
        sp.add_argument("--language", default="Auto", help=f"One of: {', '.join(LANGUAGES)} (default Auto).")
        sp.add_argument("--seed", type=int, default=None, help="Seed (default random).")
        common(sp)

    d = sub.add_parser("design", help="Speak a line in a voice described in words.")
    single(d)
    d.add_argument("--voice", required=True,
                   help="Speaker description: gender, age, timbre, accent, emotion, pace, e.g. "
                        "'Male, 60s, gruff dwarf blacksmith, deep gravelly voice, Scottish accent, slow and proud'.")

    c = sub.add_parser("clone", help="Speak a line in the voice of a reference clip.")
    single(c)
    c.add_argument("--ref", required=True, help="Reference audio (e.g. a line made with `design`).")
    c.add_argument("--ref-text", default=None, help="Exact transcript of --ref (strongly recommended; without it "
                                                    "only the speaker embedding is used).")

    b = sub.add_parser("batch", help="Run a JSON manifest of design/clone lines.")
    b.add_argument("manifest", help="JSON list of {text, out, voice | ref [, ref_text, language, seed]}.")
    b.add_argument("--relative-to-manifest", action="store_true",
                   help="Resolve 'out' and 'ref' against the manifest's folder instead of the current directory.")
    common(b)

    i = sub.add_parser("info", help="Show environment and model status.")
    i.add_argument("--json", action="store_true")

    ns = p.parse_args(argv)
    try:
        return cmd_info(ns) if ns.command == "info" else cmd_generate(ns)
    except SystemExit as e:
        if isinstance(e.code, str):
            if getattr(ns, "json", False):
                print(json.dumps({"error": e.code}))
            log(f"qwen-tts: {e.code}")
            return 1
        raise
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        if getattr(ns, "json", False):
            print(json.dumps({"error": msg}))
        log(f"qwen-tts: {msg}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
