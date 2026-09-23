# qwen-tts — voice lines

Game voice lines with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache 2.0), two 1.7B models:
`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` speaks in a voice described in words (`design`),
`Qwen/Qwen3-TTS-12Hz-1.7B-Base` in the voice of a reference clip (`clone`). 24 kHz mono; English, Chinese, Japanese,
Korean, German, French, Russian, Portuguese, Spanish, Italian. ~4.6 GB VRAM, about real time.

```bash
# one voice per character: design a reference line, clone it for every other line
qwen-tts design "Aye, that blade'll hold. Forged it meself." \
  --voice "Male, around 60, gruff old dwarf blacksmith, deep gravelly bass, thick Scottish accent, slow and proud" -o voices/dwarf.wav
qwen-tts clone "Mind the anvil, lad. It bites." --ref voices/dwarf.wav \
  --ref-text "Aye, that blade'll hold. Forged it meself." -o lines/dwarf_02.ogg
qwen-tts batch lines.json          # many lines, each model loaded once
qwen-tts info
```

Options for `design` / `clone`: `-o PATH` (default `voice/outputs/<timestamp>.wav`; `.wav` 16-bit, `.ogg`, `.flac`,
`.mp3`), `--language` (`Auto` or one of the above), `--seed N` (random), `--no-trim` (default: silence 50 dB below the
peak trimmed to 50 ms), `--json`. `TEXT` of `-` reads stdin. `clone` without `--ref-text` falls back to
speaker-embedding-only cloning (lower fidelity).

`batch` takes a JSON list of `{text, out, voice}` (design) or `{text, out, ref [, ref_text]}` (clone), each with
optional `language` / `seed`; paths resolve against the current directory (`--relative-to-manifest`: the manifest's
folder). Design items run first, so a clone item may use a `ref` that a design item in the same file produces (its
text becomes the `ref_text`):

```json
[
  {"out": "voices/witch.wav", "text": "Come closer, dearie. The cauldron doesn't bite... much.",
   "voice": "Female, old witch, raspy crackling voice, sly and mischievous"},
  {"out": "lines/witch_1.ogg", "text": "Eye of newt? You call this eye of newt?", "ref": "voices/witch.wav"},
  {"out": "lines/guard_de.ogg", "text": "Halt! Wer da?", "voice": "Male, 30s, stern city guard, loud", "language": "German"}
]
```

`--json`: `{output, text, mode, language, seed, audio_seconds, seconds, voice | ref + ref_text, seconds_total,
peak_vram_gb}` (batch: `{outputs: [...], seconds_total, peak_vram_gb}`) or `{"error": ...}`.

## What to expect

- `design` gives a different voice on every call, even with the same description. Always design once and `clone`
  for a character with more than one line (clones stay close to the reference; designed lines do not).
- Emotion comes from the `--voice` description (`design`) or from the reference clip (`clone`, which takes no
  instruction). A character with several emotions needs one reference per emotion, or a cloud TTS with emotion tags.
- Intelligibility is high; short shouted barks ("Argh! I'm hit!") are the weakest case, check them.

## Setup

`../setup.sh voice`: uv venv (Python 3.12, torch 2.14, `qwen-tts` pinned from git). The first run of each mode
downloads its model (~4.5 GB each) into the Hugging Face cache. No flash-attention; PyTorch SDPA is used.
