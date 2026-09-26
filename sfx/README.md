# stable-audio — sound effects

Sound effects and ambience loops with [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) `small-sfx`
([model](https://huggingface.co/stabilityai/stable-audio-3-small-sfx)): 433M parameters, 44.1 kHz stereo, 8 diffusion
steps, up to 120 s. Stability AI Community License (free commercial use under $1M annual revenue; you own the
outputs). Fast: model load ~10 s, then well under a second per effect on the GPU.

```bash
stable-audio generate "two steel swords clashing hard, a sharp metallic impact with a short bright ring, close mic, dry room" -o clash.wav
stable-audio generate "video game coin pickup, bright two-note chime" --count 4 -o coin.ogg     # coin_1..4.ogg
stable-audio generate "steady rain on a forest canopy, distant thunder" --loop --duration 20 -o rain.ogg
stable-audio generate "heavy footsteps on a wooden floor" --mono -o steps.wav                     # mono for 3D audio
stable-audio generate "a single heavy footstep on dirt" --count 3 --split --mono -o step.wav      # step_1..N, one step each
echo "glass bottle shattering on stone" | stable-audio generate - -o glass.wav
stable-audio info
```

| Option | Default | |
|---|---|---|
| `-o PATH` | `sfx/outputs/<timestamp>.wav` | format from the extension: `.wav` (16-bit), `.ogg`, `.flac`, `.mp3` |
| `--duration S` | 3 | one-shots are trimmed to the sound; loops are exactly this long |
| `--count N` | 1 | variations `_1.._N`, seeds `seed..seed+N-1` |
| `--loop` | | seamless loop: generates duration + crossfade, equal-power crossfades the overhang into the start |
| `--split` | | one file per sound event: each take is cut where a new sound rises out of the last one's decay (onsets of the 10 ms envelope, 12 dB below the take's loudest moment, re-armed 24 dB below; rises within 0.2 s are one sound, such as a heel and toe), files `_1.._N` over all takes |
| `--mono`, `--no-trim`, `--no-normalize` | | output is peak-normalised to -1 dBFS by default |
| `--seed N`, `--steps N`, `--cfg F`, `--negative TEXT` | random, 8, 1.0 | |
| `--device cuda\|cpu` | cuda | CPU works, ~20× slower |
| `--json` | | `{output, outputs: [{output, seed, length_s}], prompt, duration, loop, sample_rate, channels, steps, cfg, device, seconds, peak_vram_gb}` (`--split`: each output also has `start_s` in its take) or `{"error": ...}` |

## Prompting

- `TrackType: SFX, ` is prepended unless the prompt starts with `TrackType:`.
- Describe the source, the action and the recording, library-style (the model was trained on Freesound /
  AudioSparx metadata): "a blunt, powerful thud made by slamming a wooden desk drawer shut, heavy low-mid body".
  Terse prompts ("sword clash") give several events in a row.
- A take often holds 2-5 events even for "a single ...": steps, knocks, chops. `--split` makes each its own file,
  ready for randomised playback. It cuts by loudness, so it also keeps clicks and scrapes as events: look at the
  results and drop those.
- Below ~3 s the model outputs noise, so the tool always generates at least 3 s and trims to the sound (cut 55 dB
  below the peak, 30 ms fade-out); the model puts the sound at the start and lets it decay.
- You can't hear the result: look at it (`ffmpeg -i x.wav -lavfi showspectrumpic=s=800x300 x.png`). A real sound has
  structure (transient, harmonics, decay); a broken take is a uniform block of noise.

## Setup

`../setup.sh sfx`: uv venv (Python 3.12, torch 2.7.1 cu126, `stable-audio-3` pinned from git). The weights (~1.5 GB,
including the T5Gemma text encoder) are gated: accept the licence on the model page with the Hugging Face account
whose token is on the machine; the first run downloads them. The "flash_attn not installed" notices are harmless.

Hardware: torch 2.7.1 cu126 covers GPUs before Blackwell; Blackwell (sm_120) needs its cu128 build. Without a
usable GPU, `--device cpu` works.
