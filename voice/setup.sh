#!/usr/bin/env bash
# Qwen3-TTS 1.7B: venv. The first run of each mode downloads its model (~4.5 GB each) into the Hugging Face cache.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
unset LD_LIBRARY_PATH PYTHONPATH VIRTUAL_ENV

[[ -x .venv/bin/python ]] || uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt
echo "voice: ok ($(.venv/bin/python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'))"
