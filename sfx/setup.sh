#!/usr/bin/env bash
# Stable Audio 3 small-sfx: venv. The weights (~1.5 GB) are gated: accept the licence on
# https://huggingface.co/stabilityai/stable-audio-3-small-sfx with the account whose token is on this machine
# (`hf auth login`); the first run downloads them.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME PYTHONUSERBASE VIRTUAL_ENV
export PYTHONNOUSERSITE=1

[[ -x .venv/bin/python ]] || uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install -q --python .venv/bin/python -r requirements.txt
echo "sfx: ok ($(.venv/bin/python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'))"
