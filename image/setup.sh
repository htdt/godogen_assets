#!/usr/bin/env bash
# Qwen-Image-2.1: venv, weights (~33 GB into the Hugging Face cache), one-time quantization to models/ (~14.5 GB).
# Idempotent: finished steps are skipped. Afterwards the bf16 download can be removed from the HF cache.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
unset LD_LIBRARY_PATH PYTHONPATH VIRTUAL_ENV

[[ -x .venv/bin/python ]] || uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt

if [[ ! -f models/qwen-image-2.1-int8/model_index.json ]]; then
  .venv/bin/hf download Qwen/Qwen-Image-2.1 --exclude "assets/*"
  .venv/bin/python quantize.py      # int8 transformer (torchao) + nf4 text encoder
fi
echo "image: ok ($(.venv/bin/python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'))"
