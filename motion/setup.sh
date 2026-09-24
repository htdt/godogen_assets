#!/usr/bin/env bash
# Kimodo motion stack, installed into motion/:
#   kimodo/            upstream NVIDIA Kimodo (pinned), editable-installed into kimenv/
#   kimenv/            Python 3.12 venv (torch 2.6.0 cu124); run tools as kimenv/bin/python -P ..., never the console
#                      scripts (-P: from this folder, the kimodo/ checkout would shadow the installed package)
#   kimodo-practical/  the generation lib (pinned): kimogen.py (prompts, constraints, gated best-of-N) + bake
#   text_encoders/     Llama-3 encoder mirror: symlinks into the HF cache (~16 GB download), no gated access needed
# Then bakes add-moves' default move set, basic.json -> basic/ (gen-moves, incremental: a no-op when unchanged).
# Needs git and cmake-capable build tools.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
unset LD_LIBRARY_PATH PYTHONPATH VIRTUAL_ENV
KIMODO_COMMIT=6bb58488037dd65360ff0c5d1692b403a23309f7
KP_COMMIT=036f0fdb5f01b792ff18b8fd2f85ad7e9856341b

[[ -d kimodo/.git ]] || git clone -q https://github.com/nv-tlabs/kimodo
git -C kimodo checkout -q $KIMODO_COMMIT
[[ -d kimodo-practical/.git ]] || git clone -q https://github.com/htdt/kimodo-practical
git -C kimodo-practical checkout -q $KP_COMMIT

[[ -x kimenv/bin/python ]] || uv venv -q --seed --python 3.12 kimenv
kimenv/bin/pip install -q cmake ninja hatchling scikit-build-core
kimenv/bin/pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# --no-build-isolation so the MotionCorrection extension build sees the venv's cmake
kimenv/bin/pip show -q kimodo >/dev/null 2>&1 \
  || (cd kimodo && PATH=$PWD/../kimenv/bin:$PATH ../kimenv/bin/pip install -q --no-build-isolation -e ".[all]")

# The official base (meta-llama/Meta-Llama-3-8B-Instruct) is gated; this assembles the same layout from the public
# byte-identical mirror + the McGill-NLP LLM2Vec adapters. It writes kimodo-practical/text_encoders.
if [[ ! -e text_encoders ]]; then
  kimenv/bin/python kimodo-practical/kimodo/setup_text_encoder.py
  ln -s kimodo-practical/text_encoders text_encoders
fi
echo "motion: ok ($(kimenv/bin/python -P -c 'import kimodo, torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'))"

../bin/gen-moves basic.json -o basic >/dev/null
