#!/usr/bin/env bash
# Make-It-Animatable: upstream checkout (pinned), Python 3.11 conda env (torch 2.1.2 cu121, bpy 4.3), weights,
# FBX2glTF, and the Mixamo skeleton templates. Idempotent. Needs micromamba, git and the Blender from ../setup.sh.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME PYTHONUSERBASE VIRTUAL_ENV
export PYTHONNOUSERSITE=1
E=$PWD/.conda
MIA_COMMIT=8fb51382ff6da556cdb95cc03a48200603f3a493

[[ -d Make-It-Animatable/.git ]] || git clone -q https://github.com/jasongzy/Make-It-Animatable.git
git -C Make-It-Animatable checkout -q $MIA_COMMIT
git -C Make-It-Animatable submodule update -q --init --recursive

[[ -x $E/bin/python ]] || micromamba create -q -y -p "$E" -c conda-forge python=3.11
# MIA's requirements.txt carries the torch / PyG / pytorch3d indexes; gradio is pinned to the tested version.
(cd Make-It-Animatable && "$E/bin/pip" install -q -r requirements.txt "gradio==6.28.0")

cd Make-It-Animatable
# v1 weights + data (incl. the bundled "Standard Run.fbx"), ~2.5 GB
[[ -d output/best/new ]] || "$E/bin/hf" download jasongzy/Make-It-Animatable \
  --include "output/best/new/*" --include "data/*" --local-dir .
if [[ ! -x util/FBX2glTF ]]; then
  curl -fsSL -o util/FBX2glTF https://github.com/facebookincubator/FBX2glTF/releases/download/v0.9.7/FBX2glTF-linux-x64
  chmod +x util/FBX2glTF
fi
# MIA's skeleton templates live in the gated jasongzy/Mixamo dataset; rebuild them from "Standard Run.fbx" instead
[[ -f data/Mixamo/bones.fbx ]] || "$E/bin/python" ../make_templates.py
echo "rig: ok ($("$E/bin/python" -c 'import torch, bpy; print("torch", torch.__version__, "bpy", bpy.app.version_string)' 2>/dev/null | tail -1))"
