#!/usr/bin/env bash
# lipsync: MediaPipe venv + Face Landmarker model, Rhubarb Lip Sync. Uses the Blender from ../setup.sh blender.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME PYTHONUSERBASE VIRTUAL_ENV
export PYTHONNOUSERSITE=1

[[ -x .venv/bin/python ]] || uv venv -q --python 3.12 .venv   # mediapipe does not install into Blender's Python
uv pip install -q --python .venv/bin/python mediapipe==1.0.1
mkdir -p models
[[ -f models/face_landmarker.task ]] || curl -fsSL -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
if [[ ! -x rhubarb/rhubarb ]]; then
  tmp=$(mktemp -d)
  curl -fsSL -o "$tmp/rhubarb.zip" \
    https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.14.0/Rhubarb-Lip-Sync-1.14.0-Linux.zip
  unzip -q "$tmp/rhubarb.zip" -d "$tmp" && mv "$tmp/Rhubarb-Lip-Sync-1.14.0-Linux" rhubarb && rm -rf "$tmp"
fi
echo "lipsync: ok (mediapipe $(.venv/bin/python -c 'import mediapipe; print(mediapipe.__version__)'), $(rhubarb/rhubarb --version | tail -1))"
