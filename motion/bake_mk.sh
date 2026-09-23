#!/usr/bin/env bash
# bake_mk.sh [--only move,move]: generate kimodo-practical's validated 17-move MK set with Kimodo and bake it to
# kimodo-practical/kimodo/out/web_mk (the default move set of add-moves). GPU (~2.5 GB VRAM) + the Llama-3 text
# encoder as a local service on the CPU (~16 GB RAM), ~15 min. --only regenerates single moves, then re-bakes all.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
unset LD_LIBRARY_PATH PYTHONPATH VIRTUAL_ENV
only=""; [[ ${1:-} = --only ]] && only=$2
PY=$PWD/kimenv/bin/python
export TEXT_ENCODERS_DIR=$PWD/text_encoders TEXT_ENCODER_DEVICE=cpu TEXT_ENCODER_MODE=api

if ! curl -fs -o /dev/null http://127.0.0.1:9550/; then
  echo "starting the text encoder service (loads 16 GB, 1-3 min)" >&2
  GRADIO_SERVER_NAME=127.0.0.1 "$PY" -P -m kimodo.scripts.run_text_encoder_server > "${TMPDIR:-/tmp}/kimodo_encoder.log" 2>&1 &
  encoder=$!; trap 'kill $encoder 2>/dev/null' EXIT
  until curl -fs -o /dev/null http://127.0.0.1:9550/; do
    kill -0 $encoder 2>/dev/null || { tail -20 "${TMPDIR:-/tmp}/kimodo_encoder.log" >&2; exit 1; }
    sleep 5
  done
fi

cd kimodo-practical/kimodo
if [[ -n $only ]]; then
  "$PY" kimogen.py gen --spec moveset_mk.json --only "$only"
else
  if [[ ! -f out/stance_pose.json ]]; then
    "$PY" kimogen.py gen --spec moveset_mk.json --only idle_stance   # the idle's medoid frame becomes THE stance,
    "$PY" kimogen.py stance                                          # which bookends every other move
  fi
  "$PY" kimogen.py gen --spec moveset_mk.json
fi
"$PY" kimogen.py report
"$PY" bake_kimodo.py --spec moveset_mk.json --web out/web_mk
echo "baked: $PWD/out/web_mk"
