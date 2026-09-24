#!/usr/bin/env bash
# setup.sh [part ...]: build the local environments and download the weights. No arguments = every part.
# Idempotent: finished steps are skipped, so re-run it after a failure.
#
#   parts: blender image mesh rig motion lipsync sfx voice
#          link   symlink bin/* into ~/.local/bin (or put bin/ on PATH yourself)
set -euo pipefail
root="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$root"
BLENDER_VERSION=4.5.14   # LTS; used headless by mia-rig, lipsync and tools/

setup_blender() {
  [[ -x deps/blender/blender ]] && return
  mkdir -p deps
  curl -fsSL "https://download.blender.org/release/Blender${BLENDER_VERSION%.*}/blender-$BLENDER_VERSION-linux-x64.tar.xz" \
    | tar xJ -C deps
  mv "deps/blender-$BLENDER_VERSION-linux-x64" deps/blender
}

link_bin() {
  mkdir -p ~/.local/bin
  for f in bin/*; do ln -sfn "$root/$f" ~/.local/bin/; done
  ls -l ~/.local/bin | grep -F "$root/bin"
}

for tool in git curl nvidia-smi; do
  command -v $tool >/dev/null || { echo "setup: $tool not found (README.md, Setup)" >&2; exit 1; }
done
parts=("$@")
[[ ${#parts[@]} -gt 0 ]] || parts=(blender image mesh rig motion lipsync sfx voice)
for part in "${parts[@]}"; do
  echo "== $part"
  case $part in
    blender) setup_blender ;;
    link) link_bin ;;
    image|mesh|rig|motion|lipsync|sfx|voice) bash "$part/setup.sh" ;;
    *) echo "setup: unknown part '$part'" >&2; exit 1 ;;
  esac
done
