# Sourced by public launchers. Handles shell/preflight errors before exec hands off to the tool's own CLI.
asset_json=0
for asset_arg in "$@"; do [[ $asset_arg != --json ]] || asset_json=1; done
asset_exit() {
  local status=$?
  [[ $status -ne 0 ]] || return 0
  if [[ $asset_json = 1 ]]; then
    printf '{"error": "%s: command failed; see stderr"}\n' "$(basename "$0")"
  fi
  exit 1
}
trap asset_exit EXIT
