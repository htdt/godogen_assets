# godogen_assets — maintaining this repo

The local asset generators for godogen: one part per folder (code, `setup.sh`, `README.md`), one launcher per command
in `bin/`. Using the tools: `README.md`.

- Commit only what reproduces a working setup: code, launchers, setup scripts, docs. Environments, upstream
  checkouts, weights, outputs, logs, test inputs and experiments stay out (`.gitignore`); explore elsewhere and
  bring back only the result.
- Setup scripts pin what was tested (upstream commits, wheels, package versions) and stay idempotent. Nothing
  machine-specific in them or in the docs: no absolute paths, no GPU architecture, no one-PC workarounds or
  benchmark tables.
- Every command keeps the contract: stdout = output path(s) or one JSON object per result with `--json`, progress on
  stderr, exit 0 / 1, and a launcher that runs its own environment with a cleaned caller environment.
- A changed interface updates the part's README and, if it changes the pipeline, the top-level README in the same
  commit. Keep docs to what a user or agent needs to run the tool; describe the current state, not its history.
