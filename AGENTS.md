# godogen_assets — maintaining this repo

The local asset generators for godogen: one part per folder (code, `setup.sh`, `README.md`), one launcher per command
in `bin/`. Using and setting up the tools: `README.md`, whose principles (top) every change follows.

- Commit only what reproduces a working setup: code, launchers, setup scripts, docs. Environments, upstream
  checkouts, weights, outputs, logs, test inputs and experiments stay out (`.gitignore`); explore elsewhere and
  bring back only the result.
- Setup scripts are the tested recipe for the reference machine (README, Setup): they pin what was tested there
  (upstream commits, wheels, package versions) and stay idempotent. Choices that depend on the hardware are written
  down in the part's README (Setup, "Hardware:") for the setup agent, not branched on in the scripts. Nothing else
  machine-specific in scripts or docs: no absolute paths, no one-PC workarounds, no benchmark tables.
- Every command keeps the contract: stdout = output path(s) or one JSON object per result with `--json`, progress on
  stderr, exit 0 / 1, and a launcher that runs its own environment with a cleaned caller environment.
- A part's README puts what using the tool needs first (options, outputs, limits), then `How it works` (for tuning
  or fixing the part), then `Setup`. A changed interface updates the part's README and, if it changes the pipeline,
  the top-level README in the same commit. Keep docs to what a user or agent needs; describe the current state, not
  its history.
