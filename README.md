# arcduck2

A from-scratch agent layer for the ARC Prize 2026 ARC-AGI-3 Kaggle competition. It plugs into the
Tufa Labs TAAF framework and reuses Daniel Franzen's model-serving cells unchanged.

## Layout
- `src/arcduck2/` - the agent: `session` (game bridge and guards), `sandbox` (the python tool),
  `agent` (per-game loop), `scheduler` (slot priority), `solver` (TAAF Solver), `llm`, `perception`, `prompts`, `config`.
- `tests/` - 67 tests (fake game, scripted model, local HTTP server).
- `scripts/` - notebook builder and checkers, plus smoke tests against the real engine.
- `dist/` - the built notebook and `kernel-metadata.json`.

## Setup (Windows)
```
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe arc_agi arcengine numpy scipy requests pillow pytest pytest-asyncio pyflakes imageio matplotlib
uv pip install --python .venv/Scripts/python.exe -e <path-to>/tufa-arc-agi-framework --no-deps
uv pip install --python .venv/Scripts/python.exe -e . --no-deps
```
`tufa-arc-agi-framework` comes from https://github.com/tonghuikang/daniel-franzen-arc-agi-3.

## Test
```
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe scripts/smoke_online.py ls20-9607627b,ft09-0d8bbf25 60      # real games via ARC's public API, fake model
.venv/Scripts/python.exe scripts/smoke_benchmark.py deadline                        # through TAAF's Benchmark
```

## Build the notebook
```
.venv/Scripts/python.exe scripts/build_notebook.py <franzen-notebook.ipynb> dist
.venv/Scripts/python.exe scripts/check_notebook.py dist/arc-agi-3-arcduck2.ipynb
.venv/Scripts/python.exe scripts/exec_notebook_cells.py dist/arc-agi-3-arcduck2.ipynb submission
```
The Franzen notebook is `kaggle/dfranzen/arc-agi-3-milestone-2-solution.ipynb` in the repo above.

## Not yet verified
Nothing here has run on Kaggle's GPU with the real model. Do an interactive "Save & Run" first
(10 games, 90 min by default), read the `[arcduck2 ...]` status lines and `arcd_trace/*.jsonl`,
and only then spend a scored submission.
