"""Execute the notebook cells that this project wrote, in a fresh interpreter, with Kaggle's
runtime variables faked. Proves the unpack-and-import path and the solver swap actually run."""
import json
import sys
import tempfile
import time
import types
from pathlib import Path

nb = json.load(open(sys.argv[1], encoding="utf-8"))
mode = sys.argv[2]                      # "submission" | "interactive"
tmp = Path(tempfile.mkdtemp(prefix="nbexec_"))


def find(marker: str) -> str:
    return next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"]))


ns: dict = {"Path": Path, "sys": sys, "time": time}

# --- cell 4b: unpack + import (only the destination path is redirected into a temp dir)
src = find("ARCDUCK2_ZIP_B64").replace("/kaggle/arcduck2_src", (tmp / "arcduck2_src").as_posix())
exec(compile(src, "<cell 4b>", "exec"), ns)
loaded_from = Path(ns["arcduck2"].__file__).resolve()
assert str(tmp.resolve()) in str(loaded_from), f"arcduck2 was NOT loaded from the unpacked copy: {loaded_from}"
print("unpack/import OK ->", loaded_from)

# --- customization cell with a fake benchmark
ns.update(
    TRUE_SUBMISSION=(mode == "submission"),
    NOTEBOOK_START_EPOCH=time.time() - 600,
    bm=types.SimpleNamespace(n_passes=None, game_weights=None, solver=None, games=[]),
    target=types.SimpleNamespace(max_runtime_s=32400.0),
)
exec(compile(find("bm.solver = DuckV2Solver("), "<custom cell>", "exec"), ns)
solver = ns["bm"].solver
remaining = solver.hard_deadline_epoch - time.time()
print(f"[{mode}] solver={solver.label} deadline_in={remaining / 60:.0f}min per_game_cap={solver.max_runtime_s_per_game / 60:.0f}min "
      f"excluded={len(ns['demo_excluded_games'])}")
if mode == "submission":
    assert 8.5 * 3600 < remaining < 8.75 * 3600, remaining       # 9h minus margin minus the 10 min already elapsed
else:
    assert remaining <= 90 * 60 + 5 and len(ns["demo_excluded_games"]) == 15
assert ns["bm"].n_passes == 1 and ns["bm"].game_weights is None
solver.setup()
assert solver._cfg.slots == 10 and solver._cfg.max_runtime_s_per_game >= 60
solver.teardown()
print("OK")
