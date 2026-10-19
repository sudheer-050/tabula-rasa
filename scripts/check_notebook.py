"""Static checks on the assembled notebook: syntax, undefined names, leftover harness settings.

    python scripts/check_notebook.py dist/arc-agi-3-arcduck2.ipynb

Cells run in one shared namespace, so the code cells are concatenated in order and linted as a
single module. Shell escapes (!cmd) and cell magics (%%...) are neutralised first.
"""
from __future__ import annotations

import json
import re
import sys

from pyflakes import api, reporter

# Names the Kaggle/IPython runtime provides without an import.
RUNTIME_NAMES = {"get_ipython", "display", "HTML", "target", "bm"}


def main(path: str) -> int:
    nb = json.load(open(path, encoding="utf-8"))
    chunks = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        src = re.sub(r"^%%.*$", "pass", src, flags=re.M)
        src = re.sub(r"^(\s*)!.*$", r"\1pass", src, flags=re.M)
        chunks.append(f"# ---- cell {i}\n{src}\n")
    module = "\n".join(chunks)

    problems: list[str] = []

    class Collect(reporter.Reporter):
        def __init__(self):
            super().__init__(sys.stdout, sys.stderr)

        def unexpectedError(self, filename, msg):
            problems.append(f"unexpected: {msg}")

        def syntaxError(self, filename, msg, lineno, offset, text):
            problems.append(f"syntax: {msg} (line {lineno})")

        def flake(self, message):
            text = message.message % message.message_args
            if "undefined name" in text:
                name = message.message_args[0]
                if name in RUNTIME_NAMES:
                    return
                problems.append(f"line {message.lineno}: {text}")

    api.check(module, "notebook", Collect())
    full = module
    for stale in ("ARC3_", "harness-changes.patch", "LOCAL_ANALYZER", "HarnessSolver"):
        if stale in full:
            problems.append(f"leftover harness reference: {stale}")
    if "DuckV2Solver(" not in full:
        problems.append("DuckV2Solver is never constructed")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print(f"notebook OK: {len(chunks)} code cells, no undefined names, no stale harness settings")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
