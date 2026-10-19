"""Assemble the Kaggle notebook: Franzen's proven serving/launch cells + the arcduck2 solver.

    python scripts/build_notebook.py <path-to-franzen-notebook.ipynb> <out-dir>

Fails loudly if any anchor text it expects to replace is missing, so a changed upstream
notebook can never be silently half-patched.
"""
from __future__ import annotations

import base64
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Python 3.12 image that Franzen's offline wheels (cp312) were built for; read from the user's v2 notebook.
PINNED_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c"
SRC = ROOT / "src" / "arcduck2"


def zip_source_b64() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(SRC.glob("*.py")):
            zf.writestr(f"arcduck2/{path.name}", path.read_text(encoding="utf-8"))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def cell_text(cell: dict) -> str:
    return "".join(cell["source"])


def make_cell(kind: str, text: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def replace_between(text: str, start: str, end: str, new: str, include_end: bool = True) -> str:
    i = text.index(start)
    j = text.index(end, i) + (len(end) if include_end else 0)
    return text[:i] + new + text[j:]


ABOUT = """# ARC-AGI-3: arcduck2 agent layer

A from-scratch agent layer (`arcduck2`) running on the open Duck/Franzen serving stack.
Model serving, precaching and game launch are reused unchanged from the Milestone 2 notebook;
the agent loop, sandbox, prompts, scheduler and game bridge are new.

What changed in the agent layer:
* the model's reasoning is kept in history (removing it caused a large regression upstream);
* a deterministic-game memory refuses action sequences already known to end in GAME_OVER;
* repeat and no-op guards that start on level 2, so level 1 stays free for experimentation;
* `UNDO` is exposed; a model that never calls `RESET` after GAME_OVER gets one issued for it;
* context is trimmed in large blocks (keeps the server's prefix cache valid) and a memory brief
  with the model's own `notes`, known-fatal paths and the current board is rebuilt afterwards;
* a priority scheduler allocates the limited model slots by expected score gain;
* a hard deadline winds every game down before the notebook limit, so the run ends cleanly;
* a status line is printed every 10 minutes and a per-game trace is written to the working dir.

Keep all attached source, wheelhouse, and model inputs enabled and select the RTX PRO 6000.
"""

ENV_BLOCK = """setup_env = {
    ########## transformers config ##########
    'USE_TF': '0',
    'TRANSFORMERS_NO_TF': '1',
    'TRANSFORMERS_NO_TORCHVISION': '1',

    ########## arcduck2 agent layer ##########
    'ARCD_BASE_URL': SERVER_BASE_URL,
    'ARCD_MODEL': SERVED_MODEL_NAME,
    'ARCD_TEMPERATURE': '0.7',
    'ARCD_TOP_P': '0.95',
    'ARCD_TOP_K': '20',
    'ARCD_REASONING_FIELD': 'reasoning_content',
    'ARCD_MAX_OUTPUT': 12 * 1024,
    'ARCD_CONTEXT_BUDGET': 116 * 1024,
    'ARCD_CONTEXT_DRAIN_TARGET': 58 * 1024,
    'ARCD_MAX_HISTORY_TURNS': 150,
    'ARCD_HISTORY_TURN_DRAIN': 30,
    'ARCD_TOOL_TIMEOUT': 45,
    'ARCD_TOOL_OUTPUT_TOKENS': 3072,
    'ARCD_IMAGE_SCALE': 10,
    'ARCD_EXPOSE_UNDO': '1',
    'ARCD_GUARDS_FROM_LEVEL': 1,
    'ARCD_SLOTS': 10,
    'ARCD_WARMUP_AFTER': 240,
    'ARCD_STATUS_EVERY': 600,
    'ARCD_DEBUG_DIR': '/kaggle/working/arcd_trace',
}
os.environ.update({k: str(v) for k, v in setup_env.items()})
"""

SOURCE_CELL = '''# Unpack the arcduck2 agent layer (shipped inside this notebook; no extra dataset needed).
import base64, io, zipfile
ARCDUCK2_ZIP_B64 = (
{chunks}
)
_dst = Path('/kaggle/arcduck2_src')
if _dst.exists():
    import shutil; shutil.rmtree(_dst)
_dst.mkdir(parents=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(''.join(ARCDUCK2_ZIP_B64)))) as zf:
    zf.extractall(_dst)
sys.path.insert(0, str(_dst))
import arcduck2
from arcduck2.solver import DuckV2Solver
print('arcduck2', arcduck2.__version__, 'loaded from', arcduck2.__file__)
'''

CUSTOM_CELL = '''# Swap the original harness solver for arcduck2 and set the run budget.
NOTEBOOK_LIMIT_S = 32400            # Kaggle's 9 hour limit
SAFETY_MARGIN_S = 15 * 60           # every game is wound down this long before the limit
hard_deadline_epoch = NOTEBOOK_START_EPOCH + NOTEBOOK_LIMIT_S - SAFETY_MARGIN_S

bm.n_passes = 1
bm.game_weights = None

# Interactive "Save & Run" plays a 10-game subset offline; a real rerun plays the live games.
demo_excluded_games = [] if TRUE_SUBMISSION else ['bp35', 'cd82', 'cn04', 'dc22', 'g50t', 'ka59', 'lf52', 'ls20', 'm0r0', 's5i5', 'sk48', 'sp80', 'su15', 'tn36', 'wa30']
INTERACTIVE_MINUTES = 90            # length of a test run outside a real submission

if TRUE_SUBMISSION:
    solver_deadline = hard_deadline_epoch
else:
    solver_deadline = min(hard_deadline_epoch, time.time() + INTERACTIVE_MINUTES * 60)
    if float(getattr(target, 'max_runtime_s', 0.0) or 0.0) != 32400.0:
        raise RuntimeError(f'Expected the 32400-second notebook budget, got {target.max_runtime_s!r}.')

bm.solver = DuckV2Solver(
    hard_deadline_epoch=solver_deadline,
    max_runtime_s_per_game=max(60.0, solver_deadline - time.time()),
)
print('hard deadline in', round((solver_deadline - time.time()) / 60), 'minutes; solver =', bm.solver.label)
'''


GPU_CHECK_CELL = '''import subprocess as _sp
_gpu = _sp.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
               capture_output=True, text=True).stdout.strip()
print("GPU:", _gpu or "none visible")
_gpu_name, _, _gpu_mem = _gpu.partition(",")
_gpu_ok = "RTX PRO 6000" in _gpu_name.upper() or (_gpu_mem.strip().isdigit() and int(_gpu_mem) >= 90000)
if not _gpu_ok:
    raise RuntimeError(
        f"Wrong accelerator: {_gpu or 'no GPU'!r}. This notebook needs the RTX PRO 6000 (96 GB). "
        "Select it in the notebook settings (or push with machine_shape NvidiaRtxPro6000) and run again.")
'''

SERVER_GATE_CELL = '''import json as _json
import urllib.request as _rq

_server_err = ""
for _attempt in range(6):
    try:
        _req = _rq.Request(
            SERVER_BASE_URL + "/chat/completions",
            data=_json.dumps({"model": SERVED_MODEL_NAME, "max_tokens": 8, "temperature": 0,
                              "messages": [{"role": "user", "content": "Reply with the single word OK."}]}).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
        with _rq.urlopen(_req, timeout=600) as _resp:
            _body = _json.loads(_resp.read())
        assert _body["choices"][0]["message"], "empty completion"
        _server_err = ""
        break
    except Exception as _exc:  # noqa: BLE001
        _server_err = f"{type(_exc).__name__}: {_exc}"
        time.sleep(15)
if _server_err:
    _tail = ""
    _log = WORKING_DIR / "serve.log"
    if _log.exists():
        _tail = _log.read_text(errors="replace")[-3000:]
    raise RuntimeError(f"The model server is not answering ({_server_err}). Last of serve.log:\\n{_tail}")
print("model server answered a test request")
'''


def main() -> None:
    nb_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    assert len(cells) == 23, f"unexpected upstream notebook layout: {len(cells)} cells"
    assert "%%writefile /kaggle/harness-changes.patch" in cell_text(cells[2]), "patch cell moved"

    out_cells: list[dict] = [make_cell("markdown", ABOUT)]
    for idx in range(3, 23):
        text = cell_text(cells[idx])
        kind = cells[idx]["cell_type"]
        if idx == 4:
            text = replace_between(text, "# Apply harness patch", "print('harness patch applied successfully')",
                                   "# Copy the original (unmodified) source bundle; the agent layer is shipped separately.\n"
                                   "BUNDLE_DIR = Path('/kaggle/taaf-kaggle-source-share')\n"
                                   "!rm -Rf $BUNDLE_DIR\n!cp -a $ORIG_BUNDLE_DIR $BUNDLE_DIR\nprint('source bundle ready')\n")
            assert "BUNDLE_DIR = Path(" in text
            i = text.index("setup_env = {")
            text = text[:i] + ENV_BLOCK
            text = text.replace("os.environ['ARC3_WARMUP_ACTION_GAMES'] = '1'\n", "").replace(
                "os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '900'\n", "")
            assert "harness-changes.patch" not in text and "ARC3_" not in text, "stale harness settings left in env cell"
        if idx == 16:
            text = CUSTOM_CELL
        out_cells.append(make_cell(kind, text))
        if idx == 4:
            out_cells.append(make_cell("markdown", "## 1b. Preflight: accelerator\n\nStop immediately if this is not the RTX PRO 6000. A pushed notebook defaults to a T4, where the model server crashes and a run used to finish 'successfully' having played nothing."))
            out_cells.append(make_cell("code", GPU_CHECK_CELL))
        if idx == 12:
            out_cells.append(make_cell("markdown", "## 5b. Preflight: model server\n\nStop immediately unless the model server answers a real request."))
            out_cells.append(make_cell("code", SERVER_GATE_CELL))
        if idx == 10:
            b64 = zip_source_b64()
            chunks = "\n".join(f"    {b64[i:i + 100]!r}" for i in range(0, len(b64), 100))
            out_cells.append(make_cell("markdown", "## 4b. Load the arcduck2 agent layer\n\nUnpack the solver source shipped inside this notebook and import it."))
            out_cells.append(make_cell("code", SOURCE_CELL.format(chunks=chunks)))

    nb["cells"] = out_cells
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "arc-agi-3-arcduck2.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")

    meta = {
        "id": "sudheer56/arc-agi-3-arcduck2",
        "title": "ARC-AGI-3 arcduck2",
        "code_file": "arc-agi-3-arcduck2.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": ["dfranzen/pennyroyal-v253", "dfranzen/taaf-kaggle-source-bundle-copy"],
        "kernel_sources": [],
        "machine_shape": "NvidiaRtxPro6000",
        "docker_image_pinning_type": "original",
        "docker_image": PINNED_IMAGE,
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": [
            "dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1",
            "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1",
        ],
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'arc-agi-3-arcduck2.ipynb'} ({len(out_cells)} cells, {len(b64) // 1024} KiB of embedded source)")


if __name__ == "__main__":
    main()
