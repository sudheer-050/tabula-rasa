"""Turn raw 0-15 frames into things a model can read: text, images, diffs.

Pure functions over numpy arrays. No game or network dependencies, so every
function here is unit-testable on its own.
"""
from __future__ import annotations

import base64
from typing import Any

import numpy as np

ARC_RGB: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255), 1: (204, 204, 204), 2: (153, 153, 153), 3: (102, 102, 102),
    4: (51, 51, 51), 5: (0, 0, 0), 6: (229, 58, 163), 7: (255, 123, 204),
    8: (249, 60, 49), 9: (30, 147, 255), 10: (136, 216, 241), 11: (255, 220, 0),
    12: (255, 133, 27), 13: (146, 18, 49), 14: (79, 204, 48), 15: (163, 86, 214),
}
_HEX = "0123456789abcdef"


def as_array(grid: Any) -> np.ndarray:
    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError(f"grid must be 2-D, got shape {arr.shape}")
    return arr.astype(np.int16, copy=False)


def ascii_grid(grid: Any) -> str:
    """One hex digit per cell, one line per row, with column/row rulers."""
    arr = as_array(grid)
    h, w = arr.shape
    tens = "".join(str((x // 10) % 10) if x % 10 == 0 else " " for x in range(w))
    ones = "".join(str(x % 10) for x in range(w))
    lines = [f"    {tens}", f"    {ones}"]
    for y in range(h):
        row = "".join(_HEX[int(v) & 15] for v in arr[y])
        lines.append(f"{y:3d} {row}")
    return "\n".join(lines)


def grid_hash(grid: Any) -> int:
    return hash(as_array(grid).astype(np.int8).tobytes())


def diff_mask(prev: Any, cur: Any) -> np.ndarray:
    a, b = as_array(prev), as_array(cur)
    if a.shape != b.shape:
        return np.ones(b.shape, dtype=bool)
    return a != b


def diff_summary(prev: Any, cur: Any, max_cells: int = 40) -> str:
    """Compact human-readable description of what changed between two frames."""
    a, b = as_array(prev), as_array(cur)
    if a.shape != b.shape:
        return f"frame size changed {a.shape} -> {b.shape} (full redraw)"
    mask = a != b
    n = int(mask.sum())
    if n == 0:
        return "no cells changed"
    ys, xs = np.nonzero(mask)
    head = f"{n} cells changed, bbox x={int(xs.min())}..{int(xs.max())} y={int(ys.min())}..{int(ys.max())}"
    trans: dict[tuple[int, int], int] = {}
    for y, x in zip(ys.tolist(), xs.tolist()):
        key = (int(a[y, x]), int(b[y, x]))
        trans[key] = trans.get(key, 0) + 1
    parts = [f"{o:x}->{c:x} x{k}" for (o, c), k in sorted(trans.items(), key=lambda kv: -kv[1])[:8]]
    out = head + "; recolors: " + ", ".join(parts)
    if n <= max_cells:
        cells = ", ".join(f"({x},{y}):{int(a[y, x]):x}->{int(b[y, x]):x}" for y, x in zip(ys.tolist(), xs.tolist()))
        out += f"; cells: {cells}"
    return out


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """4-connected components of a boolean mask as (y0, y1, x0, x1, size), inclusive bounds.

    Uses scipy when it imports and works; otherwise a pure-Python flood fill, so a broken
    scipy install can never take the agent down."""
    try:
        from scipy import ndimage

        labeled, n = ndimage.label(mask)
        comps = []
        for idx, sl in enumerate(ndimage.find_objects(labeled), start=1):
            if sl is None:
                continue
            ys, xs = sl
            comps.append((int(ys.start), int(ys.stop - 1), int(xs.start), int(xs.stop - 1), int((labeled[sl] == idx).sum())))
        return comps
    except Exception:  # noqa: BLE001 - any scipy/numpy ABI problem falls through to the fallback
        pass
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for sy, sx in zip(*np.nonzero(mask)):
        if seen[sy, sx]:
            continue
        stack = [(int(sy), int(sx))]
        seen[sy, sx] = True
        y0 = y1 = int(sy)
        x0 = x1 = int(sx)
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            y0, y1, x0, x1 = min(y0, y), max(y1, y), min(x0, x), max(x1, x)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        comps.append((y0, y1, x0, x1, size))
    return comps


def _components_with_mask(mask: np.ndarray) -> list[tuple[int, int, int, int, int, np.ndarray]]:
    """Like ``_components`` but also returns each component's own cropped boolean mask
    (aligned to its bbox), needed for a shape hash, true adjacency, and containment --
    a separate function so ``segments``/``object_changes`` keep their existing, tested
    contract untouched."""
    try:
        from scipy import ndimage

        labeled, n = ndimage.label(mask)
        comps = []
        for idx, sl in enumerate(ndimage.find_objects(labeled), start=1):
            if sl is None:
                continue
            ys, xs = sl
            local = labeled[sl] == idx
            comps.append((int(ys.start), int(ys.stop - 1), int(xs.start), int(xs.stop - 1), int(local.sum()), local))
        return comps
    except Exception:  # noqa: BLE001 - any scipy/numpy ABI problem falls through to the fallback
        pass
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for sy, sx in zip(*np.nonzero(mask)):
        if seen[sy, sx]:
            continue
        stack = [(int(sy), int(sx))]
        seen[sy, sx] = True
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        ys = [c[0] for c in cells]
        xs = [c[1] for c in cells]
        y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
        local = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
        for y, x in cells:
            local[y - y0, x - x0] = True
        comps.append((y0, y1, x0, x1, len(cells), local))
    return comps


def segmentation(grid: Any, background: int | None = None, min_size: int = 1, limit: int = 200) -> dict:
    """A graph view of the board: 4-connected same-colour components as nodes, each with a
    position-invariant ``hash`` (equal hash = same shape and colour regardless of where it
    moved -- use it to track an object across frames, or spot duplicates in one frame) and
    ``children`` (ids of objects whose bbox it fully encloses), plus an ``adjacency_list`` of
    which objects actually touch (true 4-connected adjacency between their pixels, not just
    overlapping bounding boxes). Nodes are ordered top-most, then left-most.

    Returns ``{"nodes": [...], "adjacency_list": [[i, j], ...]}``.
    """
    arr = as_array(grid)
    if background is None:
        vals, counts = np.unique(arr, return_counts=True)
        background = int(vals[int(np.argmax(counts))])
    raw: list[dict] = []
    for color in np.unique(arr).tolist():
        if color == background:
            continue
        for y0, y1, x0, x1, size, local in _components_with_mask(arr == color):
            if size < min_size:
                continue
            raw.append({"color": int(color), "y0": y0, "y1": y1, "x0": x0, "x1": x1, "size": size, "mask": local})
    raw.sort(key=lambda c: (c["y0"], c["x0"]))
    raw = raw[:limit]

    h, w = arr.shape
    full_masks: list[np.ndarray] = []
    nodes: list[dict] = []
    for i, c in enumerate(raw):
        shape_bits = np.packbits(c["mask"]).tobytes()
        nodes.append({
            "id": i,
            "color": c["color"],
            "size": c["size"],
            "hash": hash((c["color"], c["mask"].shape, shape_bits)),
            "x0": c["x0"], "y0": c["y0"], "x1": c["x1"], "y1": c["y1"],
            "children": [],
        })
        full = np.zeros((h, w), dtype=bool)
        full[c["y0"]:c["y1"] + 1, c["x0"]:c["x1"] + 1] = c["mask"]
        full_masks.append(full)

    for i, a in enumerate(raw):
        for j, b in enumerate(raw):
            if i == j:
                continue
            same_box = (a["x0"], a["y0"], a["x1"], a["y1"]) == (b["x0"], b["y0"], b["x1"], b["y1"])
            if not same_box and a["x0"] <= b["x0"] and a["y0"] <= b["y0"] and a["x1"] >= b["x1"] and a["y1"] >= b["y1"]:
                nodes[i]["children"].append(j)

    adjacency_list: list[list[int]] = []
    for i in range(len(raw)):
        m = full_masks[i]
        dilated = np.zeros((h, w), dtype=bool)
        dilated[:-1, :] |= m[1:, :]
        dilated[1:, :] |= m[:-1, :]
        dilated[:, :-1] |= m[:, 1:]
        dilated[:, 1:] |= m[:, :-1]
        for j in range(i + 1, len(raw)):
            if np.any(dilated & full_masks[j]):
                adjacency_list.append([i, j])

    return {"nodes": nodes, "adjacency_list": adjacency_list}


def segments(grid: Any, background: int | None = None, min_size: int = 1, limit: int = 60) -> list[dict]:
    """4-connected same-colour components, largest first. Background defaults to the modal colour."""
    arr = as_array(grid)
    if background is None:
        vals, counts = np.unique(arr, return_counts=True)
        background = int(vals[int(np.argmax(counts))])
    out: list[dict] = []
    for color in np.unique(arr).tolist():
        if color == background:
            continue
        for y0, y1, x0, x1, size in _components(arr == color):
            if size < min_size:
                continue
            out.append({
                "color": int(color),
                "size": size,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "cx": round(float(x0 + x1) / 2, 1),
                "cy": round(float(y0 + y1) / 2, 1),
            })
    out.sort(key=lambda s: (-s["size"], s["color"], s["y0"], s["x0"]))
    return out[:limit]


def object_changes(prev: Any, cur: Any, limit: int = 12) -> list[str]:
    """Which same-coloured components appeared, vanished, or moved between frames."""
    a, b = as_array(prev), as_array(cur)
    if a.shape != b.shape:
        return ["full redraw (shape changed)"]
    sa, sb = segments(a, background=0, limit=400), segments(b, background=0, limit=400)
    used: set[int] = set()
    notes: list[str] = []
    for s in sa:
        match = None
        for j, t in enumerate(sb):
            if j in used or t["color"] != s["color"] or t["size"] != s["size"]:
                continue
            if (t["x1"] - t["x0"], t["y1"] - t["y0"]) != (s["x1"] - s["x0"], s["y1"] - s["y0"]):
                continue
            if match is None or abs(t["x0"] - s["x0"]) + abs(t["y0"] - s["y0"]) < abs(sb[match]["x0"] - s["x0"]) + abs(sb[match]["y0"] - s["y0"]):
                match = j
        if match is None:
            notes.append(f"gone: color {s['color']:x} size {s['size']} at ({s['x0']},{s['y0']})")
            continue
        used.add(match)
        t = sb[match]
        dx, dy = t["x0"] - s["x0"], t["y0"] - s["y0"]
        if dx or dy:
            notes.append(f"moved: color {s['color']:x} size {s['size']} by dx={dx} dy={dy} to ({t['x0']},{t['y0']})")
    for j, t in enumerate(sb):
        if j not in used:
            notes.append(f"new: color {t['color']:x} size {t['size']} at ({t['x0']},{t['y0']})")
    return notes[:limit]


def render_png(grid: Any, scale: int = 10, diff_against: Any | None = None, rulers: bool = True) -> bytes:
    """Render a frame as PNG (no Pillow). Changed cells (vs ``diff_against``) get a magenta outline.

    With ``rulers`` a margin is added with a tick every 8 cells starting at 0 (x along the top,
    y down the left side), so cell (x, y) sits at pixel (pad + x*scale, pad + y*scale)."""
    from .png import encode_png

    arr = as_array(grid)
    h, w = arr.shape
    scale = max(1, int(scale))
    palette = np.array([ARC_RGB[i] for i in range(16)], dtype=np.uint8)
    img = np.repeat(np.repeat(palette[np.clip(arr, 0, 15)], scale, axis=0), scale, axis=1)
    if diff_against is not None:
        magenta = np.array((255, 0, 255), dtype=np.uint8)
        for y, x in zip(*np.nonzero(diff_mask(diff_against, arr))):
            y0, x0 = int(y) * scale, int(x) * scale
            y1, x1 = y0 + scale - 1, x0 + scale - 1
            img[y0, x0:x1 + 1] = magenta
            img[y1, x0:x1 + 1] = magenta
            img[y0:y1 + 1, x0] = magenta
            img[y0:y1 + 1, x1] = magenta
    if rulers:
        pad = max(14, scale + 4)
        canvas = np.full((h * scale + pad, w * scale + pad, 3), 245, dtype=np.uint8)
        canvas[pad:, pad:] = img
        step = 8 if w >= 32 else 4
        for x in range(0, w, step):
            canvas[: pad - 2, pad + x * scale] = 0
        for y in range(0, h, step):
            canvas[pad + y * scale, : pad - 2] = 0
        img = canvas
    return encode_png(np.ascontiguousarray(img))


def png_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
