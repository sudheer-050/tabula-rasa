import numpy as np
from arcduck2 import perception as P


def grid():
    g = np.zeros((64, 64), dtype=np.int8)
    g[10:13, 20:23] = 8
    g[40, 5] = 3
    return g


def test_ascii_shape_and_rulers():
    text = P.ascii_grid(grid())
    lines = text.splitlines()
    assert len(lines) == 66
    assert lines[2].startswith("  0 ")
    assert len(lines[2]) == 4 + 64
    assert lines[2 + 10][4 + 20:4 + 23] == "888"


def test_diff_summary_and_mask():
    a, b = grid(), grid()
    assert P.diff_summary(a, b) == "no cells changed"
    b[10:13, 20:23] = 0
    b[10:13, 24:27] = 8
    s = P.diff_summary(a, b)
    assert "18 cells changed" in s and "8->0 x9" in s and "0->8 x9" in s
    assert P.diff_mask(a, b).sum() == 18


def test_segments_ignores_background():
    segs = P.segments(grid())
    assert {s["color"] for s in segs} == {8, 3}
    big = segs[0]
    assert big["size"] == 9 and (big["x0"], big["y0"], big["x1"], big["y1"]) == (20, 10, 22, 12)


def test_object_changes_moved_new_gone():
    a, b = grid(), grid()
    b[10:13, 20:23] = 0
    b[10:13, 24:27] = 8
    notes = P.object_changes(a, b)
    assert any(n.startswith("moved") and "dx=4" in n for n in notes)
    c = grid()
    c[40, 5] = 0
    assert any(n.startswith("gone") for n in P.object_changes(a, c))
    d = grid()
    d[50, 50] = 11
    assert any(n.startswith("new") for n in P.object_changes(a, d))


def test_png_is_valid_and_scaled():
    png = P.render_png(grid(), scale=10, diff_against=np.zeros((64, 64), dtype=np.int8))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(png))
    assert img.size[0] >= 640 and img.size[1] >= 640
    assert P.png_data_url(png).startswith("data:image/png;base64,")


def test_shape_mismatch_is_safe():
    a = np.zeros((64, 64), dtype=np.int8)
    b = np.zeros((32, 32), dtype=np.int8)
    assert "size changed" in P.diff_summary(a, b)
    assert P.diff_mask(a, b).all()
