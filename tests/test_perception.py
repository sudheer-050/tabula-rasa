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


def test_segmentation_nodes_match_segments():
    seg = P.segmentation(grid())
    assert {n["color"] for n in seg["nodes"]} == {8, 3}
    assert len(seg["nodes"]) == 2
    big = next(n for n in seg["nodes"] if n["color"] == 8)
    assert big["size"] == 9 and (big["x0"], big["y0"], big["x1"], big["y1"]) == (20, 10, 22, 12)
    assert big["children"] == []                        # unrelated dot elsewhere isn't "inside" it
    assert seg["adjacency_list"] == []                   # the two objects don't touch


def test_segmentation_hash_is_position_invariant_and_shape_sensitive():
    a = np.zeros((64, 64), dtype=np.int8)
    a[10:13, 10:13] = 8                                   # a 3x3 block of color 8
    b = np.zeros((64, 64), dtype=np.int8)
    b[40:43, 50:53] = 8                                   # same shape and colour, moved elsewhere
    c = np.zeros((64, 64), dtype=np.int8)
    c[10:12, 10:13] = 8                                   # same colour, different shape (2x3 not 3x3)
    ha = P.segmentation(a)["nodes"][0]["hash"]
    hb = P.segmentation(b)["nodes"][0]["hash"]
    hc = P.segmentation(c)["nodes"][0]["hash"]
    assert ha == hb                                       # moved but otherwise identical -> same identity
    assert ha != hc                                       # different shape -> different identity


def test_segmentation_containment_and_adjacency():
    g = np.zeros((64, 64), dtype=np.int8)
    g[10, 10:15] = 5
    g[14, 10:15] = 5
    g[10:15, 10] = 5
    g[10:15, 14] = 5                                      # a hollow 5x5 ring outline of colour 5
    g[12, 12] = 7                                          # a dot in the middle, not touching the ring
    g[20:22, 20:22] = 8                                    # a 2x2 block
    g[20:22, 22:24] = 9                                    # a 2x2 block immediately to its right -- touching
    seg = P.segmentation(g)
    by_color = {n["color"]: n for n in seg["nodes"]}
    ring, dot, left, right = by_color[5], by_color[7], by_color[8], by_color[9]
    assert dot["id"] in ring["children"]                   # dot is enclosed by the ring's bbox
    assert dot["id"] not in left["children"] and dot["id"] not in right["children"]
    pair = {tuple(sorted(e)) for e in seg["adjacency_list"]}
    assert tuple(sorted((left["id"], right["id"]))) in pair   # touching blocks are adjacent
    assert tuple(sorted((ring["id"], dot["id"]))) not in pair  # contained but not touching -> not adjacent
