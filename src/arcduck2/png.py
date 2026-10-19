"""Minimal PNG encoder (numpy + zlib only).

Pillow's import broke on Kaggle's pinned Python 3.12 image (mixed PIL module versions), and
a whole run silently did nothing because of it. Encoding a flat-colour RGB image needs none of
Pillow, so the agent no longer depends on it.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def encode_png(rgb: np.ndarray, level: int = 6) -> bytes:
    """Encode an (H, W, 3) uint8 array as an 8-bit RGB PNG."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected (H, W, 3) uint8, got {rgb.shape} {rgb.dtype}")
    h, w, _ = rgb.shape
    raw = np.zeros((h, w * 3 + 1), dtype=np.uint8)          # first byte of each row = filter type 0
    raw[:, 1:] = rgb.reshape(h, w * 3)
    header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return _SIGNATURE + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(raw.tobytes(), level)) + _chunk(b"IEND", b"")
