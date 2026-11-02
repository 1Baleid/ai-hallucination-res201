"""Perceptual hashes, square thumbnails and normalised cross-correlation (kernel-side).

This module runs inside the private Kaggle kernels (bundled by ``kaggle_kernels/build_kernels.py``)
and locally. It uses only numpy, scipy and pillow and Python 3.10 syntax. Intra-package imports are
not used here; private helpers carry the ``_ih_`` prefix so the bundled kernels have no name clashes.

Conventions
-----------
* A grayscale image is a 2-D ``uint8`` array (rows x columns).
* A bit vector is a 1-D ``bool`` array in row-major order of the hash block, most significant bit
  first. ``bits_to_hex`` gives the same string as ``str(imagehash.phash(...))`` and ``pack`` gives the
  same bytes as ``bytes.fromhex`` of that string.
* ``phash_bits`` reproduces ``imagehash.phash`` (version 4.3.x) bit for bit: grayscale ("L") image,
  PIL LANCZOS resize to (hash_size * highfreq_factor) squared, DCT-II (scipy, unnormalised) along
  axis 0 then axis 1, top-left hash_size x hash_size block, strict "greater than the median" threshold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

try:  # imagehash.phash calls scipy.fftpack.dct; scipy.fft.dct computes the same DCT-II if fftpack goes
    from scipy.fftpack import dct as _ih_dct
except ImportError:  # pragma: no cover - fallback for a future scipy without fftpack
    from scipy.fft import dct as _ih_dct

_IH_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

#: popcount of every byte value (256-entry lookup table used by ``hamming``)
POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

ImageLike = Union[str, Path, Image.Image, np.ndarray]


# --------------------------------------------------------------------------- conversions
def _ih_array_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a 2-D (gray) or 3-D (RGB/RGBA) array to a PIL image without changing uint8 values.

    Non-uint8 arrays: booleans map to 0/255; floats are clipped to [0, 255] and rounded; integer
    arrays whose maximum exceeds 255 (16-bit data) are rescaled linearly by 255 / max.
    """
    a = np.asarray(arr)
    if a.dtype == np.bool_:
        a = a.astype(np.uint8) * 255
    elif a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating):
            a = np.clip(np.nan_to_num(a, nan=0.0), 0.0, 255.0)
            a = np.round(a).astype(np.uint8)
        else:
            a = a.astype(np.int64)
            hi = int(a.max()) if a.size else 0
            if hi > 255:
                a = np.round(a.astype(np.float64) * (255.0 / hi))
            a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        return Image.fromarray(np.ascontiguousarray(a))
    if a.ndim == 3 and a.shape[2] in (3, 4):
        return Image.fromarray(np.ascontiguousarray(a))
    if a.ndim == 3 and a.shape[2] == 1:
        return Image.fromarray(np.ascontiguousarray(a[:, :, 0]))
    raise ValueError(f"expected a 2-D gray or 3-D RGB/RGBA array, got shape {a.shape}")


def _ih_to_gray_pil(im: Image.Image) -> Image.Image:
    """Return a mode "L" version of a PIL image (16-bit and float modes are rescaled, not clipped)."""
    if im.mode == "L":
        return im
    if im.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):
        return _ih_array_to_pil(np.asarray(im))
    return im.convert("L")


def _ih_as_gray_pil(img: ImageLike) -> Image.Image:
    """Accept an array, a PIL image or a path and return a mode "L" PIL image."""
    if isinstance(img, np.ndarray):
        return _ih_to_gray_pil(_ih_array_to_pil(img))
    if isinstance(img, Image.Image):
        return _ih_to_gray_pil(img)
    with Image.open(img) as im:
        im.load()
        gray = _ih_to_gray_pil(im)
        return im.copy() if gray is im else gray


def load_gray(path_or_image: ImageLike, size: int | None = None, draft: bool = False) -> np.ndarray:
    """Load an image as a 2-D uint8 grayscale array.

    Parameters
    ----------
    path_or_image:
        File path, PIL image or numpy array (2-D gray, or 3-D RGB/RGBA).
    size:
        If given, the image is resized to ``size`` x ``size`` (square, aspect ratio not kept) with
        PIL LANCZOS; this is the geometry used for hashing, thumbnails and the spot check.
    draft:
        For JPEG files only: let the JPEG decoder produce a reduced-scale grayscale image directly
        (``Image.draft("L", (size, size))``; the result is never smaller than ``size``), which is
        several times faster for the large RANZCR JPEGs. Ignored for other formats and inputs.
    """
    if size is not None and int(size) < 1:
        raise ValueError("size must be a positive integer")
    if isinstance(path_or_image, (np.ndarray, Image.Image)):
        gray = _ih_as_gray_pil(path_or_image)
    else:
        with Image.open(path_or_image) as im:
            if draft and size is not None and im.format == "JPEG":
                im.draft("L", (int(size), int(size)))
            im.load()
            gray = _ih_to_gray_pil(im)
            if gray is im:
                gray = im.copy()
    if size is not None:
        gray = gray.resize((int(size), int(size)), _IH_LANCZOS)
    return np.array(gray, dtype=np.uint8)


# --------------------------------------------------------------------------- hashes
def phash_bits(img: ImageLike, hash_size: int = 16, highfreq_factor: int = 4) -> np.ndarray:
    """Perceptual hash bits (1-D bool, length hash_size**2), identical to ``imagehash.phash``.

    The image (array, PIL image or path) is converted to mode "L", resized with LANCZOS to
    (hash_size * highfreq_factor) squared, transformed with an unnormalised DCT-II along axis 0 then
    axis 1, and the top-left hash_size x hash_size block is thresholded strictly above its median.
    """
    if hash_size < 2:
        raise ValueError("hash_size must be at least 2")
    if highfreq_factor < 1:
        raise ValueError("highfreq_factor must be at least 1")
    side = int(hash_size) * int(highfreq_factor)
    pixels = np.asarray(_ih_as_gray_pil(img).resize((side, side), _IH_LANCZOS))
    coeffs = _ih_dct(_ih_dct(pixels, axis=0), axis=1)
    low = coeffs[:hash_size, :hash_size]
    return np.asarray(low > np.median(low), dtype=bool).reshape(-1)


def dhash_bits(img: ImageLike, hash_size: int = 16) -> np.ndarray:
    """Horizontal difference hash bits (1-D bool, length hash_size**2), identical to ``imagehash.dhash``."""
    if hash_size < 2:
        raise ValueError("hash_size must be at least 2")
    pixels = np.asarray(_ih_as_gray_pil(img).resize((hash_size + 1, hash_size), _IH_LANCZOS))
    return np.asarray(pixels[:, 1:] > pixels[:, :-1], dtype=bool).reshape(-1)


def bits_to_hex(bits: np.ndarray) -> str:
    """Hex string of a bit vector (most significant bit first), the format of ``str(ImageHash)``."""
    flat = np.asarray(bits, dtype=bool).reshape(-1)
    if flat.size == 0:
        raise ValueError("empty bit vector")
    if flat.size % 8 == 0:
        return np.packbits(flat).tobytes().hex()
    width = (flat.size + 3) // 4
    value = int("".join("1" if b else "0" for b in flat), 2)
    return format(value, "0{}x".format(width))


def hex_to_bits(hex_str: str, n_bits: int) -> np.ndarray:
    """Inverse of ``bits_to_hex``: a 1-D bool array of length ``n_bits``."""
    s = str(hex_str).strip().lower()
    if n_bits < 1:
        raise ValueError("n_bits must be positive")
    if len(s) != (n_bits + 3) // 4:
        raise ValueError(f"hex string of length {len(s)} does not encode {n_bits} bits")
    if n_bits % 8 == 0:
        return np.unpackbits(np.frombuffer(bytes.fromhex(s), dtype=np.uint8)).astype(bool)
    value = int(s, 16)
    if value >> n_bits:
        raise ValueError("hex value has more than n_bits significant bits")
    return np.array([c == "1" for c in format(value, "0{}b".format(n_bits))], dtype=bool)


def pack(bit_rows: np.ndarray) -> np.ndarray:
    """Pack bit rows (n x n_bits, or one 1-D row) into an (n x n_bits/8) uint8 matrix (MSB first)."""
    rows = np.asarray(bit_rows, dtype=bool)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2:
        raise ValueError("bit_rows must be 1-D or 2-D")
    if rows.shape[1] % 8 != 0:
        raise ValueError("the number of bits per row must be a multiple of 8")
    return np.packbits(rows, axis=1)


def packed_from_hex(hex_list) -> np.ndarray:
    """Pack many equal-length hex hashes into an (n x n_bytes) uint8 matrix (fast path for matching)."""
    items = [str(h).strip().lower() for h in hex_list]
    if not items:
        return np.zeros((0, 0), dtype=np.uint8)
    width = len(items[0])
    if width == 0 or width % 2 != 0:
        raise ValueError("hex hashes must have an even, non-zero length")
    for h in items:
        if len(h) != width:
            raise ValueError("hex hashes have different lengths")
    buf = bytes.fromhex("".join(items))
    return np.frombuffer(buf, dtype=np.uint8).reshape(len(items), width // 2).copy()


def hamming(q_packed: np.ndarray, db_packed: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Hamming distances between packed hashes via a 256-entry popcount table.

    ``q_packed`` is (n_q x n_bytes) or one 1-D query; ``db_packed`` is (n_db x n_bytes). Returns an
    int32 matrix (n_q x n_db), or a 1-D array (n_db,) for a 1-D query. The database is processed in
    blocks of ``chunk`` rows and the queries in blocks sized so that one XOR block stays near 16 MB.
    """
    if int(chunk) < 1:
        raise ValueError("chunk must be at least 1")
    q = np.asarray(q_packed, dtype=np.uint8)
    db = np.asarray(db_packed, dtype=np.uint8)
    single = q.ndim == 1
    if single:
        q = q[None, :]
    if db.ndim == 1:
        db = db[None, :]
    if q.ndim != 2 or db.ndim != 2:
        raise ValueError("packed hashes must be 1-D or 2-D uint8 arrays")
    n_q, n_db = q.shape[0], db.shape[0]
    if n_db and n_q and q.shape[1] != db.shape[1]:
        raise ValueError(f"byte width differs: queries {q.shape[1]}, database {db.shape[1]}")
    out = np.zeros((n_q, n_db), dtype=np.int32)
    if n_q and n_db:
        n_bytes = q.shape[1]
        step = int(chunk)
        q_block = max(1, (1 << 24) // max(1, step * n_bytes))
        for i0 in range(0, n_q, q_block):
            qi = q[i0:i0 + q_block]
            for j0 in range(0, n_db, step):
                x = np.bitwise_xor(qi[:, None, :], db[None, j0:j0 + step, :])
                out[i0:i0 + qi.shape[0], j0:j0 + x.shape[1]] = POPCOUNT_TABLE[x].sum(axis=2, dtype=np.int32)
    return out[0] if single else out


# --------------------------------------------------------------------------- thumbnails and NCC
def thumb(img: ImageLike, side: int = 128, draft: bool = True) -> np.ndarray:
    """Square grayscale thumbnail (side x side, float32) for normalised cross-correlation.

    Paths to JPEG files use the draft decoder (see ``load_gray``) unless ``draft`` is False.
    """
    return load_gray(img, size=side, draft=draft).astype(np.float32)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalised cross-correlation of two equal-shape images, in [-1, 1].

    Invariant to affine intensity changes (a -> s * a + c with s > 0). Returns 0.0 when either
    image is constant (the correlation is undefined), so a constant image never passes a threshold.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.size == 0:
        raise ValueError("empty images")
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if not np.isfinite(den) or den <= 0.0:
        return 0.0
    return float(np.clip(np.sum(x * y) / den, -1.0, 1.0))
