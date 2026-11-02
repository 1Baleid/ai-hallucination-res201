"""Tests for cxrdev.imghash: bit-for-bit parity with imagehash, packing, Hamming, loading, NCC."""
from __future__ import annotations

import io

import imagehash
import numpy as np
import pytest
from PIL import Image

from cxrdev.imghash import (
    POPCOUNT_TABLE,
    bits_to_hex,
    dhash_bits,
    hamming,
    hex_to_bits,
    load_gray,
    ncc,
    pack,
    packed_from_hex,
    phash_bits,
    thumb,
)


def _structured_images() -> list[np.ndarray]:
    """Random and structured grayscale test images of several shapes."""
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:200, 0:300]
    imgs = [
        rng.integers(0, 256, size=(64, 64), dtype=np.uint8),
        rng.integers(0, 256, size=(123, 457), dtype=np.uint8),
        rng.integers(0, 256, size=(1024, 1024), dtype=np.uint8),
        (xx * 255 // 299).astype(np.uint8),                                   # horizontal gradient
        (yy * 255 // 199).astype(np.uint8),                                   # vertical gradient
        (((xx // 25 + yy // 25) % 2) * 255).astype(np.uint8),                  # checkerboard
        np.full((50, 70), 128, dtype=np.uint8),                                # constant (median ties)
        np.zeros((8, 8), dtype=np.uint8),                                      # tiny, black
        (127 + 100 * np.sin(xx / 7.0) * np.cos(yy / 11.0)).astype(np.uint8),   # smooth texture
        np.clip(rng.normal(128, 60, size=(300, 200)), 0, 255).astype(np.uint8),
        np.where(rng.random((90, 90)) > 0.97, 255, 0).astype(np.uint8),        # sparse dots
        np.tile(np.array([0, 255], dtype=np.uint8), (33, 16)),                 # 1-px stripes, odd shape
    ]
    return imgs


@pytest.mark.parametrize("hash_size", [8, 16])
def test_phash_matches_imagehash_bit_for_bit(hash_size: int) -> None:
    for k, arr in enumerate(_structured_images()):
        ref = imagehash.phash(Image.fromarray(arr), hash_size=hash_size)
        bits = phash_bits(arr, hash_size=hash_size)
        assert bits.dtype == bool and bits.shape == (hash_size * hash_size,)
        assert np.array_equal(bits, ref.hash.reshape(-1)), f"image {k}"
        assert bits_to_hex(bits) == str(ref), f"image {k}"


@pytest.mark.parametrize("hash_size,highfreq", [(8, 4), (16, 4), (16, 2), (12, 3)])
def test_phash_parameters_match_imagehash(hash_size: int, highfreq: int) -> None:
    rng = np.random.default_rng(hash_size * 10 + highfreq)
    for _ in range(10):
        arr = rng.integers(0, 256, size=tuple(rng.integers(16, 400, size=2)), dtype=np.uint8)
        ref = imagehash.phash(Image.fromarray(arr), hash_size=hash_size, highfreq_factor=highfreq)
        assert np.array_equal(phash_bits(arr, hash_size, highfreq), ref.hash.reshape(-1))


def test_phash_color_pil_and_path_inputs(tmp_path) -> None:
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, size=(97, 131, 3), dtype=np.uint8)
    rgba = rng.integers(0, 256, size=(64, 80, 4), dtype=np.uint8)
    for arr in (rgb, rgba):
        pil = Image.fromarray(arr)
        ref = imagehash.phash(pil, hash_size=16)
        assert np.array_equal(phash_bits(arr), ref.hash.reshape(-1))
        assert np.array_equal(phash_bits(pil), ref.hash.reshape(-1))
    gray = rng.integers(0, 256, size=(150, 120), dtype=np.uint8)
    path = tmp_path / "g.png"
    Image.fromarray(gray).save(path)
    ref = imagehash.phash(Image.open(path), hash_size=16)
    assert np.array_equal(phash_bits(path), ref.hash.reshape(-1))
    assert np.array_equal(phash_bits(str(path)), ref.hash.reshape(-1))
    assert np.array_equal(phash_bits(load_gray(path)), ref.hash.reshape(-1))


@pytest.mark.parametrize("hash_size", [8, 16])
def test_dhash_matches_imagehash(hash_size: int) -> None:
    for arr in _structured_images():
        ref = imagehash.dhash(Image.fromarray(arr), hash_size=hash_size)
        bits = dhash_bits(arr, hash_size)
        assert np.array_equal(bits, ref.hash.reshape(-1))
        assert bits_to_hex(bits) == str(ref)


def test_hash_size_validation() -> None:
    arr = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        phash_bits(arr, hash_size=1)
    with pytest.raises(ValueError):
        phash_bits(arr, highfreq_factor=0)
    with pytest.raises(ValueError):
        dhash_bits(arr, hash_size=1)


def test_hex_roundtrip_and_odd_lengths() -> None:
    rng = np.random.default_rng(3)
    for n_bits in (64, 256, 36, 25, 9):
        bits = rng.random(n_bits) > 0.5
        h = bits_to_hex(bits)
        assert len(h) == (n_bits + 3) // 4
        assert np.array_equal(hex_to_bits(h, n_bits), bits)
    # imagehash formatting for a size that is not a multiple of 8 bits
    arr = rng.integers(0, 256, size=(40, 40), dtype=np.uint8)
    ref = imagehash.phash(Image.fromarray(arr), hash_size=6)
    assert bits_to_hex(phash_bits(arr, hash_size=6)) == str(ref)
    assert np.array_equal(hex_to_bits(str(ref), 36), ref.hash.reshape(-1))
    with pytest.raises(ValueError):
        hex_to_bits("abc", 256)
    with pytest.raises(ValueError):
        hex_to_bits("ff", 7)       # value needs 8 bits
    with pytest.raises(ValueError):
        bits_to_hex(np.zeros(0, dtype=bool))


def test_pack_and_packed_from_hex() -> None:
    rng = np.random.default_rng(4)
    rows = rng.random((5, 256)) > 0.5
    packed = pack(rows)
    assert packed.shape == (5, 32) and packed.dtype == np.uint8
    hexes = [bits_to_hex(r) for r in rows]
    for i, h in enumerate(hexes):
        assert packed[i].tobytes() == bytes.fromhex(h)
    assert np.array_equal(packed_from_hex(hexes), packed)
    assert np.array_equal(packed_from_hex([h.upper() for h in hexes]), packed)
    assert pack(rows[0]).shape == (1, 32)
    assert packed_from_hex([]).shape == (0, 0)
    with pytest.raises(ValueError):
        pack(np.zeros((2, 12), dtype=bool))
    with pytest.raises(ValueError):
        packed_from_hex(["ab", "abcd"])
    with pytest.raises(ValueError):
        packed_from_hex(["abc"])


def _brute(q: np.ndarray, db: np.ndarray) -> np.ndarray:
    qb = np.unpackbits(q, axis=1)
    dbb = np.unpackbits(db, axis=1)
    return (qb[:, None, :] != dbb[None, :, :]).sum(axis=2)


def test_popcount_table() -> None:
    assert POPCOUNT_TABLE.shape == (256,)
    assert all(int(POPCOUNT_TABLE[i]) == bin(i).count("1") for i in range(256))


@pytest.mark.parametrize("n_q,n_db,chunk", [(1, 1, 4096), (7, 100, 3), (13, 4097, 4096), (3, 10, 1), (40, 33, 7)])
def test_hamming_matches_brute_force(n_q: int, n_db: int, chunk: int) -> None:
    rng = np.random.default_rng(n_q * 1000 + n_db)
    q = rng.integers(0, 256, size=(n_q, 32), dtype=np.uint8)
    db = rng.integers(0, 256, size=(n_db, 32), dtype=np.uint8)
    d = hamming(q, db, chunk=chunk)
    assert d.shape == (n_q, n_db) and d.dtype == np.int32
    assert np.array_equal(d, _brute(q, db))
    one = hamming(q[0], db, chunk=chunk)
    assert one.shape == (n_db,) and np.array_equal(one, d[0])


def test_hamming_agrees_with_imagehash_difference() -> None:
    imgs = _structured_images()
    hashes = [imagehash.phash(Image.fromarray(a), hash_size=16) for a in imgs]
    packed = pack(np.array([h.hash.reshape(-1) for h in hashes]))
    d = hamming(packed, packed)
    for i in range(len(hashes)):
        for j in range(len(hashes)):
            assert d[i, j] == hashes[i] - hashes[j]
    assert np.all(np.diag(d) == 0)


def test_hamming_edge_cases() -> None:
    q = np.zeros((2, 32), dtype=np.uint8)
    assert hamming(q, np.zeros((0, 32), dtype=np.uint8)).shape == (2, 0)
    assert hamming(np.zeros((0, 32), dtype=np.uint8), q).shape == (0, 2)
    assert hamming(q, np.full(32, 255, dtype=np.uint8)).tolist() == [[256], [256]]
    with pytest.raises(ValueError):
        hamming(q, np.zeros((3, 8), dtype=np.uint8))
    with pytest.raises(ValueError):
        hamming(q, q, chunk=0)


def test_load_gray_modes_and_sizes(tmp_path) -> None:
    rng = np.random.default_rng(11)
    gray = rng.integers(0, 256, size=(60, 90), dtype=np.uint8)
    Image.fromarray(gray).save(tmp_path / "l.png")
    assert np.array_equal(load_gray(tmp_path / "l.png"), gray)
    rgb = np.stack([gray, gray, gray], axis=2)
    Image.fromarray(rgb).save(tmp_path / "rgb.png")
    assert np.array_equal(load_gray(tmp_path / "rgb.png"), gray)        # equal channels -> same gray
    rgba = np.dstack([rgb, np.full((60, 90), 255, dtype=np.uint8)])
    Image.fromarray(rgba).save(tmp_path / "rgba.png")
    assert np.array_equal(load_gray(tmp_path / "rgba.png"), gray)
    sixteen = (gray.astype(np.uint16) * 257)
    Image.fromarray(sixteen).save(tmp_path / "i16.png")                  # 16-bit PNG
    g16 = load_gray(tmp_path / "i16.png")
    assert g16.dtype == np.uint8 and np.abs(g16.astype(int) - gray.astype(int)).max() <= 1
    out = load_gray(gray, size=32)
    assert out.shape == (32, 32) and out.dtype == np.uint8
    ref = np.asarray(Image.fromarray(gray).resize((32, 32), Image.Resampling.LANCZOS))
    assert np.array_equal(out, ref)
    assert np.array_equal(load_gray(Image.fromarray(gray)), gray)
    assert load_gray(gray.astype(np.float64) + 0.4).dtype == np.uint8
    assert np.array_equal(load_gray(gray.astype(np.float64)), gray)
    assert load_gray(np.ones((4, 4), dtype=bool)).max() == 255
    with pytest.raises(ValueError):
        load_gray(gray, size=0)
    with pytest.raises(ValueError):
        load_gray(np.zeros((2, 2, 2), dtype=np.uint8))


def test_load_gray_draft_jpeg(tmp_path) -> None:
    yy, xx = np.mgrid[0:1600, 0:1200]
    big = (127 + 80 * np.sin(xx / 90.0) * np.cos(yy / 130.0)).astype(np.uint8)
    path = tmp_path / "big.jpg"
    Image.fromarray(big).save(path, quality=95)
    full = load_gray(path, size=128)
    fast = load_gray(path, size=128, draft=True)
    assert full.shape == fast.shape == (128, 128)
    assert ncc(full, fast) > 0.999
    with Image.open(path) as im:
        im.draft("L", (128, 128))
        assert im.size[0] < 1200      # the decoder really reduced the scale
    # draft is ignored for PNG
    Image.fromarray(big).save(tmp_path / "big.png")
    assert np.array_equal(load_gray(tmp_path / "big.png", size=64, draft=True),
                          load_gray(tmp_path / "big.png", size=64))
    with Image.open(io.BytesIO(path.read_bytes())) as im:
        assert im.format == "JPEG"


def test_thumb_and_ncc_properties() -> None:
    rng = np.random.default_rng(12)
    a = rng.integers(0, 256, size=(300, 260), dtype=np.uint8)
    t = thumb(a, 128)
    assert t.shape == (128, 128) and t.dtype == np.float32
    assert ncc(t, t) == pytest.approx(1.0)
    assert ncc(t, 255.0 - t) == pytest.approx(-1.0)
    assert ncc(t, 0.5 * t + 17.0) == pytest.approx(1.0)
    b = thumb(rng.integers(0, 256, size=(300, 260), dtype=np.uint8), 128)
    assert -1.0 <= ncc(t, b) <= 1.0 and abs(ncc(t, b)) < 0.1
    assert ncc(t, b) == pytest.approx(ncc(b, t))
    assert ncc(np.full((5, 5), 3.0), t[:5, :5]) == 0.0
    assert ncc(np.full((5, 5), 3.0), np.full((5, 5), 3.0)) == 0.0
    with pytest.raises(ValueError):
        ncc(t, t[:10])
    with pytest.raises(ValueError):
        ncc(np.zeros(0), np.zeros(0))
