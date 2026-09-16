"""Row-blocked quantization must be byte-identical to whole-tensor.

`affine.quantize_blocked` streams a weight in row blocks so a 5 GB f32 copy of a
1.27B-param tensor is never resident. That is legitimate only because an MLX
affine group never spans output rows and every statistic
`weighted_affine_quant` scores is a per-(row, group) sum. This pins the claim on
random weights at every width the pipeline emits, at block sizes small enough
that boundaries land in every position.
"""

from __future__ import annotations

import numpy as np
import pytest

from maccelerate import affine


class FakeReader:
    """Minimal stand-in for SourceReader: bf16 rows served from memory."""

    def __init__(self, name, f32):
        self.name = name
        self.header = {name: {"dtype": "BF16", "shape": f32.shape}}
        self._u16 = affine.f32_to_bf16_u16(f32)

    def read_block(self, name, r0, r1):
        assert name == self.name
        return self._u16[r0:r1]


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group_size", [64])
def test_blocked_equals_whole(bits, group_size):
    rng = np.random.default_rng(0)
    out_dim, in_dim = 128, 256
    f32 = (rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.05)
    reader = FakeReader("w", f32)
    # whole path sees the same bf16-rounded values the blocked path streams
    w = affine.bf16_to_f32(reader._u16)

    ch = np.abs(rng.standard_normal(in_dim).astype(np.float32)) + 0.5
    calibrated = bits in affine.CALIBRATED_BITS
    if calibrated:
        ref = affine.weighted_affine_quant(w, bits, group_size, ch)
    else:
        ref = affine.mlx_affine_quant(w, bits, group_size=group_size)
    ref_bytes = tuple(t[2] for t in ref)
    ref_shapes = tuple(t[1] for t in ref)

    for block_mb in (64, 8, 1, 0):
        got = affine.quantize_blocked(reader, "w", bits, group_size,
                                      ch if calibrated else None, block_mb)
        assert tuple(t[1] for t in got) == ref_shapes, (bits, block_mb)
        assert tuple(t[2] for t in got) == ref_bytes, (
            f"bits={bits} block={block_mb}MB differs from whole-tensor")


@pytest.mark.parametrize("bits,gs,in_dim,expect", [
    (6, 64, 5120, (6, 64)),
    (8, 64, 4096, (8, 64)),
    (4, 128, 1024, (4, 128)),
])
def test_solve_geometry(bits, gs, in_dim, expect):
    assert affine.solve_geometry(affine.words_per_row(in_dim, bits),
                                 in_dim // gs, in_dim) == expect


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(1)
    for bits in (2, 3, 4, 8):
        n = 256
        q = rng.integers(0, 1 << bits, size=(4, n), dtype=np.uint8)
        packed = affine.pack_bits(q, bits)
        assert np.array_equal(affine.unpack_bits(packed, bits, n), q)


def test_words_per_row_matches_mx_quantize():
    """The packed column count must agree with MLX's own packing."""
    mx = affine.mx()
    for bits in (2, 3, 4, 5, 6, 8):
        w = mx.random.normal((32, 256)).astype(mx.bfloat16)
        packed, _, _ = mx.quantize(w, group_size=64, bits=bits)
        mx.eval(packed)
        assert packed.shape[1] == affine.words_per_row(256, bits)