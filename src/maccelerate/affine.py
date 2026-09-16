"""MLX affine quantization core: packing, RTN, and the imatrix-weighted search.

Ported verbatim (function bodies unchanged) from `mlx-serve`:
  * `pack_bits` / `unpack_bits` / `dequant_np` / `_mlx_minmax_sb` / `_bf16_round`
    / `weighted_affine_quant` — `tests/dsv4_imatrix.py` (commit 16a47ec)
  * `bf16_to_f32` / `f32_to_bf16_u16` / `mlx_affine_quant` — `tests/convert_dsv4_weights.py`

Two paths emit the identical triple — `(("U32", shape, bytes), ("BF16", …),
("BF16", …))` — ready for `safetensors.write_safetensors_raw`:

  plain   `mlx_affine_quant(w, bits, gs)`      -> `mx.quantize` (the reference)
  weighted `weighted_affine_quant(w, bits, gs, ch)` -> activation-weighted search

Byte packing is MLX's, not llama.cpp's: `pack_bits` covers 2/3/4/8 bits, which
is why a 5- or 6-bit weight must take the plain path (see `quantize_blocked`).

`quantize_blocked` streams rows so a 1.27B-param tensor (5 GB as f32, and far
more inside the weighted search) is never resident — the property that makes a
27B conversion safe on a 64 GB machine.
"""

from __future__ import annotations

import numpy as np

_mx = None


def mx():
    """Lazy mlx import; the pure-numpy parts must work without a GPU."""
    global _mx
    if _mx is None:
        import mlx.core

        _mx = mlx.core
        _mx.set_default_device(_mx.cpu)
    return _mx


# ---------------------------------------------------------------------------
# bf16 helpers
# ---------------------------------------------------------------------------
def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_u16(f32):
    """Round-to-nearest-even f32 -> bf16 bit pattern (matches hardware/mlx)."""
    u = f32.astype(np.float32).view(np.uint32)
    rounded = u + 0x7FFF + ((u >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def _bf16_round(x):
    return bf16_to_f32(f32_to_bf16_u16(x.astype(np.float32)))


# ---------------------------------------------------------------------------
# MLX affine byte layout
# ---------------------------------------------------------------------------
def pack_bits(q, bits):
    """q: integer array [..., in_dim] in [0, 2^bits-1] -> uint32 array matching
    `mx.quantize` (mlx/backend/cpu/quantized.cpp `quantize<T,U>`)."""
    lead = q.shape[:-1]
    n = q.shape[-1]
    if bits in (2, 4, 8):
        per = 32 // bits
        assert n % per == 0, f"in_dim {n} not divisible by {per}"
        g = q.reshape(-1, per).astype(np.uint32)
        word = np.zeros(g.shape[0], dtype=np.uint32)
        for k in range(per):
            word |= g[:, k] << np.uint32(k * bits)
        return word.reshape(*lead, n // per)
    if bits == 3:
        assert n % 8 == 0 and (n * 3 // 8) % 4 == 0, f"in_dim {n} unpackable at 3 bits"
        g = q.reshape(-1, 8).astype(np.uint32)
        w24 = np.zeros(g.shape[0], dtype=np.uint32)
        for k in range(8):
            w24 |= g[:, k] << np.uint32(3 * k)
        b = np.empty((g.shape[0], 3), dtype=np.uint8)
        b[:, 0] = w24 & 0xFF
        b[:, 1] = (w24 >> 8) & 0xFF
        b[:, 2] = (w24 >> 16) & 0xFF
        return np.ascontiguousarray(b.reshape(*lead, n * 3 // 8)).view(np.uint32)
    raise ValueError(f"unsupported bits {bits}")


def unpack_bits(packed_u32, bits, in_dim):
    """Inverse of pack_bits."""
    lead = packed_u32.shape[:-1]
    if bits in (2, 4, 8):
        per = 32 // bits
        w = packed_u32.reshape(-1, 1)
        shifts = np.uint32(bits) * np.arange(per, dtype=np.uint32)
        q = (w >> shifts) & np.uint32((1 << bits) - 1)
        return q.reshape(*lead, in_dim).astype(np.uint8)
    if bits == 3:
        by = np.ascontiguousarray(packed_u32).view(np.uint8).reshape(-1, 3)
        w24 = by[:, 0].astype(np.uint32) | (by[:, 1].astype(np.uint32) << 8) \
            | (by[:, 2].astype(np.uint32) << 16)
        shifts = np.uint32(3) * np.arange(8, dtype=np.uint32)
        q = (w24.reshape(-1, 1) >> shifts) & np.uint32(7)
        return q.reshape(*lead, in_dim).astype(np.uint8)
    raise ValueError(f"unsupported bits {bits}")


def dequant_np(packed_u32, scales_u16, biases_u16, bits, group_size):
    """Numpy dequant of a packed triple: q*s + b in f32."""
    q = unpack_bits(packed_u32, bits, scales_u16.shape[-1] * group_size).astype(np.float32)
    s = bf16_to_f32(scales_u16)
    b = bf16_to_f32(biases_u16)
    return q.reshape(*s.shape, group_size) * s[..., None] + b[..., None]


def words_per_row(in_dim, bits):
    return in_dim * bits // 32


# ---------------------------------------------------------------------------
# plain RTN path (mx.quantize)
# ---------------------------------------------------------------------------
def mlx_affine_quant(w_f32, bits, group_size=64):
    """f32 -> (packed u32, bf16 scales, bf16 biases) via mx.quantize on bf16 input."""
    m = mx()
    wb = m.array(f32_to_bf16_u16(np.ascontiguousarray(w_f32)).view(np.uint16)).view(m.bfloat16)
    wb = wb.reshape(w_f32.shape)
    wq, scales, biases = m.quantize(wb, group_size=group_size, bits=bits)
    m.eval(wq, scales, biases)
    out = (
        ("U32", wq.shape, np.array(wq, copy=False).tobytes()),
        ("BF16", scales.shape, np.array(scales.view(m.uint16), copy=False).tobytes()),
        ("BF16", biases.shape, np.array(biases.view(m.uint16), copy=False).tobytes()),
    )
    del wq, scales, biases, wb
    return out


# ---------------------------------------------------------------------------
# activation-weighted path
# ---------------------------------------------------------------------------
def _mlx_minmax_sb(X, n_bins):
    """MLX's own affine (s, b) per group. X: [..., gs] f32 -> (s, b) each [...]."""
    xmin = X.min(-1)
    xmax = X.max(-1)
    mask = np.abs(xmin) > np.abs(xmax)
    scale = np.maximum((xmax - xmin) / n_bins, np.float32(1e-7))
    scale = np.where(mask, scale, -scale)
    edge = np.where(mask, xmin, xmax)
    q0 = np.rint(edge / scale)
    nz = q0 != 0
    scale = np.where(nz, edge / np.where(nz, q0, 1.0), scale)
    bias = np.where(nz, edge, np.float32(0.0))
    return scale.astype(np.float32), bias.astype(np.float32)


def weighted_affine_quant(w_f32, bits, group_size, ch_weights,
                          nstep=14, refine=3, return_stats=False):
    """f32 [out, in] + per-input-channel weights [in] -> triples, same contract
    as `mlx_affine_quant`.

    Search: MLX minmax (s, b) as candidate 0, an iscale multi-start sweep, a
    weighted-least-squares refit per labeling, then alternating refinement.
    Every statistic is a sum within one (row, group) — the function is
    row-independent, which is what legitimises `quantize_blocked`."""
    out_dim, in_dim = w_f32.shape
    assert in_dim % group_size == 0
    G = in_dim // group_size
    n_bins = float((1 << bits) - 1)

    X = np.ascontiguousarray(w_f32, dtype=np.float32).reshape(out_dim, G, group_size)
    om = np.asarray(ch_weights, dtype=np.float32)
    assert om.shape == (in_dim,)
    floor = float(om.mean()) * 1e-4 + 1e-30
    W = (om + floor).reshape(1, G, group_size)

    sw = W.sum(-1)
    swx = (W * X).sum(-1)
    swx2 = (W * X * X).sum(-1)

    best_err = np.full((out_dim, G), np.inf, dtype=np.float32)
    best_s = np.ones((out_dim, G), dtype=np.float32)
    best_b = np.zeros((out_dim, G), dtype=np.float32)

    def q_sums(q):
        Wq = W * q
        return Wq.sum(-1), (Wq * q).sum(-1), (Wq * X).sum(-1)

    def closed_err(s, b, sl, sl2, sxl):
        return (swx2 + s * s * sl2 + b * b * sw + 2 * s * b * sl
                - 2 * s * sxl - 2 * b * swx)

    def consider(s, b, sl, sl2, sxl):
        nonlocal best_err, best_s, best_b
        err = closed_err(s, b, sl, sl2, sxl)
        upd = err < best_err
        if upd.any():
            best_err = np.where(upd, err, best_err)
            best_s = np.where(upd, s, best_s)
            best_b = np.where(upd, b, best_b)

    def refit(sl, sl2, sxl):
        D = sw * sl2 - sl * sl
        ok = D > 0
        Dsafe = np.where(ok, D, 1.0)
        s = np.where(ok, (sw * sxl - swx * sl) / Dsafe, np.nan)
        b = np.where(ok, (sl2 * swx - sl * sxl) / Dsafe, np.nan)
        return s, b, ok

    def nonlocal_update(upd, err, s, b):
        nonlocal best_err, best_s, best_b
        best_err = np.where(upd, err, best_err)
        best_s = np.where(upd, s, best_s)
        best_b = np.where(upd, b, best_b)

    def consider_refit(sl, sl2, sxl):
        s, b, ok = refit(sl, sl2, sxl)
        if ok.any():
            err = closed_err(s, b, sl, sl2, sxl)
            upd = ok & (err < best_err)
            if upd.any():
                nonlocal_update(upd, err, s, b)

    s0, b0 = _mlx_minmax_sb(X, n_bins)
    q = np.clip(np.rint((X - b0[..., None]) / s0[..., None]), 0, n_bins)
    sl, sl2, sxl = q_sums(q)
    consider(s0, b0, sl, sl2, sxl)
    consider_refit(sl, sl2, sxl)

    xmin = X.min(-1)
    span = X.max(-1) - xmin
    span_safe = np.where(span > 0, span, 1.0)
    for k in range(nstep):
        iscale = (n_bins - 1.0 + 0.25 * k) / span_safe
        q = np.clip(np.rint((X - xmin[..., None]) * iscale[..., None]), 0, n_bins)
        sl, sl2, sxl = q_sums(q)
        consider_refit(sl, sl2, sxl)

    for _ in range(refine):
        q = np.clip(np.rint((X - best_b[..., None]) / best_s[..., None]), 0, n_bins)
        sl, sl2, sxl = q_sums(q)
        consider_refit(sl, sl2, sxl)

    sb = _bf16_round(best_s)
    bb = _bf16_round(best_b)
    dead = (sb == 0) | ~np.isfinite(sb) | ~np.isfinite(bb)
    if dead.any():
        sb = np.where(dead, np.float32(1.0), sb)
        bb = np.where(dead, _bf16_round(swx / sw), bb)
    q = np.clip(np.rint((X - bb[..., None]) / sb[..., None]), 0, n_bins)
    q = q.astype(np.uint8).reshape(out_dim, in_dim)

    packed = pack_bits(q, bits)
    triples = (
        ("U32", packed.shape, packed.tobytes()),
        ("BF16", (out_dim, G), f32_to_bf16_u16(sb).tobytes()),
        ("BF16", (out_dim, G), f32_to_bf16_u16(bb).tobytes()),
    )
    if not return_stats:
        return triples
    qf = q.reshape(out_dim, G, group_size).astype(np.float32)
    resid = sb[..., None] * qf + bb[..., None] - X
    werr = float((W * resid * resid).sum())
    wnorm = float(swx2.sum())
    return triples, {"weighted_err": werr, "weighted_rel_err": werr / max(wnorm, 1e-30)}


# ---------------------------------------------------------------------------
# streaming, memory-bounded quantization
# ---------------------------------------------------------------------------
CALIBRATED_BITS = (2, 3, 4, 8)  # widths `pack_bits` implements


def quantize_blocked(reader, name, bits, gs, ch, row_block_mb=64):
    """Quantize a 2-D weight by streaming row blocks.

    Returns the usual triples carrying the full-tensor shape, but a whole tensor
    and its f32 copy are never both resident.

    Row blocking is exact: an MLX affine group never spans output rows, every
    statistic `weighted_affine_quant` scores is a sum over a single
    (row, group), and its only cross-row input is `ch`. Concatenating the
    per-block row-major bytes therefore reproduces the whole-tensor result byte
    for byte (pinned by `tests/test_blocked_quant.py`).

    `bits` outside CALIBRATED_BITS falls back to `mlx_affine_quant`, even when
    `ch` is supplied — `pack_bits` has no 5/6-bit layout.
    """
    meta = reader.header[name]
    out_dim, in_dim = meta["shape"]
    assert in_dim % gs == 0, f"{name}: in_dim {in_dim} not divisible by group_size {gs}"
    use_calib = ch is not None and bits in CALIBRATED_BITS
    rows = max(1, (row_block_mb * 1024 * 1024) // (in_dim * 4))

    packed, scales, biases = [], [], []
    for r0 in range(0, out_dim, rows):
        r1 = min(out_dim, r0 + rows)
        blk = reader.read_block(name, r0, r1)
        blk = bf16_to_f32(blk) if meta["dtype"] == "BF16" else blk.astype(np.float32)
        if use_calib:
            t = weighted_affine_quant(blk, bits, gs, ch)
        else:
            t = mlx_affine_quant(blk, bits, group_size=gs)
        packed.append(t[0][2])
        scales.append(t[1][2])
        biases.append(t[2][2])
        del blk, t

    wpr = words_per_row(in_dim, bits)
    g = in_dim // gs
    packed_bytes = b"".join(packed)
    scales_bytes = b"".join(scales)
    biases_bytes = b"".join(biases)
    assert len(packed_bytes) == out_dim * wpr * 4, "packed block geometry drift"
    assert len(scales_bytes) == out_dim * g * 2, "scale block geometry drift"
    assert len(biases_bytes) == out_dim * g * 2, "bias block geometry drift"
    return (("U32", (out_dim, wpr), packed_bytes),
            ("BF16", (out_dim, g), scales_bytes),
            ("BF16", (out_dim, g), biases_bytes))


def solve_geometry(w_cols, s_cols, in_dim):
    """mlx-serve's `affineParamsFromGeometry`: (bits, group_size) from packed
    shape alone. Returns None where the engine would return null."""
    if w_cols == 0 or s_cols == 0 or in_dim == 0:
        return None
    if (w_cols * 32) % in_dim != 0 or in_dim % s_cols != 0:
        return None
    bits = (w_cols * 32) // in_dim
    gs = in_dim // s_cols
    if bits not in (2, 3, 4, 5, 6, 8) or gs not in (32, 64, 128):
        return None
    return bits, gs