"""Minimal GGUF reader (stdlib only) — metadata, tensor type table, tensor data.

Deliberately dependency-free: the whole point is to read an official Unsloth
Dynamic GGUF's *per-tensor type table* (`general.architecture`, each tensor's
ggml type) without pulling in the `gguf` package or loading any weights.

Also holds the ggml-type -> MLX-affine-width policy, which is the one modelling
decision in the pipeline: Unsloth's Dynamic allocator decides per-tensor types
(k-quant / IQ families); we keep each tensor's *bit count* and re-encode on
MLX's affine grid. MLX affine cannot reproduce llama.cpp's codebooks, so this
is "same allocation", never numerical parity.
"""

from __future__ import annotations

import struct
from collections import Counter
from pathlib import Path

import numpy as np

# llama.cpp ggml_type enum (stable across GGUF v3).
GGML_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
}

# ggml type -> MLX affine bit width. The family gives the width; the grid is MLX's.
TYPE_BITS = {
    "Q2_K": 2, "IQ2_XXS": 2, "IQ2_XS": 2, "IQ2_S": 2, "IQ1_S": 2, "IQ1_M": 2,
    "Q3_K": 3, "IQ3_XXS": 3, "IQ3_S": 3,
    "Q4_0": 4, "Q4_1": 4, "Q4_K": 4, "IQ4_NL": 4, "IQ4_XS": 4,
    "Q5_0": 5, "Q5_1": 5, "Q5_K": 5,
    "Q6_K": 6,
    "Q8_0": 8, "Q8_1": 8, "Q8_K": 8,
}
DENSE_TYPES = {"F32", "F16", "BF16"}
# Stored at a width MLX affine has no equivalent for; the pack substitutes up.
SUBSTITUTED = {"IQ1_S", "IQ1_M", "Q8_K"}

_FIXED = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
          5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8),
          11: ("<q", 8), 12: ("<d", 8)}


class Gguf:
    """Header + tensor type table + (optionally) tensor data of one GGUF file."""

    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError(f"{path}: not a GGUF file")
            self.version = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]

            def rd_str():
                n = struct.unpack("<Q", f.read(8))[0]
                return f.read(n).decode("utf-8", "replace")

            def rd_val(t):
                if t == 8:
                    return rd_str()
                if t == 9:
                    et = struct.unpack("<I", f.read(4))[0]
                    n = struct.unpack("<Q", f.read(8))[0]
                    return [rd_val(et) for _ in range(n)]
                fmt, sz = _FIXED[t]
                return struct.unpack(fmt, f.read(sz))[0]

            self.kv = {}
            for _ in range(n_kv):
                k = rd_str()
                t = struct.unpack("<I", f.read(4))[0]
                self.kv[k] = rd_val(t)

            # name -> (type_id, dims, offset_relative_to_data)
            self.tensors = {}
            for _ in range(n_tensors):
                name = rd_str()
                nd = struct.unpack("<I", f.read(4))[0]
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
                tt = struct.unpack("<I", f.read(4))[0]
                off = struct.unpack("<Q", f.read(8))[0]
                self.tensors[name] = (tt, dims, off)
            end = f.tell()

        align = int(self.kv.get("general.alignment", 32)) or 32
        self.data_start = end + (-end % align)

    # -- metadata helpers -------------------------------------------------
    @property
    def arch(self):
        return self.kv.get("general.architecture")

    def meta(self, suffix, arch=True):
        """`qwen35.embedding_length` style lookup; arch-prefixed by default."""
        return self.kv.get(f"{self.arch}.{suffix}" if arch else suffix)

    def type_name(self, name):
        return GGML_NAMES.get(self.tensors[name][0], f"type{self.tensors[name][0]}")

    def file_type(self):
        return self.kv.get("general.file_type")

    def type_histogram(self):
        return dict(Counter(self.type_name(n) for n in self.tensors))

    def as_dict(self):
        return {n: {"type": self.type_name(n), "dims": list(d)}
                for n, (_, d, _) in self.tensors.items()}

    # -- data -------------------------------------------------------------
    def read_f32(self, name):
        """Read one F32 tensor's values (used for the imatrix's in_sum2/counts)."""
        tt, dims, off = self.tensors[name]
        if GGML_NAMES.get(tt) != "F32":
            raise ValueError(f"{name}: expected F32, got {GGML_NAMES.get(tt, tt)}")
        n = int(np.prod(dims)) if dims else 1
        with open(self.path, "rb") as f:
            f.seek(self.data_start + off)
            raw = f.read(n * 4)
        if len(raw) != n * 4:
            raise ValueError(f"{name}: short read")
        return np.frombuffer(raw, dtype=np.float32).copy()

    def nbytes(self, name):
        """Byte length of a tensor from its dims (types here are F32/F16/BF16)."""
        tt, dims, _ = self.tensors[name]
        t = GGML_NAMES.get(tt, "")
        size = {"F32": 4, "F16": 2, "BF16": 2}.get(t)
        if size is None:
            raise ValueError(f"{name}: unsupported data type {t}")
        return int(np.prod(dims)) * size