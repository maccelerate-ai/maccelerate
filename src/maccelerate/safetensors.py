"""SafeTensors I/O — raw readers/writers used by the whole pipeline.

Ported from `mlx-serve/tests/convert_dsv4_weights.py` (commit 16a47ec) so
`maccelerate` is standalone. Only the format layer is kept; the MLX quantizer
lives in `maccelerate.affine`.
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np

DTYPE_BYTES = {
    "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1, "U8": 1,
    "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "U32": 4,
}

_NP_DTYPE = {
    "F8_E4M3": np.uint8, "F8_E8M0": np.uint8, "U8": np.uint8,
    "I8": np.int8, "BF16": np.uint16, "F16": np.float16,
    "F32": np.float32, "I32": np.int32, "I64": np.int64, "U32": np.uint32,
}


def read_header(path):
    """(header_dict, data_offset) for one SafeTensors file."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(hlen)), 8 + hlen


class ShardReader:
    """Random-access tensor reader over one SafeTensors file."""

    def __init__(self, path):
        self.path = str(path)
        self.header, self.data_off = read_header(self.path)
        self.header.pop("__metadata__", None)

    def names(self):
        return list(self.header.keys())

    def read(self, name):
        """(numpy array in its raw container dtype, safetensors dtype string)."""
        meta = self.header[name]
        dt, shape = meta["dtype"], meta["shape"]
        begin, end = meta["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.data_off + begin)
            buf = f.read(end - begin)
        nbytes = int(np.prod(shape)) * DTYPE_BYTES[dt] if shape else DTYPE_BYTES[dt]
        assert len(buf) == nbytes, f"{name}: expected {nbytes} bytes, got {len(buf)}"
        arr = np.frombuffer(buf, dtype=_NP_DTYPE[dt]).reshape(shape)
        return arr, dt

    def read_block(self, name, row0, row1):
        """Rows [row0, row1) of a 2-D tensor, avoiding a whole-tensor read."""
        meta = self.header[name]
        dt, shape = meta["dtype"], meta["shape"]
        assert len(shape) == 2, f"{name}: not 2-D ({shape})"
        row_bytes = shape[1] * DTYPE_BYTES[dt]
        begin = meta["data_offsets"][0]
        with open(self.path, "rb") as f:
            f.seek(self.data_off + begin + row0 * row_bytes)
            buf = f.read((row1 - row0) * row_bytes)
        assert len(buf) == (row1 - row0) * row_bytes, f"{name}: short block read"
        return np.frombuffer(buf, dtype=_NP_DTYPE[dt]).reshape(row1 - row0, shape[1])


def index_weight_map(src):
    """{tensor_name: shard_basename} from a sharded HF checkpoint index."""
    idx = json.loads((open(os.path.join(src, "model.safetensors.index.json"))).read())
    return idx["weight_map"]


def write_safetensors_raw(path, tensors, metadata=None):
    """Write tensors given as (dtype_str, shape, raw_bytes) triples."""
    header = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    for name, (dt, shape, raw) in tensors.items():
        header[name] = {"dtype": dt, "shape": list(shape),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    hjson = json.dumps(header).encode()
    pad = (8 - (len(hjson) % 8)) % 8
    hjson += b" " * pad
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for _, (_, _, raw) in tensors.items():
            f.write(raw)
    os.replace(tmp, path)