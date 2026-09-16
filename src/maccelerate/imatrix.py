"""Official imatrix GGUF -> HF-named SafeTensors imatrix.

Unsloth (and llama.cpp imatrix GGUF generally) publish calibration statistics as
tensors named `<tensor>.in_sum2` plus `<tensor>.counts`. mlx-serve's converter
wants one f32 value per input channel (the mean-squared activation) keyed by the
SOURCE weight name:

    mean_sq[name] = in_sum2[name] / counts[name]

The mapping uses the same family table as the allocation reader, so the two
cannot drift apart. Provenance (`total_tokens`, datasets, revision) is recorded
in the SafeTensors `__metadata__` and surfaced in the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .gguf import Gguf
from .names import detect_family, family_from_names
from .safetensors import write_safetensors_raw


def convert_imatrix(imatrix_path, out_path, revision=None, url=None, arch=None):
    path = Path(imatrix_path)
    gguf = Gguf(path)
    # imatrix GGUFs carry no `general.architecture`; recover the family by name.
    if arch is not None:
        family = family_from_names(list(gguf.tensors), arch=arch)
    else:
        try:
            family = detect_family(gguf)
        except ValueError:
            family = family_from_names(list(gguf.tensors))

    chunk_count = gguf.kv.get("imatrix.chunk_count")
    chunk_size = gguf.kv.get("imatrix.chunk_size")
    datasets = gguf.kv.get("imatrix.datasets") or []
    total_tokens = (int(chunk_count) * int(chunk_size)
                    if chunk_count and chunk_size else None)

    out, skipped, nonfinite = {}, [], []
    for name in list(gguf.tensors):
        if not name.endswith(".in_sum2"):
            continue
        stem = name[:-len(".in_sum2")]
        counts_name = stem + ".counts"
        if counts_name not in gguf.tensors:
            skipped.append(stem)
            continue
        src_name = family.map_name(stem)
        if src_name is None:
            skipped.append(stem)
            continue
        sum2 = gguf.read_f32(name)
        counts = gguf.read_f32(counts_name)
        ncall = float(counts[0]) if counts.size else 0.0
        if ncall <= 0:
            skipped.append(stem)
            continue
        vals = (sum2 / np.float32(ncall)).astype(np.float32)
        if not np.isfinite(vals).all() or (vals < 0).any():
            nonfinite.append(src_name)
            continue
        out[src_name] = ("F32", vals.shape, np.ascontiguousarray(vals).tobytes())
    if nonfinite:
        raise ValueError(f"non-finite/negative imatrix entries: {nonfinite[:5]}")

    metadata = {
        "total_tokens": str(total_tokens if total_tokens is not None else 0),
        "source_file": path.name,
        "source_revision": revision or "",
        "source_url": url or "",
        "datasets": json.dumps(datasets),
        "chunk_count": str(chunk_count or 0),
        "chunk_size": str(chunk_size or 0),
        "entries": str(len(out)),
        "format": "hf-named-in_sum2-over-counts",
    }
    write_safetensors_raw(str(out_path), out, metadata=metadata)
    return {"entries": len(out), "skipped": skipped, "total_tokens": total_tokens,
            "chunk_count": chunk_count, "chunk_size": chunk_size,
            "datasets": list(datasets), "metadata": metadata}