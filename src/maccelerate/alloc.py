"""Build a per-tensor MLX allocation table from an Unsloth Dynamic GGUF.

This is the provenance step. It reads each tensor's ggml type out of the
official GGUF and maps the family to an MLX affine width (see `gguf.TYPE_BITS`),
so the resulting pack reproduces **Unsloth's per-tensor allocation** rather than
an independently measured one. Nothing is re-derived from reconstruction error;
`mlx-serve`'s `qwen38_iq_allocate.py` remains the measured-allocation producer.

Output shape (consumed by `convert.convert` and `mlx-serve`'s converter):

    {"allocation": {src_name: {"bits", "group_size", "ggml_type"}},
     "per_class":  {class: {"params", "bytes", "widths"}},
     "manifest":   {...provenance...}}
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .affine import words_per_row
from .gguf import DENSE_TYPES, SUBSTITUTED, TYPE_BITS, Gguf
from .names import bytes_for, detect_family

CLASS_LABELS = {
    "mlp_gate_up": "MLP gate + up", "mlp_down": "MLP down",
    "gdn_qkv": "GDN in_proj_qkv", "gdn_z": "GDN in_proj_z",
    "gdn_out": "GDN out_proj", "gdn_ab": "GDN a/b gates",
    "attn": "attention q/k/v/o", "lm_head": "lm_head", "embed": "embed_tokens",
    "mtp": "MTP head", "mtp_fc": "MTP concat projection", "dense": "dense (bf16)",
}


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hf_source_shapes(src):
    """{name: shape} over every shard of an HF SafeTensors checkpoint."""
    from .safetensors import read_header

    src = Path(src)
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    shapes = {}
    for shard in sorted(set(idx["weight_map"].values())):
        header, _ = read_header(src / shard)
        for name, meta in header.items():
            if name != "__metadata__":
                shapes[name] = tuple(meta["shape"])
    return shapes


def build_alloc(gguf_path, variant, group_size=64, keep_mtp_fc_dense=False,
                src=None, gguf_url=None, gguf_revision=None, gguf_sha256=None,
                imatrix=None, imatrix_revision=None, hash_gguf=False,
                extra_manifest=None):
    gguf = Gguf(gguf_path)
    family = detect_family(gguf)

    allocation, dense, substituted, unmapped = {}, [], {}, []
    params_of, in_dim_of = {}, {}
    for gname in gguf.tensors:
        tname = gguf.type_name(gname)
        gguf_dims = gguf.tensors[gname][1]
        src_name = family.map_name(gname)
        if src_name is None:
            unmapped.append(gname)
            continue
        # GGUF dims are (in, out, ...) row-major; params are the product either way.
        n_params = 1
        for d in gguf_dims:
            n_params *= int(d)
        params_of[src_name] = n_params
        in_dim_of[src_name] = int(gguf_dims[0]) if gguf_dims else 0
        if tname in DENSE_TYPES:
            dense.append(src_name)
            continue
        if tname not in TYPE_BITS:
            raise ValueError(f"{gname}: unsupported ggml type {tname}")
        if tname in SUBSTITUTED:
            substituted[src_name] = {"from": tname, "bits": TYPE_BITS[tname]}
        allocation[src_name] = {"bits": TYPE_BITS[tname], "group_size": group_size,
                               "ggml_type": tname}
    if unmapped:
        raise ValueError(f"{len(unmapped)} GGUF tensors did not map, "
                         f"e.g. {unmapped[:5]}")

    # The loader can read a quantized concat projection, and the official UD
    # GGUF ships it quantized; the sanctioned BF16-head reference keeps it dense.
    if keep_mtp_fc_dense:
        fc = "mtp.fc.weight"
        if allocation.pop(fc, None) is not None and fc not in dense:
            dense.append(fc)

    # geometry: the GGUF's ne[0] is the input dim, so divisibility is checkable
    # without the bf16 source; `--src` additionally proves name closure.
    bad = []
    for name, spec in allocation.items():
        in_dim = in_dim_of[name]
        if in_dim % spec["group_size"]:
            bad.append((name, in_dim, spec["group_size"]))
        elif words_per_row(in_dim, spec["bits"]) * 32 != in_dim * spec["bits"]:
            bad.append((name, in_dim, f"bits={spec['bits']} not packable"))
    closure = None
    if src:
        shapes = hf_source_shapes(src)
        nonvision = {k for k in shapes if not k.startswith("model.visual.")}
        missing = sorted(set(allocation) - nonvision)
        extra = sorted(nonvision - set(allocation) - set(dense))
        if missing or extra:
            raise ValueError(
                f"closure failed: {len(missing)} allocated names absent from source "
                f"(e.g. {missing[:3]}); {len(extra)} source tensors neither allocated "
                f"nor dense (e.g. {extra[:3]})")
        for name in allocation:
            if int(shapes[name][-1]) != in_dim_of[name]:
                bad.append((name, in_dim_of[name], f"src in-dim {shapes[name][-1]}"))
        closure = {"hf_tensors": len(shapes), "nonvision": len(nonvision),
                   "quantized": len(allocation), "dense": len(dense)}
    if bad:
        raise ValueError(f"geometry does not solve: {bad[:5]}")

    per_class = {}
    for name, spec in allocation.items():
        cls = family.classify(name)[0]
        d = per_class.setdefault(cls, {"params": 0, "bytes": 0, "widths": {}})
        p = params_of[name]
        d["params"] += p
        d["bytes"] += bytes_for(p, spec["bits"], spec["group_size"])
        k = f"{spec['bits']}x{spec['group_size']}"
        d["widths"][k] = d["widths"].get(k, 0) + 1

    type_hist = dict(Counter(s["ggml_type"] for s in allocation.values()))
    width_hist = dict(Counter(f"{s['bits']}x{s['group_size']}"
                              for s in allocation.values()))

    manifest = {
        "variant": variant,
        "gguf": {"file": Path(gguf_path).name, "url": gguf_url,
                 "revision": gguf_revision, "sha256": gguf_sha256,
                 "size_bytes": Path(gguf_path).stat().st_size,
                 "gguf_version": gguf.version, "n_tensors": len(gguf.tensors),
                 "architecture": gguf.arch, "file_type": gguf.file_type()},
        "imatrix": {"file": imatrix, "revision": imatrix_revision},
        "family": {"name": family.arch, "num_layers": family.num_layers,
                   "mtp_index": family.mtp_index},
        "group_size": group_size,
        "mtp_fc_dense_pinned": bool(keep_mtp_fc_dense),
        "ggml_type_histogram": type_hist,
        "allocated_width_histogram": width_hist,
        "substituted_types": substituted,
        "quantized_tensors": len(allocation),
        "dense_tensors": len(dense),
        "closure": closure,
        "build": {"producer": "maccelerate.alloc.build_alloc"},
    }
    if hash_gguf and not gguf_sha256:
        manifest["gguf"]["sha256"] = sha256_file(gguf_path)
    if extra_manifest:
        manifest.update(extra_manifest)

    return {"allocation": allocation, "per_class": per_class, "manifest": manifest}


def alloc_table_markdown(per_class, quant_bytes, quant_params):
    rows = ["| weight class | params | on disk | widths |", "|---|---|---|---|"]
    for cls in sorted(per_class, key=lambda c: -per_class[c]["bytes"]):
        d = per_class[cls]
        if d["bytes"] == 0:
            continue
        widths = ", ".join(f"{k.replace('x', '-bit/gs-')} x{v}"
                           for k, v in sorted(d["widths"].items()))
        rows.append(f"| {CLASS_LABELS.get(cls, cls)} | {d['params']/1e9:.2f}B | "
                    f"{d['bytes']/1e9:.2f} GB | {widths} |")
    rows.append(f"| **total quantized** | **{quant_params/1e9:.2f}B** | "
                f"**{quant_bytes/1e9:.2f} GB** | |")
    return "\n".join(rows)


def effective_bpw(per_class):
    q_bytes = sum(d["bytes"] for d in per_class.values())
    q_params = sum(d["params"] for d in per_class.values())
    return (q_bytes * 8 / q_params) if q_params else 0.0