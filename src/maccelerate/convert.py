"""Streamed converter: HF bf16 source + allocation table -> native MLX pack.

Memory is the design constraint. A 27B model's `embed_tokens`/`lm_head` are
1.27B params, i.e. 5 GB in f32, and the imatrix-weighted search runs at roughly
16x that — so the obvious "read tensor, quantize, write" formulation OOM-kills a
64 GB machine once a few tensors are in flight. Here:

  * weights are quantized in row blocks (`affine.quantize_blocked`), bounded by
    `row_block_mb`; blocks are provably identical to whole-tensor quantization
  * the writer buffers at most `shard_gb` before flushing a shard
  * `jobs=1` runs entirely in-process (no fork, no pickling of GB results)

Measured peak for the 27B / UD-Q6_K_M build: 7.8 GB RSS, 132 s.

The pack layout is mlx-serve's: MLX affine weights (`*.weight` U32 packed +
`*.scales`/`*.biases` bf16), text-only (the vision tower is dropped), MTP kept
inline, delta-encoded norms folded, depthwise conv1d transposed to `[C, K, 1]`.
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from . import affine, alloc as alloc_mod
from .gguf import Gguf
from .names import bytes_for, detect_family
from .safetensors import ShardReader, write_safetensors_raw

COPY_FILES = ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt", "chat_template.jinja",
              "special_tokens_map.json", "added_tokens.json", "LICENSE")


class SourceReader:
    """Block-random-access reader across every shard of an HF checkpoint."""

    def __init__(self, src):
        src = Path(src)
        idx = json.loads((src / "model.safetensors.index.json").read_text())
        self.weight_map = idx["weight_map"]
        self._shards = {}
        self.header = {}          # name -> {"dtype", "shape"}
        for name, shard in self.weight_map.items():
            r = self._shards.get(shard)
            if r is None:
                r = self._shards[shard] = ShardReader(src / shard)
            meta = r.header[name]
            self.header[name] = {"dtype": meta["dtype"], "shape": meta["shape"]}

    def _locate(self, name):
        shard = self.weight_map[name]
        return self._shards[shard], name

    def read_block(self, name, row0, row1):
        r, n = self._locate(name)
        return r.read_block(n, row0, row1)

    def read_raw(self, name):
        r, n = self._locate(name)
        arr, dt = r.read(n)
        return dt, arr.shape, np.ascontiguousarray(arr).tobytes()

    def names(self):
        return list(self.weight_map)


# ---------------------------------------------------------------------------
# worker (fork path)
# ---------------------------------------------------------------------------
_JOB = {}


def _init_worker(src, alloc, imatrix_path, row_block_mb, family):
    _JOB["reader"] = SourceReader(src)
    _JOB["alloc"] = alloc
    _JOB["row_block_mb"] = row_block_mb
    _JOB["family"] = family
    if imatrix_path:
        r = ShardReader(imatrix_path)
        _JOB["im"] = {n: r.read(n)[0] for n in r.names()}
    else:
        _JOB["im"] = {}


def _quantize_one(name):
    reader = _JOB["reader"]
    spec = _JOB["alloc"][name]
    bits, gs = spec["bits"], spec["group_size"]
    ch = _JOB["im"].get(name)
    # `affine.pack_bits` implements MLX's layout for 2/3/4/8 bits; 5- and 6-bit
    # take the plain `mx.quantize` path and are reported as uncalibrated.
    if bits not in affine.CALIBRATED_BITS:
        ch = None
    calibrated = ch is not None
    if calibrated:
        in_dim = reader.header[name]["shape"][1]
        assert ch.shape == (in_dim,), f"{name}: imatrix {ch.shape} vs in {in_dim}"
        ch = np.ascontiguousarray(ch, dtype=np.float32)
    triples = affine.quantize_blocked(reader, name, bits, gs, ch,
                                      _JOB["row_block_mb"])
    return name, bits, gs, calibrated, triples


# ---------------------------------------------------------------------------
# card
# ---------------------------------------------------------------------------
CARD = """---
license: apache-2.0
base_model: {base_model}
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- mlx-serve
- {arch_tag}
- apple-silicon
- mtp
---

# {repo_name}

{headline}

{size_gb:.1f} GB on disk, {bpw:.2f} bits per weight averaged over everything that is quantized. **Text only** — the vision tower is not included.

## Allocation

{method}

{alloc_table}

{pinning}

## Serving

{serving_notes}

## Conversion

{conversion_notes}

## Provenance

```
variant          {variant}
source GGUF      {gguf_file} @ {gguf_revision}
source sha256    {gguf_sha256}
bf16 source      {bf16_repo} @ {bf16_revision}
imatrix          {imatrix_file}
converter        maccelerate {converter}
```
"""


def render_card(dst, card, per_class, *, repo_name, variant, headline, method,
                pinning, serving_notes, conversion_notes, manifest):
    q_bytes = sum(d["bytes"] for d in per_class.values())
    q_params = sum(d["params"] for d in per_class.values())
    text = CARD.format(
        base_model=card.get("base_model", ""),
        arch_tag=card.get("arch_tag", "mlx"),
        repo_name=repo_name, variant=variant, headline=headline, method=method,
        pinning=pinning, serving_notes=serving_notes,
        conversion_notes=conversion_notes,
        size_gb=manifest.get("total_bytes", 0) / 1e9,
        bpw=(q_bytes * 8 / q_params) if q_params else 0.0,
        alloc_table=alloc_mod.alloc_table_markdown(per_class, q_bytes, q_params),
        gguf_file=manifest.get("gguf", {}).get("file", ""),
        gguf_revision=manifest.get("gguf", {}).get("revision", ""),
        gguf_sha256=manifest.get("gguf", {}).get("sha256", ""),
        bf16_repo=manifest.get("bf16_source", {}).get("repo", ""),
        bf16_revision=manifest.get("bf16_source", {}).get("revision", ""),
        imatrix_file=manifest.get("imatrix", {}).get("file", ""),
        converter=manifest.get("converter_commit", "unknown"),
    )
    (dst / "README.md").write_text(text)


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------
def convert(src, dst, alloc_path, *, gguf_path=None, imatrix=None, jobs=3,
            row_block_mb=64, shard_gb=2.0, verify=False, card=None,
            manifest_extra=None, log=print):
    src, dst = Path(src), Path(os.path.expanduser(dst))
    dst.mkdir(parents=True, exist_ok=True)
    doc = json.loads(Path(os.path.expanduser(alloc_path)).read_text())
    alloc, per_class = doc["allocation"], doc["per_class"]
    manifest = dict(doc.get("manifest", {}))
    if manifest_extra:
        manifest.update(manifest_extra)

    # family comes from the GGUF that produced the allocation
    if gguf_path is None:
        raise ValueError("convert() needs gguf_path to resolve the model family")
    family = detect_family(Gguf(gguf_path))

    reader = SourceReader(src)
    order = []
    for name in sorted(reader.names()):
        if name.startswith("model.visual."):
            continue
        order.append((name, "q" if name in alloc else "d"))
    qnames = [n for n, k in order if k == "q"]
    log(f"{len(order)} source tensors ({len(qnames)} quantized, "
        f"{len(order) - len(qnames)} dense); vision tower dropped")

    shard_bytes = int(shard_gb * 1024 ** 3)
    out, out_bytes, out_idx, out_map, total = {}, 0, 0, {}, 0
    stats = {"calibrated": 0, "plain": 0, "shift": 0, "conv": 0, "widths": Counter()}
    verify_fail = []

    def flush():
        nonlocal out, out_bytes, out_idx, total
        if not out:
            return
        out_idx += 1
        fname = f"model-{out_idx:05d}.safetensors"
        write_safetensors_raw(dst / fname, out)
        for k in out:
            out_map[k] = fname
        total += out_bytes
        log(f"  wrote {fname}  {out_bytes / 1e9:.2f} GB  ({len(out)} tensors)")
        out, out_bytes = {}, 0

    def emit(key, triple, nbytes):
        nonlocal out_bytes
        out[key] = triple
        out_bytes += nbytes

    def handle_dense(name):
        nonlocal stats
        dt, shape, raw = reader.read_raw(name)
        nk = family.pack_name(name)
        arr = np.frombuffer(raw, dtype={"BF16": np.uint16, "F16": np.float16,
                                        "F32": np.float32, "I32": np.int32,
                                        "I64": np.int64, "U8": np.uint8}[dt]).reshape(shape)
        if family.conv1d_transpose(nk, arr.ndim):
            arr = np.ascontiguousarray(np.swapaxes(arr, 1, 2))
            stats["conv"] += 1
        if family.needs_norm_shift(nk, arr.ndim):
            assert dt == "BF16", f"{nk}: norm shift on {dt}"
            arr = affine.f32_to_bf16_u16(affine.bf16_to_f32(arr) + 1.0)
            stats["shift"] += 1
        raw = np.ascontiguousarray(arr).tobytes()
        emit(nk, (dt, arr.shape, raw), len(raw))

    if jobs == 1:
        _init_worker(src, alloc, imatrix, row_block_mb, family)
        pool_cm = contextlib.nullcontext()
        qiter = iter(_quantize_one(n) for n in qnames)
    else:
        ctx = mp.get_context("fork")
        pool_cm = ctx.Pool(jobs, initializer=_init_worker,
                           initargs=(src, alloc, imatrix, row_block_mb, family))
        qiter = pool_cm.imap(_quantize_one, qnames, chunksize=1)

    with pool_cm:
        for name, kind in order:
            nk = family.pack_name(name)
            if kind == "d":
                handle_dense(name)
            else:
                got, bits, gs, calibrated, triples = next(qiter)
                assert got == name, f"pool order drifted: {got} != {name}"
                base = nk[:-len(".weight")]
                emit(base + ".weight", triples[0], len(triples[0][2]))
                emit(base + ".scales", triples[1], len(triples[1][2]))
                emit(base + ".biases", triples[2], len(triples[2][2]))
                stats["calibrated" if calibrated else "plain"] += 1
                key = f"{bits}x{gs}"
                stats["widths"][key] += 1
                if verify:
                    in_dim = reader.header[name]["shape"][1]
                    solved = affine.solve_geometry(triples[0][1][-1],
                                                   triples[1][1][-1], in_dim)
                    if solved != (bits, gs):
                        verify_fail.append((nk, solved, (bits, gs)))
            if out_bytes >= shard_bytes:
                flush()
    flush()

    if verify_fail:
        for nk, solved, want in verify_fail[:10]:
            log(f"  VERIFY FAIL {nk}: geometry solves to {solved}, allocated {want}")
        raise SystemExit(f"{len(verify_fail)} weights would not resolve to their "
                         f"allocated width")

    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": out_map}, indent=2))

    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("vision_config", None)
    widths = Counter()
    for spec in alloc.values():
        widths[(spec["bits"], spec["group_size"])] += 1
    dom_bits, dom_gs = max(widths, key=widths.get)
    qb = {"group_size": dom_gs, "bits": dom_bits, "mode": "affine"}
    for name, spec in sorted(alloc.items()):
        qb[family.pack_name(name)[:-len(".weight")]] = {
            "group_size": spec["group_size"], "bits": spec["bits"], "mode": "affine"}
    cfg["quantization"] = qb
    cfg["quantization_config"] = qb
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    for f in COPY_FILES:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)

    manifest["total_bytes"] = total
    manifest["shards"] = out_idx
    manifest["width_histogram"] = dict(sorted(stats["widths"].items()))
    manifest["conversion"] = {
        "wall_tool": "maccelerate.convert.convert",
        "imatrix_weighted_tensors": stats["calibrated"],
        "rtn_tensors": stats["plain"],
        "norm_plus_one_folded": stats["shift"],
        "conv1d_transposed": stats["conv"],
        "double_quantized": False,
        "row_block_mb": row_block_mb,
        "shard_gb": shard_gb,
        "jobs": jobs,
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1))

    if card is not None:
        render_card(dst, card, per_class,
                    repo_name=card.get("repo_name", dst.name), variant=card["variant"],
                    headline=card["headline"], method=card["method"],
                    pinning=card["pinning"], serving_notes=card["serving_notes"],
                    conversion_notes=card["conversion_notes"], manifest=manifest)

    log(f"done: {total / 1e9:.2f} GB in {out_idx} shards")
    log(f"  calibrated {stats['calibrated']}  plain mx.quantize {stats['plain']}  "
        f"norm+1 {stats['shift']}  conv1d transposed {stats['conv']}")
    log("  widths: " + " ".join(f"{k}:{v}" for k, v in sorted(stats["widths"].items())))
    if verify:
        log(f"  verify: all {len(qnames)} quantized weights re-solve to their "
            f"allocated (bits, group_size) from packed geometry alone")
    return manifest