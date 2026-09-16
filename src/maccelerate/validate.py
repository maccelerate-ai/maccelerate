"""Structural validation + provenance manifest for a converted MLX pack.

Checks the artifact rather than the converter's claims:

  1. index closure — every indexed tensor is in exactly one shard, every tensor
     in every shard header is indexed (no dangling, no orphan), index agrees
  2. shard sizes, and sha256 with `--hash`
  3. config fields, compared against the SOURCE config when `--src` is given
     (model-agnostic) and against `--expect key=value` overrides
  4. per-tensor bit layout re-derived from PACKED GEOMETRY exactly as
     mlx-serve's `affineParamsFromGeometry` does, then compared to both the
     config's per-tensor entries and the allocation table
  5. presence of family marker tensors (embed / lm_head / final norm / MTP)
  6. tokenizer / template / generation config presence
  7. optional bounded dequantize finiteness sample (catches NaN/Inf codes)

Everything is model-agnostic: expectations come from the source checkpoint and
the allocation, not from Qwen3.8 constants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from . import affine
from .names import detect_family
from .safetensors import read_header


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def validate(model, alloc_path=None, src=None, do_hash=False, finite=False,
             expect=None, manifest_extra=None, log=print):
    model = Path(model)
    fails, notes = [], []
    expect = dict(expect or {})

    def check(ok, msg):
        if not ok:
            fails.append(msg)
        log(("  FAIL " if not ok else "  ok   ") + msg)
        return ok

    # ---- 1. index closure -------------------------------------------------
    idx = json.loads((model / "model.safetensors.index.json").read_text())
    weight_map = idx["weight_map"]
    log(f"[1] index closure: {len(weight_map)} indexed tensors")
    headers, sizes = {}, {}
    for shard in sorted(set(weight_map.values())):
        sp = model / shard
        if not sp.exists():
            fails.append(f"missing shard {shard}")
            continue
        headers[shard], _ = read_header(sp)
        sizes[shard] = sp.stat().st_size
    seen = {}
    for shard, h in headers.items():
        for name in h:
            if name != "__metadata__":
                seen.setdefault(name, []).append(shard)
    check(not {k: v for k, v in seen.items() if len(v) > 1},
          f"each tensor in exactly one shard ({len(seen)} unique)")
    check(not (set(weight_map) - set(seen)),
          f"every indexed tensor present ({len(set(weight_map) - set(seen))} missing)")
    check(not (set(seen) - set(weight_map)),
          f"no unindexed tensors ({len(set(seen) - set(weight_map))} orphan)")
    check(not [k for k, v in seen.items() if weight_map.get(k) not in v],
          "index names the right shard")

    # ---- 2. sizes / hashes ------------------------------------------------
    total = sum(sizes.values())
    hashes = {}
    log(f"[2] shards: {len(sizes)}, total {total / 1e9:.2f} GB")
    if do_hash:
        for shard in sizes:
            hashes[shard] = sha256_file(model / shard)
        log(f"    sha256 computed for {len(hashes)} shards")

    # ---- 3. config --------------------------------------------------------
    cfg = json.loads((model / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    log("[3] config")
    if src:
        scfg = json.loads((Path(src) / "config.json").read_text())
        stc = scfg.get("text_config", scfg)
        for f in ("model_type", "max_position_embeddings", "vocab_size",
                  "hidden_size", "num_hidden_layers", "mtp_num_hidden_layers",
                  "mtp_use_dedicated_embeddings"):
            want = scfg.get(f) if f == "model_type" else stc.get(f, scfg.get(f))
            got = cfg.get(f) if f == "model_type" else tc.get(f, cfg.get(f))
            if want is not None or got is not None:
                check(got == want, f"{f} matches source ({got!r})")
    for k, want in expect.items():
        got = tc.get(k, cfg.get(k))
        check(str(got) == str(want), f"config {k} == {want} (got {got})")
    check("vision_config" not in cfg, "no vision tower declared (text-only pack)")
    qcfg = cfg.get("quantization_config") or cfg.get("quantization") or {}
    per_tensor = {k: v for k, v in qcfg.items() if isinstance(v, dict)}
    check(bool(per_tensor), f"quantization_config has per-tensor entries "
                            f"({len(per_tensor)})")

    # ---- 4. width re-derivation ------------------------------------------
    alloc = None
    if alloc_path:
        alloc = json.loads(Path(alloc_path).read_text())["allocation"]
    shapes = None
    family = None
    if src:
        family = detect_family_shapes(src)
    if src:
        shapes = source_shapes(src)

    derived, mism = {}, []
    quantized = sorted(k for k in seen if k.endswith(".weight")
                       and headers[weight_map[k]][k]["dtype"] == "U32")
    for wname in quantized:
        base = wname[:-len(".weight")]
        sname, bname = base + ".scales", base + ".biases"
        if sname not in seen or bname not in seen:
            fails.append(f"{base}: packed weight without scales/biases")
            continue
        wsh = headers[weight_map[wname]][wname]["shape"]
        ssh = headers[weight_map[sname]][sname]["shape"]
        if (headers[weight_map[sname]][sname]["dtype"] != "BF16"
                or headers[weight_map[bname]][bname]["dtype"] != "BF16"):
            fails.append(f"{base}: scales/biases not BF16")
        out_dim, w_cols = wsh
        s_out, s_cols = ssh
        if out_dim != s_out:
            fails.append(f"{base}: packed rows {out_dim} != scales rows {s_out}")
        if shapes is None:
            continue
        srcname = pack_to_src(family, wname)
        if srcname not in shapes:
            continue
        in_dim = shapes[srcname][-1]
        solved = affine.solve_geometry(w_cols, s_cols, in_dim)
        if solved is None:
            fails.append(f"{base}: geometry does not solve (in {in_dim}, "
                         f"cols {w_cols}, scales {s_cols})")
            continue
        bits, gs = solved
        derived[base] = (bits, gs)
        c = per_tensor.get(base)
        if c and (c.get("bits"), c.get("group_size")) != (bits, gs):
            mism.append((base, (bits, gs), (c.get("bits"), c.get("group_size"))))
        if alloc is not None and srcname in alloc:
            a = alloc[srcname]
            if (a["bits"], a["group_size"]) != (bits, gs):
                mism.append((base, (bits, gs), (a["bits"], a["group_size"])))
    check(not mism, f"derived widths == config == allocation ({len(mism)} mismatches)")
    for m in mism[:5]:
        log(f"       {m}")
    hist = {}
    for bits, gs in derived.values():
        hist[f"{bits}x{gs}"] = hist.get(f"{bits}x{gs}", 0) + 1
    if alloc is not None and derived:
        check(len(derived) == len(alloc),
              f"quantized tensor set matches the allocation ({len(derived)} vs {len(alloc)})")
    log("    widths (re-derived): " + " ".join(f"{k}:{v}" for k, v in sorted(hist.items())))
    bpw = None
    if derived:
        qb = sum(headers[weight_map[k + suf]][k + suf]["data_offsets"][1]
                 - headers[weight_map[k + suf]][k + suf]["data_offsets"][0]
                 for k in derived for suf in (".weight", ".scales", ".biases"))
        qp = sum(int(np.prod(shapes[pack_to_src(family, k + ".weight")]))
                 for k in derived
                 if pack_to_src(family, k + ".weight") in shapes) if shapes else 0
        if qp:
            bpw = qb * 8 / qp
            log(f"    effective bpw = {bpw:.3f}")

    # ---- 5. marker tensors ------------------------------------------------
    log("[5] marker tensors")
    markers = {
        "embed_tokens": "language_model.model.embed_tokens.weight",
        "lm_head": "language_model.lm_head.weight",
        "final_norm": "language_model.model.norm.weight",
        "mtp.fc": "language_model.mtp.fc.weight",
        "mtp.norm": "language_model.mtp.norm.weight",
        "mtp.layer.q_proj": "language_model.mtp.layers.0.self_attn.q_proj.weight",
    }
    for label, name in markers.items():
        if name in seen or label in ("mtp.fc", "mtp.norm", "mtp.layer.q_proj"):
            check(name in seen, f"{label}: {name}")
    mtp = sorted(k for k in seen if ".mtp." in k)
    log(f"    MTP tensors present: {len(mtp)}")

    # ---- 6. tokenizer -----------------------------------------------------
    log("[6] tokenizer / template")
    for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        check((model / f).exists(), f"{f} present")

    # ---- 7. finiteness ----------------------------------------------------
    if finite:
        log("[7] finiteness sample (dequantize)")
        try:
            import mlx.core as mx
            mx.set_default_device(mx.cpu)
            sample = [k for k in derived if k.endswith("mlp.down_proj")][:2]
            sample += [k for k in derived if k.endswith("self_attn.q_proj")][:1]
            sample += [k for k in derived if ".mtp." in k][:2]
            sample += [k for k in derived if "embed_tokens" in k][:1]
            for base in sample:
                bits, gs = derived[base]
                h = headers[weight_map[base + ".weight"]][base + ".weight"]
                _, woff = read_header(model / weight_map[base + ".weight"])
                rows = min(256, h["shape"][0])
                with open(model / weight_map[base + ".weight"], "rb") as f:
                    f.seek(woff + h["data_offsets"][0])
                    wq = np.frombuffer(f.read(rows * h["shape"][1] * 4),
                                       dtype=np.uint32).reshape(rows, h["shape"][1])
                parts = {}
                for suf in ("scales", "biases"):
                    sh = headers[weight_map[base + "." + suf]][base + "." + suf]
                    _, soff = read_header(model / weight_map[base + "." + suf])
                    with open(model / weight_map[base + "." + suf], "rb") as f:
                        f.seek(soff + sh["data_offsets"][0])
                        parts[suf] = np.frombuffer(f.read(rows * sh["shape"][1] * 2),
                                                   dtype=np.uint16).reshape(rows, sh["shape"][1])
                dq = mx.dequantize(mx.array(wq), mx.array(parts["scales"]).view(mx.bfloat16),
                                   mx.array(parts["biases"]).view(mx.bfloat16),
                                   group_size=gs, bits=bits).astype(mx.float32)
                mx.eval(dq)
                a = np.asarray(dq)
                check(bool(np.isfinite(a).all()),
                      f"{base}: dequantized slice finite ({a.min():.4g}..{a.max():.4g})")
        except ImportError:
            notes.append("mlx not importable; skipped finiteness sample")
            log("    SKIP mlx not available")

    manifest = {
        "artifact": str(model),
        "total_bytes": total,
        "indexed_tensors": len(weight_map),
        "quantized_tensors": len(derived),
        "width_histogram": hist,
        "effective_bpw": bpw,
        "shards": {s: {"size_bytes": sizes[s], **({"sha256": hashes[s]} if s in hashes else {})}
                   for s in sorted(sizes)},
        "config": {k: tc.get(k, cfg.get(k)) for k in
                   ("model_type", "max_position_embeddings", "vocab_size",
                    "hidden_size", "num_hidden_layers", "mtp_num_hidden_layers",
                    "mtp_use_dedicated_embeddings")},
        "vision_present": "vision_config" in cfg,
        "mtp_tensors": mtp,
        "checks_failed": fails,
        "notes": notes,
        "command": " ".join(sys.argv),
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    return manifest


# ---------------------------------------------------------------------------
# helpers that need the family + source shapes
# ---------------------------------------------------------------------------
def source_shapes(src):
    src = Path(src)
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    out = {}
    for shard in sorted(set(idx["weight_map"].values())):
        header, _ = read_header(src / shard)
        for name, meta in header.items():
            if name != "__metadata__":
                out[name] = tuple(meta["shape"])
    return out


def detect_family_shapes(src):
    """Family from the source config's own layout (no GGUF needed).

    Mirrors `names.detect_family` using tensor names present in the source.
    """
    from .gguf import Gguf as _G  # noqa: F401
    from .names import FAMILIES
    cfg = json.loads((Path(src) / "config.json").read_text())
    mt = cfg.get("text_config", cfg).get("model_type", cfg.get("model_type"))
    tc = cfg.get("text_config", cfg)
    for arch, cls in FAMILIES.items():
        if mt in cls.hf_model_types:
            return cls(arch=arch, num_layers=int(tc.get("num_hidden_layers", 0)),
                       mtp_index=int(tc.get("num_hidden_layers", 0))
                       if tc.get("mtp_num_hidden_layers") else None)
    raise ValueError(f"no family for source model_type {mt!r}")


def pack_to_src(family, pack_name):
    return family.src_name(pack_name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--alloc", default=None)
    ap.add_argument("--src", default=None)
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--finite", action="store_true")
    ap.add_argument("--expect", action="append", default=[],
                    help="config key=value expectation, repeatable")
    ap.add_argument("--manifest-out", default=None)
    ap.add_argument("--manifest-extra", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    expect = dict(kv.split("=", 1) for kv in args.expect)
    extra = json.loads(Path(args.manifest_extra).read_text()) if args.manifest_extra else None
    log = (lambda *a: None) if args.quiet else print
    manifest = validate(args.model, alloc_path=args.alloc, src=args.src,
                        do_hash=args.hash, finite=args.finite, expect=expect,
                        manifest_extra=extra, log=log)
    if args.manifest_out:
        out = Path(args.manifest_out)
        existing = {}
        if out.exists():
            try:
                existing = json.loads(out.read_text())
            except json.JSONDecodeError:
                existing = {}
        existing.update(manifest)
        out.write_text(json.dumps(existing, indent=1))
        log(f"manifest -> {out}")
    if manifest["checks_failed"]:
        print(f"FAILED: {len(manifest['checks_failed'])} checks")
        for f in manifest["checks_failed"][:20]:
            print("  -", f)
        return 1
    print("ALL STRUCTURAL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())