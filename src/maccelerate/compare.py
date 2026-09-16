"""Compare a converted MLX pack against the official GGUF it was derived from.

Answers "is our conversion identical to the upstream GGUF?" with numbers, and
the answer differs per layer of the stack:

  1. INVENTORY — which tensors exist. Provably identical: every GGUF tensor maps
     to exactly one pack weight, quantized ones gaining `.scales`/`.biases`, and
     nothing extra.
  2. DENSE (bf16) — the norms, biases, conv and SSM state. Identical up to
     *representational* conventions the two runtimes legitimately disagree on,
     and the tool distinguishes them empirically rather than hardcoding:
       * `reordered`  — llama.cpp permutes the GDN/SSM head axis (qkv, a, b,
         dt, A, conv1d) to its internal layout; the pack keeps HF/mlx-serve
         order. The two are equal as **multisets**, which is what is checked.
         `ssm_a` also carries llama.cpp's `-exp(A_log)` pre-transform.
       * `folded`     — the MTP head's norms are stored `+1`-folded by
         llama.cpp and raw by the pack (mlx-serve folds at load).
  3. QUANTIZED — the affine tensors. **Same allocation, different numbers**:
     MLX affine cannot reproduce llama.cpp k-quant/IQ codebooks. Reported as
     relative RMSE against the bf16 source and pack-vs-GGUF.

Requires the `gguf` package (`pip install 'maccelerate[compare]'`).

  maccelerate compare --gguf UD-Q6_K_M.gguf --pack out/ --src Qwen3.8-27B-bf16
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from . import affine
from .gguf import Gguf
from .names import detect_family
from .safetensors import ShardReader

# llama.cpp stores `ssm_a` as -exp(A_log); everything else SSM-side is a pure
# permutation of the HF values.
GGUF_PREDEFORM = {"ssm_a": "neg_exp"}


def _leaf(gguf_name):
    parts = gguf_name.split(".")
    return ".".join(parts[2:]) if parts[0] == "blk" and len(parts) > 2 else gguf_name


def _rms_rel(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.sqrt(((a - b) ** 2).mean()) / np.sqrt((b ** 2).mean()))


def classify_alignment(gguf_vals, pack_vals, multiset_tol=0.01, min_elem=0.20):
    """('exact'|'reordered'|'different', elementwise_rel, multiset_rel).

    Pure numpy so it is unit-testable without a GGUF.

    A permutation is claimed only when BOTH absolute guards hold:

      * `multiset <= multiset_tol` — the values are the same multiset, to well
        inside any plausible quantization error, and
      * `elementwise >= min_elem` — the two are grossly misaligned (a real head
        permutation gives ~100 %; Q8_0 quantization gives <1 %).

    Both guards matter. Sorting is a *smoothing* operation, so `sorted(x+e)`
    tracks `sorted(x)` far better than `x+e` tracks `x` for any noise `e`; a
    ratio-only rule therefore calls a 5 %-noise tensor "permuted". This is also
    why a quantized tensor must never be judged by a multiset comparison
    against an exact reference — only exact-vs-exact comparisons (bf16 vs f32)
    are meaningful here.
    """
    g = np.asarray(gguf_vals).reshape(-1)
    p = np.asarray(pack_vals).reshape(-1)
    if g.shape != p.shape:
        return "different", float("inf"), float("inf")
    if np.array_equal(g, p):
        return "exact", 0.0, 0.0
    if not np.any(g):
        return "different", float("inf"), float("inf")
    elem = _rms_rel(p, g)
    # compare as MULTISETS: sort each side independently, otherwise the
    # permutation we are trying to detect cancels out and multi == elem.
    multi = _rms_rel(np.sort(p), np.sort(g))
    if multi <= multiset_tol and elem >= min_elem:
        return "reordered", elem, multi
    return "different", elem, multi


class _Reader:
    def __init__(self, root, index_map):
        self.root, self.map, self._cache = Path(root), index_map, {}

    def read(self, name):
        shard = self.map[name]
        if shard not in self._cache:
            self._cache[shard] = ShardReader(self.root / shard)
        return self._cache[shard].read(name)


def _dequant_pack(reader, base, spec):
    sh = reader.read(base + ".weight")[0]
    sc = reader.read(base + ".scales")[0]
    bi = reader.read(base + ".biases")[0]
    mx = affine.mx()
    dq = mx.dequantize(mx.array(sh), mx.array(sc).view(mx.bfloat16),
                       mx.array(bi).view(mx.bfloat16),
                       group_size=spec["group_size"], bits=spec["bits"]).astype(mx.float32)
    mx.eval(dq)
    return np.asarray(dq)


def compare(gguf_path, pack_dir, src=None, max_params=90_000_000, log=print):
    from gguf import GGUFReader
    from gguf.quants import dequantize

    gmeta = Gguf(gguf_path)
    family = detect_family(gmeta)
    pack_dir = Path(os.path.expanduser(pack_dir))
    idx = json.loads((pack_dir / "model.safetensors.index.json").read_text())["weight_map"]
    cfg = json.loads((pack_dir / "config.json").read_text())
    qcfg = cfg.get("quantization_config") or cfg.get("quantization") or {}
    per_tensor = {k: v for k, v in qcfg.items() if isinstance(v, dict)}

    pack_reader = _Reader(pack_dir, idx)
    src_index, src_reader = {}, None
    if src:
        src_index = json.loads((Path(src) / "model.safetensors.index.json").read_text())["weight_map"]
        src_reader = _Reader(src, src_index)

    gnames = {t.name: t for t in GGUFReader(str(gguf_path), "r").tensors}

    def gguf_f32(name):
        t = gnames[name]
        v = dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32)
        pre = GGUF_PREDEFORM.get(_leaf(name))
        if pre == "neg_exp":                      # invert to the pack's convention
            v = np.log(np.abs(v)).astype(np.float32)
        return v

    report = {"variant": gmeta.kv.get("general.name"), "sections": {}}
    log("=" * 74)
    log("[1] INVENTORY")
    log("=" * 74)

    expected = {}
    for gname in gmeta.tensors:
        pn = family.pack_name(family.map_name(gname))
        base = pn[:-len(".weight")] if pn.endswith(".weight") else pn
        if pn.endswith(".weight") and base in per_tensor:
            for suf in (".weight", ".scales", ".biases"):
                expected[base + suf] = gname
        else:
            expected[pn] = gname
    have = set(idx)
    missing, extra = sorted(set(expected) - have), sorted(have - set(expected))
    inv_ok = not missing and not extra
    log(f"  GGUF tensors        : {len(gmeta.tensors)}")
    log(f"  expected pack files : {len(expected)}")
    log(f"  actual pack files   : {len(have)}")
    log(f"  MISSING {len(missing)}   EXTRA {len(extra)}")
    log(f"  MTP: GGUF {len([n for n in gmeta.tensors if '.nextn.' in n or n.startswith('blk.64.')])}"
        f" -> pack {len([k for k in have if '.mtp.' in k and k.endswith('.weight')])} weights")
    log(f"  VISION: in GGUF {len([n for n in gmeta.tensors if 'visual' in n])}"
        f", in HF source {len([k for k in src_index if k.startswith('model.visual.')]) if src else '?'}"
        f", in pack 0")
    report["sections"]["inventory"] = {"gguf_tensors": len(gmeta.tensors),
                                       "expected_files": len(expected),
                                       "actual_files": len(have),
                                       "missing": missing[:10], "extra": extra[:10],
                                       "ok": inv_ok}

    # ---------------- dense ----------------
    log("")
    log("=" * 74)
    log("[2] DENSE (bf16) TENSORS")
    log("=" * 74)
    counts, problems = Counter(), []
    for gname in gmeta.tensors:
        pn = family.pack_name(family.map_name(gname))
        base = pn[:-len(".weight")] if pn.endswith(".weight") else pn
        if (pn.endswith(".weight") and base in per_tensor) or pn not in have:
            continue
        g = gguf_f32(gname)
        got = pack_reader.read(pn)[0]
        exp = affine.f32_to_bf16_u16(g)
        if exp.shape != got.shape and exp.shape == got.shape[:2]:
            exp = exp[:, :, None]                  # conv1d [C,K] -> [C,K,1]
        if exp.shape == got.shape and np.array_equal(got.reshape(-1), exp.reshape(-1)):
            counts["bit_exact"] += 1
            continue
        pv = affine.bf16_to_f32(got.reshape(-1))
        gv = g.reshape(-1)
        if ".mtp." in pn and np.array_equal(affine.f32_to_bf16_u16(g - 1).reshape(-1),
                                           got.reshape(-1).astype(np.uint16)):
            counts["mtp_folded_in_gguf"] += 1
            continue
        kind, elem, multi = classify_alignment(gv, pv)
        if kind == "reordered":
            counts["runtime_reordered"] += 1
        elif multi <= 0.01:
            counts["multiset_equal"] += 1
        else:
            counts["unexplained"] += 1
            problems.append((gname, elem, multi))
    log(f"  bit-exact to the GGUF             : {counts['bit_exact']}")
    log(f"  GGUF-permuted GDN/SSM (multiset)  : {counts['runtime_reordered']}")
    log(f"  MTP norms GGUF-folded, pack raw   : {counts['mtp_folded_in_gguf']}")
    log(f"  unexplained                       : {counts['unexplained']}")
    for n, e, m in problems[:10]:
        log(f"     {n}  elem {e:.3e}  multiset {m:.3e}")
    report["sections"]["dense"] = dict(counts)
    report["sections"]["dense"]["problems"] = problems[:10]

    # ---------------- quantized ----------------
    log("")
    log("=" * 74)
    log("[3] QUANTIZED TENSORS  (same allocation; numbers need not match)")
    log("=" * 74)
    alloc = Counter(s["bits"] for s in per_tensor.values())
    log("  allocation: " + " ".join(f"{b}-bit x{c}" for b, c in sorted(alloc.items())))
    log("")
    log(f"  {'tensor':40s} {'b':>2} {'pack vs bf16':>13} {'gguf vs bf16':>13} {'pack vs gguf':>13} {'order':>10}")
    rows = []
    # one representative per width, preferring non-permuted tensors
    cand = {}
    for base, spec in sorted(per_tensor.items(), key=lambda kv: (kv[1]["bits"], kv[0])):
        w = base + ".weight"
        if spec["bits"] in cand or w not in idx:
            continue
        if int(np.prod(pack_reader.read(w)[0].shape)) * 4 <= max_params:
            cand.setdefault(spec["bits"], []).append(base)
    for bits in sorted(cand):
        for base in cand[bits]:
            gname = expected[base + ".weight"]
            g = gguf_f32(gname)
            m = _dequant_pack(pack_reader, base, per_tensor[base])
            ref = None
            sname = family.map_name(gname)
            if src_reader is not None and sname in src_index:
                ref = affine.bf16_to_f32(src_reader.read(sname)[0])
            # Is the GGUF permuted relative to HF order? A permutation makes the
            # elementwise error ~100%+, which no sane quantizer does (Q8_0 is
            # well under 1%). Do NOT decide this from a multiset comparison:
            # sorting a quantized vector against an exact one shrinks the error
            # regardless of any permutation.
            permuted = False
            if ref is not None and ref.shape == g.shape:
                elem_g = _rms_rel(g, ref)
                permuted = elem_g > 0.25
                if not permuted and len(cand[bits]) > 1:
                    continue                      # prefer a comparable tensor
            em = 100 * _rms_rel(m, ref) if ref is not None and ref.shape == m.shape else float("nan")
            eg = 100 * _rms_rel(g, ref) if ref is not None and ref.shape == g.shape else float("nan")
            dv = 100 * (_rms_rel(np.sort(np.asarray(m).reshape(-1)),
                                  np.sort(np.asarray(g).reshape(-1)))
                       if permuted else _rms_rel(m, g))
            log(f"  {base.split('language_model.')[-1][:40]:40s} {bits:>2} "
                f"{em:>12.4f}% {eg:>12.4f}% {dv:>12.4f}% "
                f"{'gguf-perm' if permuted else 'same':>10}")
            rows.append({"tensor": base, "bits": bits,
                         "pack_vs_bf16_pct": em, "gguf_vs_bf16_pct": eg,
                         "pack_vs_gguf_pct": dv,
                         "gguf_order": "permuted" if permuted else "same"})
            break
    report["sections"]["quantized"] = {"allocation": dict(alloc), "rows": rows}
    report["sections"]["inventory"]["ok"] = inv_ok
    return report


def main(argv=None):
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--src", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    try:
        rep = compare(args.gguf, args.pack, src=args.src,
                      log=(lambda *a: None) if args.quiet else print)
    except ImportError as e:
        print(f"compare needs the `gguf` package: pip install 'maccelerate[compare]' ({e})",
              file=sys.stderr)
        return 2
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rep, indent=1))
        print(f"report -> {args.json_out}")
    if not args.quiet:
        d = rep["sections"]["dense"]
        q = rep["sections"]["quantized"]
        print()
        print(f"VERDICT: inventory {'identical' if rep['sections']['inventory']['ok'] else 'MISMATCH'}"
              f" | allocation {sum(q['allocation'].values())} tensors reproduced"
              f" | dense: {d.get('bit_exact',0)} bit-exact,"
              f" {d.get('runtime_reordered',0)} GGUF-permuted (multiset-equal),"
              f" {d.get('mtp_folded_in_gguf',0)} MTP-norm convention,"
              f" {d.get('unexplained',0)} unexplained"
              f" | quantized: same allocation, DIFFERENT numbers (see table)")
    bad = (not rep["sections"]["inventory"]["ok"]
           or rep["sections"]["dense"].get("unexplained", 0) > 0)
    return 1 if bad else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())