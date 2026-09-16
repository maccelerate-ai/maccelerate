"""`maccelerate` command line.

Typical use — reproduce an Unsloth Dynamic GGUF's per-tensor allocation as a
native MLX pack:

    maccelerate alloc    --gguf UD-Q6_K_M.gguf --variant UD-Q6_K_M \
                      --src Qwen3.8-27B-bf16 --out alloc.json --hash
    maccelerate imatrix  --imatrix imatrix_unsloth.gguf --out imatrix.safetensors
    maccelerate convert  --src Qwen3.8-27B-bf16 --dst out/ --alloc alloc.json \
                      --imatrix imatrix.safetensors --row-block-mb 64 \
                      --shard-gb 2 --jobs 3 --verify
    maccelerate validate --model out/ --alloc alloc.json --src Qwen3.8-27B-bf16 \
                      --hash --finite

or all of it at once with `maccelerate ud`, which also writes the model card.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .alloc import build_alloc
from .convert import convert
from .gguf import Gguf
from .imatrix import convert_imatrix
from .names import detect_family
from .validate import main as validate_main


def _alloc_cmd(a):
    raw = build_alloc(
        a.gguf, a.variant, group_size=a.group_size,
        keep_mtp_fc_dense=a.keep_mtp_fc_dense, src=a.src,
        gguf_url=a.gguf_url, gguf_revision=a.gguf_revision,
        gguf_sha256=a.gguf_sha256, imatrix=a.imatrix,
        imatrix_revision=a.imatrix_revision, hash_gguf=a.hash,
        extra_manifest=json.loads(Path(a.manifest_extra).read_text())
        if a.manifest_extra else None)
    Path(a.out).write_text(json.dumps(raw, indent=1))
    m = raw["manifest"]
    print(f"{a.variant}: {m['quantized_tensors']} quantized, "
          f"{m['dense_tensors']} dense")
    print("  widths: " + " ".join(f"{k}:{v}"
                                  for k, v in sorted(m["allocated_width_histogram"].items())))
    print("  ggml:   " + " ".join(f"{k}:{v}"
                                  for k, v in sorted(m["ggml_type_histogram"].items())))
    if m["substituted_types"]:
        print(f"  substituted widths for {len(m['substituted_types'])} tensors")
    if m["closure"]:
        c = m["closure"]
        print(f"  closure OK: {c['quantized']}+{c['dense']} == {c['nonvision']} "
              f"non-vision source tensors")
    print(f"wrote {a.out}")
    return 0


def _imatrix_cmd(a):
    info = convert_imatrix(a.imatrix, a.out, revision=a.revision, url=a.url)
    print(f"{Path(a.imatrix).name}: {info['entries']} entries -> {a.out}")
    print(f"  chunk_count={info['chunk_count']} chunk_size={info['chunk_size']} "
          f"total_tokens={info['total_tokens']} datasets={info['datasets']}")
    if info["skipped"]:
        print(f"  skipped {len(info['skipped'])} (no counts / unmapped)")
    return 0


def _default_card(variant, gguf, bf16_repo):
    family = detect_family(Gguf(gguf))
    return {
        "base_model": bf16_repo,
        "arch_tag": family.arch,
        "headline": f"Native MLX affine re-encoding of the official {variant} "
                    f"per-tensor allocation.",
        "method": (
            f"Widths come from the official `{Path(gguf).name}` checkpoint's own "
            "per-tensor ggml type table, not from a re-measured allocation. Each "
            "tensor's k-quant/IQ family is mapped to the MLX affine width with the "
            "same bit count; MLX affine cannot reproduce llama.cpp's codebooks, so "
            "this is the same allocation re-encoded, not numerical parity."),
        "pinning": "Widths are not a house pin: every tensor keeps the width the "
                   "upstream allocator gave it, so the pack is mixed-width.",
        "serving_notes": "- Serve with `mlx-serve --model <dir> --kv-quant 8`.",
        "conversion_notes": "- Quantized from the clean bf16 source, never from "
                            "dequantized GGUF weights.\n"
                            "- 4/8-bit tensors take the imatrix-weighted search; "
                            "5/6-bit tensors use `mx.quantize` because the weighted "
                            "packing path implements 2/3/4/8-bit only.\n"
                            "- Kept bf16: every norm, bias, conv and SSM state. "
                            "The vision tower is dropped. MTP is kept inline.",
    }


def _convert_cmd(a):
    card = None
    if not a.no_card:
        card = _default_card(a.variant or Path(a.alloc).stem, a.gguf, a.base_model or "")
        card["repo_name"] = a.repo_name or Path(a.dst).name
        card["variant"] = a.variant or Path(a.alloc).stem
        for k in ("headline", "method", "pinning", "serving_notes", "conversion_notes"):
            if getattr(a, k, None):
                card[k] = getattr(a, k)
    convert(a.src, a.dst, a.alloc, gguf_path=a.gguf, imatrix=a.imatrix,
            jobs=a.jobs, row_block_mb=a.row_block_mb, shard_gb=a.shard_gb,
            verify=a.verify, card=card)
    return 0


def _ud_cmd(a):
    """alloc -> imatrix -> convert -> validate, the reproducible whole chain."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(a.work) if a.work else out.parent / (out.name + "-build")
    work.mkdir(parents=True, exist_ok=True)
    alloc_path = work / "alloc.json"
    imatrix_path = work / "imatrix.safetensors"

    print(f"== alloc ({a.variant}) ==")
    _alloc_cmd(argparse.Namespace(
        gguf=a.gguf, variant=a.variant, group_size=a.group_size,
        keep_mtp_fc_dense=a.keep_mtp_fc_dense, src=a.src, gguf_url=a.gguf_url,
        gguf_revision=a.gguf_revision, gguf_sha256=a.gguf_sha256,
        imatrix=a.imatrix, imatrix_revision=a.imatrix_revision, hash=a.hash,
        manifest_extra=None, out=str(alloc_path)))
    if a.imatrix:
        print("== imatrix ==")
        _imatrix_cmd(argparse.Namespace(
            imatrix=a.imatrix, out=str(imatrix_path), revision=a.gguf_revision,
            url="https://huggingface.co/" + (a.gguf_url or "")))
    print("== convert ==")
    _convert_cmd(argparse.Namespace(
        src=a.src, dst=str(out), alloc=str(alloc_path),
        imatrix=str(imatrix_path) if a.imatrix else None,
        gguf=a.gguf, variant=a.variant, repo_name=a.repo_name,
        base_model=a.base_model, jobs=a.jobs, row_block_mb=a.row_block_mb,
        shard_gb=a.shard_gb, verify=True, no_card=False,
        headline=a.headline, method=a.method, pinning=a.pinning,
        serving_notes=a.serving_notes, conversion_notes=a.conversion_notes))
    print("== validate ==")
    rc = validate_main(["--model", str(out), "--alloc", str(alloc_path),
                        "--src", a.src] + (["--hash", "--finite"] if a.finite else []) +
                       ["--manifest-out", str(out / "manifest.json")])
    return rc


def _inspect_cmd(a):
    g = Gguf(a.gguf)
    fam = detect_family(g)
    print(f"arch={g.arch} file_type={g.file_type()} tensors={len(g.tensors)} "
          f"gguf_version={g.version}")
    print(f"family={fam.arch} num_layers={fam.num_layers} mtp_index={fam.mtp_index}")
    print("types: " + " ".join(f"{k}:{v}" for k, v in sorted(g.type_histogram().items())))
    unmapped = [n for n in g.tensors if fam.map_name(n) is None]
    if unmapped:
        print(f"unmapped tensors: {len(unmapped)} e.g. {unmapped[:5]}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="maccelerate", description=__doc__.split("\n\n")[0])
    p.add_argument("--version", action="version", version=f"maccelerate {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("alloc", help="UD GGUF type table -> allocation JSON")
    s.add_argument("--gguf", required=True)
    s.add_argument("--variant", required=True, help="exact variant, e.g. UD-Q6_K_M")
    s.add_argument("--out", required=True)
    s.add_argument("--group-size", type=int, default=64)
    s.add_argument("--src", default=None, help="bf16 source dir (proves closure)")
    s.add_argument("--gguf-url", default=None)
    s.add_argument("--gguf-revision", default=None)
    s.add_argument("--gguf-sha256", default=None)
    s.add_argument("--imatrix", default=None)
    s.add_argument("--imatrix-revision", default=None)
    s.add_argument("--hash", action="store_true")
    s.add_argument("--keep-mtp-fc-dense", action="store_true")
    s.add_argument("--manifest-extra", default=None)
    s.set_defaults(fn=_alloc_cmd)

    s = sub.add_parser("imatrix", help="imatrix GGUF -> HF-named SafeTensors")
    s.add_argument("--imatrix", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--revision", default=None)
    s.add_argument("--url", default=None)
    s.set_defaults(fn=_imatrix_cmd)

    s = sub.add_parser("convert", help="bf16 source + allocation -> MLX pack")
    s.add_argument("--src", required=True)
    s.add_argument("--dst", required=True)
    s.add_argument("--alloc", required=True)
    s.add_argument("--gguf", required=True, help="the GGUF that produced --alloc")
    s.add_argument("--imatrix", default=None)
    s.add_argument("--jobs", type=int, default=3)
    s.add_argument("--row-block-mb", type=int, default=64)
    s.add_argument("--shard-gb", type=float, default=2.0)
    s.add_argument("--verify", action="store_true")
    s.add_argument("--no-card", action="store_true")
    s.add_argument("--repo-name", default=None)
    s.add_argument("--base-model", default=None)
    s.add_argument("--variant", default=None)
    s.add_argument("--headline", default=None)
    s.add_argument("--method", default=None)
    s.add_argument("--pinning", default=None)
    s.add_argument("--serving-notes", default=None)
    s.add_argument("--conversion-notes", default=None)
    s.set_defaults(fn=_convert_cmd)

    s = sub.add_parser("validate", help="structural validation + manifest")
    s.add_argument("--model", required=True)
    s.add_argument("--alloc", default=None)
    s.add_argument("--src", default=None)
    s.add_argument("--hash", action="store_true")
    s.add_argument("--finite", action="store_true")
    s.add_argument("--expect", action="append", default=[])
    s.add_argument("--manifest-out", default=None)
    s.add_argument("--manifest-extra", default=None)
    s.set_defaults(fn=lambda a: validate_main(
        ["--model", a.model] + (["--alloc", a.alloc] if a.alloc else [])
        + (["--src", a.src] if a.src else []) + (["--hash"] if a.hash else [])
        + (["--finite"] if a.finite else [])
        + sum([["--expect", e] for e in a.expect], [])
        + (["--manifest-out", a.manifest_out] if a.manifest_out else [])
        + (["--manifest-extra", a.manifest_extra] if a.manifest_extra else [])))

    s = sub.add_parser("compare", help="prove pack-vs-GGUF inventory/alloc/dense identity")
    s.add_argument("--gguf", required=True)
    s.add_argument("--pack", required=True)
    s.add_argument("--src", default=None)
    s.add_argument("--json-out", default=None)
    s.set_defaults(fn=lambda a: __import__("maccelerate.compare", fromlist=["main"]).main(
        ["--gguf", a.gguf, "--pack", a.pack]
        + (["--src", a.src] if a.src else [])
        + (["--json-out", a.json_out] if a.json_out else [])))

    s = sub.add_parser("inspect", help="print a GGUF's type table and family")
    s.add_argument("--gguf", required=True)
    s.set_defaults(fn=_inspect_cmd)

    s = sub.add_parser("ud", help="alloc + imatrix + convert + validate, one shot")
    s.add_argument("--gguf", required=True)
    s.add_argument("--variant", required=True)
    s.add_argument("--src", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--imatrix", default=None)
    s.add_argument("--work", default=None)
    s.add_argument("--group-size", type=int, default=64)
    s.add_argument("--keep-mtp-fc-dense", action="store_true")
    s.add_argument("--gguf-url", default=None)
    s.add_argument("--gguf-revision", default=None)
    s.add_argument("--gguf-sha256", default=None)
    s.add_argument("--imatrix-revision", default=None)
    s.add_argument("--hash", action="store_true")
    s.add_argument("--jobs", type=int, default=3)
    s.add_argument("--row-block-mb", type=int, default=64)
    s.add_argument("--shard-gb", type=float, default=2.0)
    s.add_argument("--finite", action="store_true")
    s.add_argument("--repo-name", default=None)
    s.add_argument("--base-model", default=None)
    s.add_argument("--headline", default=None)
    s.add_argument("--method", default=None)
    s.add_argument("--pinning", default=None)
    s.add_argument("--serving-notes", default=None)
    s.add_argument("--conversion-notes", default=None)
    s.set_defaults(fn=_ud_cmd)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.fn(args)

if __name__ == "__main__":
    sys.exit(main())
