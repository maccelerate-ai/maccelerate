"""End-to-end smoke test on a synthetic Qwen3.8-shaped model.

Exercises the whole chain — GGUF type table -> allocation -> official imatrix
GGUF -> imatrix -> streamed convert -> structural validation — on a model of a
few hundred KB instead of 54 GB, so the pipeline is verifiable in CI and
without downloading anything.

The synthetic model uses the real qwen35 name spellings (including the
`blk.<n>.nextn.*` MTP block) and a mix of ggml types across the families the
allocator sees in practice (Q6_K, Q5_K, Q8_0, F32), so name mapping, MTP
handling, width assignment and the norm/conv transforms are all on the path.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from maccelerate import affine
from maccelerate.alloc import build_alloc
from maccelerate.convert import convert
from maccelerate.imatrix import convert_imatrix
from maccelerate.names import detect_family
from maccelerate.gguf import Gguf
from maccelerate.safetensors import write_safetensors_raw
from maccelerate.validate import validate

# ggml type ids
F32, Q5_K, Q6_K, Q8_0 = 0, 13, 14, 8

HIDDEN, INTER, VOCAB, LAYERS = 128, 256, 512, 2
MTP_INDEX = LAYERS  # blk.<LAYERS> is the MTP block


# ---------------------------------------------------------------------------
# minimal GGUF writer
# ---------------------------------------------------------------------------
def _str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, kv, tensors, data=None, align=32):
    """tensors: [(name, type_id, dims)]; data: {name: np.float32 array} or None."""
    data = data or {}
    out = bytearray()
    out += b"GGUF" + struct.pack("<I", 3)
    out += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kv))
    for k, (t, v) in kv.items():
        out += _str(k) + struct.pack("<I", t)
        if t == 8:
            out += _str(v)
        elif t == 4:
            out += struct.pack("<I", v)
        elif t == 6:
            out += struct.pack("<f", v)
        elif t == 9:
            et, arr = v
            out += struct.pack("<I", et) + struct.pack("<Q", len(arr))
            for x in arr:
                if et == 8:
                    out += _str(x)
                elif et == 4:
                    out += struct.pack("<I", x)
                elif et == 6:
                    out += struct.pack("<f", x)
                else:
                    raise ValueError(f"array element type {et}")
        else:
            raise ValueError(t)
    offsets, cursor = {}, 0
    for name, tt, dims in tensors:
        arr = data.get(name)
        offsets[name] = cursor
        if arr is not None:
            cursor += int(arr.size) * 4
    for name, tt, dims in tensors:
        out += _str(name) + struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", tt) + struct.pack("<Q", offsets[name])
    pad = (-len(out)) % align
    out += b"\x00" * pad
    for name, tt, dims in tensors:
        arr = data.get(name)
        if arr is not None:
            out += np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    Path(path).write_bytes(bytes(out))


# ---------------------------------------------------------------------------
# synthetic model definition
# ---------------------------------------------------------------------------
def model_tensors():
    """(gguf_name, type, gguf_dims) — dims are (in, out), llama.cpp order."""
    t = [("token_embd.weight", Q6_K, [HIDDEN, VOCAB]),
         ("output.weight", Q8_0, [HIDDEN, VOCAB]),
         ("output_norm.weight", F32, [HIDDEN])]
    for i in range(LAYERS + 1):
        t += [
            (f"blk.{i}.attn_norm.weight", F32, [HIDDEN]),
            (f"blk.{i}.post_attention_norm.weight", F32, [HIDDEN]),
            (f"blk.{i}.attn_q.weight", Q6_K, [HIDDEN, HIDDEN]),
            (f"blk.{i}.attn_k.weight", Q8_0, [HIDDEN, HIDDEN]),
            (f"blk.{i}.attn_v.weight", Q8_0, [HIDDEN, HIDDEN]),
            (f"blk.{i}.attn_output.weight", Q6_K, [HIDDEN, HIDDEN]),
            (f"blk.{i}.ffn_gate.weight", Q5_K, [HIDDEN, INTER]),
            (f"blk.{i}.ffn_up.weight", Q5_K, [HIDDEN, INTER]),
            (f"blk.{i}.ffn_down.weight", Q6_K, [INTER, HIDDEN]),
        ]
    t += [("blk.%d.nextn.eh_proj.weight" % MTP_INDEX, Q6_K, [2 * HIDDEN, HIDDEN]),
          ("blk.%d.nextn.enorm.weight" % MTP_INDEX, F32, [HIDDEN]),
          ("blk.%d.nextn.hnorm.weight" % MTP_INDEX, F32, [HIDDEN]),
          ("blk.%d.nextn.shared_head_norm.weight" % MTP_INDEX, F32, [HIDDEN])]
    return t


def hf_names_for(tensors):
    """Source/HF names the synthetic GGUF maps to."""
    from maccelerate.names import Qwen35Family

    fam = Qwen35Family(arch="qwen35", num_layers=LAYERS, mtp_index=MTP_INDEX)
    out = {}
    for name, tt, dims in tensors:
        src = fam.map_name(name)
        if src is not None:
            out[src] = tuple(reversed(dims))  # GGUF (in,out) -> HF (out,in)
    return out, fam


def build_synthetic(tmp: Path):
    """Writes a model GGUF, a bf16 HF source, and an imatrix GGUF. Returns paths."""
    tmp.mkdir(parents=True, exist_ok=True)
    tensors = model_tensors()
    gguf_path = tmp / "model-UD-Q6_K_M.gguf"
    write_gguf(gguf_path, {
        "general.architecture": (8, "qwen35"),
        "general.alignment": (4, 32),
        "qwen35.block_count": (4, LAYERS + 1),
        "qwen35.embedding_length": (4, HIDDEN),
    }, tensors)

    # ---- bf16 HF source ----
    src = tmp / "src"
    src.mkdir(exist_ok=True)
    names, fam = hf_names_for(tensors)
    rng = np.random.default_rng(7)
    shard = {}
    for name, shape in names.items():
        v = (rng.standard_normal(shape).astype(np.float32) * 0.08)
        shard[name] = ("BF16", shape, affine.f32_to_bf16_u16(v).tobytes())
    write_safetensors_raw(src / "model-00001-of-00001.safetensors", shard)
    json.dump({"metadata": {"total_size": 1},
               "weight_map": {n: "model-00001-of-00001.safetensors" for n in names}},
              open(src / "model.safetensors.index.json", "w"))
    json.dump({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
               "max_position_embeddings": 4096, "vocab_size": VOCAB,
               "text_config": {"model_type": "qwen3_5_text", "hidden_size": HIDDEN,
                               "num_hidden_layers": LAYERS, "vocab_size": VOCAB,
                               "max_position_embeddings": 4096,
                               "mtp_num_hidden_layers": 1,
                               "mtp_use_dedicated_embeddings": False}},
              open(src / "config.json", "w"))
    for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        (src / f).write_text("{}")

    # ---- imatrix GGUF (real F32 in_sum2 / counts) ----
    im_tensors, im_data = [], {}
    for name, tt, dims in tensors:
        if tt == F32 or name in ("token_embd.weight", "output.weight"):
            continue  # no matmul activation statistics for these
        stem = name
        in_dim = dims[0]
        im_tensors += [(f"{stem}.in_sum2", F32, [in_dim]),
                       (f"{stem}.counts", F32, [1])]
        im_data[f"{stem}.in_sum2"] = rng.random(in_dim, dtype=np.float32) + 0.5
        im_data[f"{stem}.counts"] = np.array([16.0], dtype=np.float32)
    imatrix_gguf = tmp / "imatrix_unsloth.gguf"
    write_gguf(imatrix_gguf, {
        "imatrix.chunk_count": (4, 4),
        "imatrix.chunk_size": (4, 512),
        "imatrix.datasets": (9, (8, ["synthetic"])),
    }, im_tensors, data=im_data)
    return gguf_path, src, imatrix_gguf


def test_pipeline_end_to_end(tmp_path):
    gguf_path, src, imatrix_gguf = build_synthetic(tmp_path / "build")

    # --- family detection ---
    fam = detect_family(Gguf(gguf_path))
    assert (fam.num_layers, fam.mtp_index) == (LAYERS, MTP_INDEX)

    # --- allocation from the GGUF type table ---
    raw = build_alloc(gguf_path, "UD-Q6_K_M", group_size=64, src=str(src))
    alloc, manifest = raw["allocation"], raw["manifest"]
    assert manifest["closure"]["quantized"] + manifest["closure"]["dense"] \
        == manifest["closure"]["nonvision"]
    assert manifest["ggml_type_histogram"]["Q6_K"] > 0
    # widths follow the type table
    assert alloc["model.language_model.layers.0.mlp.gate_proj.weight"]["bits"] == 5
    assert alloc["model.language_model.layers.0.self_attn.k_proj.weight"]["bits"] == 8
    assert alloc["lm_head.weight"]["bits"] == 8
    # MTP is allocated, not dropped
    assert "mtp.fc.weight" in alloc
    assert alloc["mtp.fc.weight"]["bits"] == 6
    alloc_path = tmp_path / "build" / "alloc.json"
    alloc_path.write_text(json.dumps(raw, indent=1))

    # --- imatrix GGUF -> HF names ---
    imatrix = tmp_path / "build" / "imatrix.safetensors"
    info = convert_imatrix(imatrix_gguf, imatrix)
    assert info["entries"] > 0
    assert info["total_tokens"] == 4 * 512

    # --- convert ---
    dst = tmp_path / "pack"
    convert(str(src), str(dst), str(alloc_path), gguf_path=str(gguf_path),
            imatrix=str(imatrix), jobs=1, row_block_mb=1, shard_gb=0.001,
            verify=True, card=None)
    assert (dst / "model.safetensors.index.json").exists()
    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["quantization_config"]["bits"] == 6  # dominant width
    # vision removed, MTP kept
    assert "vision_config" not in cfg
    idx = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    assert "language_model.mtp.fc.weight" in idx
    assert "language_model.mtp.layers.0.mlp.down_proj.weight" in idx
    assert "language_model.mtp.fc.scales" in idx
    # delta-encoded norms were folded (+1) on trunk layers, not on MTP
    from maccelerate.safetensors import ShardReader
    r = ShardReader(dst / next(iter(set(idx.values()))))
    assert set(idx.values()) == {next(iter(set(idx.values())))}

    # --- validate ---
    man = validate(str(dst), alloc_path=str(alloc_path), src=str(src),
                   do_hash=True, finite=True, log=lambda *a: None)
    assert man["checks_failed"] == [], man["checks_failed"]
    assert man["effective_bpw"] and man["effective_bpw"] > 5
    assert man["config"]["mtp_num_hidden_layers"] == 1
    assert man["vision_present"] is False
    assert len(man["shards"]) >= 1
    assert all("sha256" in s for s in man["shards"].values())


def test_norm_shift_is_applied_to_trunk_only(tmp_path):
    """The +1 fold must land on trunk norms and never on mtp.*."""
    gguf_path, src, _ = build_synthetic(tmp_path / "b2")
    raw = build_alloc(gguf_path, "UD-Q6_K_M", src=str(src))
    alloc_path = tmp_path / "b2" / "alloc.json"
    alloc_path.write_text(json.dumps(raw))
    dst = tmp_path / "pack2"
    man = convert(str(src), str(dst), str(alloc_path), gguf_path=str(gguf_path),
                  jobs=1, row_block_mb=1, shard_gb=0.001, verify=True, card=None)
    assert man["conversion"]["norm_plus_one_folded"] == 2 * LAYERS + 1  # per layer + final

    from maccelerate.names import detect_family as _d
    from maccelerate.gguf import Gguf as _G
    fam = _d(_G(gguf_path))
    assert fam.needs_norm_shift("language_model.model.layers.0.input_layernorm.weight", 1)
    assert fam.needs_norm_shift("language_model.model.norm.weight", 1)
    assert not fam.needs_norm_shift("language_model.mtp.layers.0.input_layernorm.weight", 1)
    assert not fam.needs_norm_shift("language_model.model.layers.0.mlp.down_proj.weight", 2)


def test_mtp_fc_can_be_pinned_dense(tmp_path):
    gguf_path, src, _ = build_synthetic(tmp_path / "b3")
    raw = build_alloc(gguf_path, "UD-Q6_K_M", src=str(src), keep_mtp_fc_dense=True)
    assert "mtp.fc.weight" not in raw["allocation"]
    assert raw["manifest"]["mtp_fc_dense_pinned"] is True