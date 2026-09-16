"""Tensor-name mapping + weight classification, per model family.

The pipeline is family-pluggable. `qwen35` (Qwen3.5/3.6/3.8-27B dense+MTP) is
implemented; adding a family means subclassing `Family` with its own name maps
and registering it — nothing in the converter is architecture-specific beyond
this module.

Two name spaces are in play and keeping them straight matters:

  GGUF         `blk.0.attn_qkv.weight`                      (llama.cpp)
  source/HF    `model.language_model.layers.0.linear_attn.in_proj_qkv.weight`
  pack         `language_model.model.layers.0.linear_attn.in_proj_qkv.weight`

Allocations are keyed by **source/HF** names (that is what the bf16 checkpoint
index uses); the converter renames to pack keys, which is what mlx-serve's
`resolveWeightPrefix` looks up.

The qwen35 maps mirror `jclyons52/ud2mlx`'s `dequant_gguf_to_hf.map_name`, plus
the MTP block (`blk.<n>.nextn.*`), which ud2mlx drops.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# qwen35 (Qwen3.5 / 3.6 / 3.8 dense family)
# ---------------------------------------------------------------------------
QWEN35_PREFIX = {
    "token_embd.weight": "model.language_model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.language_model.norm.weight",
}
QWEN35_FFN = {
    "ffn_gate.weight": "gate_proj",
    "ffn_up.weight": "up_proj",
    "ffn_down.weight": "down_proj",
}
QWEN35_LEAF = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
}
# the MTP block's concat projection + its norms
QWEN35_NEXTN = {
    "nextn.eh_proj.weight": "mtp.fc.weight",
    "nextn.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
    "nextn.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
    "nextn.shared_head_norm.weight": "mtp.norm.weight",
}
# 1-D norms Qwen ships zero-centered (delta) but mlx-serve reads as `rmsnorm * w`
QWEN35_NORM_SHIFT = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
)


@dataclass
class Family:
    """Bidirectional name mapping + weight classification for one architecture."""

    arch: str
    num_layers: int
    mtp_index: int | None = None
    vision_prefix: str | None = None
    #: HF `model_type` spellings that belong to this family (the GGUF
    #: architecture string and the HF model_type differ for qwen3.5-family).
    hf_model_types: tuple = ()

    # -- GGUF -> source/HF ------------------------------------------------
    def map_name(self, gguf_name):
        raise NotImplementedError

    # -- source/HF -> pack (mlx-serve layout) -----------------------------
    def pack_name(self, src_name):
        raise NotImplementedError

    # -- pack -> source/HF ------------------------------------------------
    def src_name(self, pack_name):
        raise NotImplementedError

    # -- role for allocation/README ---------------------------------------
    def classify(self, src_name):
        raise NotImplementedError

    def needs_norm_shift(self, pack_name, ndim):
        return False

    def conv1d_transpose(self, pack_name, ndim):
        return pack_name.endswith("conv1d.weight") and ndim == 3


class Qwen35Family(Family):
    hf_model_types = ("qwen3_5", "qwen3_5_text", "qwen3_5_moe")

    def map_name(self, gguf_name):
        if gguf_name in QWEN35_PREFIX:
            return QWEN35_PREFIX[gguf_name]
        m = re.match(r"blk\.(\d+)\.(.+)", gguf_name)
        if not m:
            return None
        i, rest = int(m.group(1)), m.group(2)
        if rest in QWEN35_NEXTN:
            return QWEN35_NEXTN[rest] if i == self.mtp_index else None
        if i < self.num_layers:
            stem = f"model.language_model.layers.{i}."
        elif i == self.mtp_index:
            stem = "mtp.layers.0."
        else:
            return None
        if rest in QWEN35_FFN:
            return f"{stem}mlp.{QWEN35_FFN[rest]}.weight"
        if rest in QWEN35_LEAF:
            return stem + QWEN35_LEAF[rest]
        return None

    def pack_name(self, src_name):
        if src_name.startswith("model.language_model."):
            return "language_model.model." + src_name[len("model.language_model."):]
        if src_name.startswith("mtp."):
            return "language_model.mtp." + src_name[len("mtp."):]
        if src_name == "lm_head.weight":
            return "language_model.lm_head.weight"
        return src_name

    def src_name(self, pack_name):
        if pack_name.startswith("language_model.model."):
            return "model.language_model." + pack_name[len("language_model.model."):]
        if pack_name.startswith("language_model.mtp."):
            return "mtp." + pack_name[len("language_model.mtp."):]
        if pack_name == "language_model.lm_head.weight":
            return "lm_head.weight"
        return pack_name

    def classify(self, name):
        """(class, quantizable, pinned) for a SOURCE weight name.

        `quantizable` is advisory: the authoritative set is the GGUF type table.
        """
        if name.startswith("model.visual."):
            return "vision", False, False
        if not name.endswith(".weight"):
            return "dense", False, False
        if name.startswith("mtp."):
            if name == "mtp.fc.weight":
                return "mtp_fc", True, False
            return "mtp", True, True
        if "embed_tokens" in name:
            return "embed", True, False
        if name == "lm_head.weight":
            return "lm_head", True, False
        if ".self_attn." in name:
            if name.endswith(("q_proj.weight", "k_proj.weight",
                              "v_proj.weight", "o_proj.weight")):
                return "attn", True, True
            return "dense", False, False
        if ".linear_attn." in name:
            if name.endswith(("in_proj_a.weight", "in_proj_b.weight")):
                return "gdn_ab", True, True
            if name.endswith("in_proj_qkv.weight"):
                return "gdn_qkv", True, False
            if name.endswith("in_proj_z.weight"):
                return "gdn_z", True, False
            if name.endswith("out_proj.weight"):
                return "gdn_out", True, False
            return "dense", False, False
        if name.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")):
            return "mlp_gate_up", True, False
        if name.endswith("mlp.down_proj.weight"):
            return "mlp_down", True, False
        return "dense", False, False

    def needs_norm_shift(self, pack_name, ndim):
        if ndim != 1 or ".mtp." in pack_name:
            return False
        if pack_name == "language_model.model.norm.weight":
            return True
        return (pack_name.startswith("language_model.model.layers.")
                and pack_name.endswith(QWEN35_NORM_SHIFT))


def family_from_names(names, arch="qwen35"):
    """Infer a Family from tensor names alone.

    imatrix GGUFs carry no `general.architecture`, so the family must be
    recovered from the names they do have. `num_layers` is the highest block
    index plus one, or the MTP block index when `blk.<n>.nextn.*` is present.
    """
    if arch not in FAMILIES:
        raise ValueError(f"unknown family {arch!r} (known: {sorted(FAMILIES)})")
    blocks = [int(m.group(1)) for n in names
              if (m := re.match(r"blk\.(\d+)\.", n))]
    mtp_index = None
    for n in names:
        m = re.match(r"blk\.(\d+)\.nextn\.", n)
        if m:
            mtp_index = int(m.group(1))
            break
    if mtp_index is not None:
        num_layers = mtp_index
    elif blocks:
        num_layers = max(blocks) + 1
    else:
        num_layers = 0
    return FAMILIES[arch](arch=arch, num_layers=num_layers, mtp_index=mtp_index)


FAMILIES = {"qwen35": Qwen35Family}


def detect_family(gguf):
    """Pick a Family for a loaded `Gguf`, deriving layer/MTP counts from metadata.

    The MTP block is detected by the presence of `blk.<n>.nextn.*` rather than
    assumed: models without an MTP head get `mtp_index = None` and `num_layers`
    equal to the full block count.
    """
    arch = gguf.arch
    if arch not in FAMILIES:
        raise ValueError(
            f"unsupported GGUF architecture {arch!r}; add a Family subclass in "
            f"maccelerate/names.py and register it in FAMILIES (known: "
            f"{sorted(FAMILIES)})")
    block_count = gguf.meta("block_count")
    if not block_count:
        raise ValueError(f"{arch}: missing {arch}.block_count")
    block_count = int(block_count)

    mtp_index = None
    for name in gguf.tensors:
        m = re.match(r"blk\.(\d+)\.nextn\.", name)
        if m:
            mtp_index = int(m.group(1))
            break
    num_layers = mtp_index if mtp_index is not None else block_count

    vision_prefix = None
    for name in gguf.tensors:
        if name.startswith("blk.") or name in ("token_embd.weight",):
            vision_prefix = None
    # GGUF carries the vision tower in a separate mmproj file; nothing to strip.
    return FAMILIES[arch](
        arch=arch,
        num_layers=num_layers,
        mtp_index=mtp_index,
        vision_prefix=vision_prefix,
    )


def bytes_for(params, bits, group_size):
    """Exact bytes the converter writes: packed weights + bf16 scales + biases."""
    return params * (bits * 8 + 256 // group_size) // 64