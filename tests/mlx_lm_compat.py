#!/usr/bin/env python3
"""Opt-in mlx-lm compatibility probe for the Qwen3.8 mixed-quant MLX pack.

This is deliberately not a pytest test: it needs a local 23 GB checkpoint and
Metal.  It never writes to the checkpoint.  Set MACCELERATE_MLX_LM_MODEL (or
pass --model) and run it in a dedicated mlx-lm environment.

The default run is the canonical unmodified ``mlx_lm.load()`` path.  The
``--sanitize-workaround`` comparison exists because mlx-lm 0.31.3 treats the
presence of Qwen3.8 MTP weights as proof that every trunk norm still needs its
raw-checkpoint +1 transform.  maccelerate packs have already transformed the
trunk norms, while preserving the raw MTP norms for mlx-serve.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any


PROMPT = "Write the numbers 1 to 40 separated by single spaces. Output nothing else."
EXPECTED_TEXT = " ".join(str(n) for n in range(1, 41))


def mac_peak_rss_gib() -> float:
    """macOS reports ru_maxrss in bytes (Linux reports KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def installed_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "mlx_lm": importlib.metadata.version("mlx-lm"),
        "mlx": importlib.metadata.version("mlx"),
    }


def pack_inventory(model_dir: Path) -> dict[str, Any]:
    config = json.loads((model_dir / "config.json").read_text())
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    mtp = sorted(key for key in weight_map if key.startswith("language_model.mtp."))
    qmap = config.get("quantization", {})
    overrides = {key: value for key, value in qmap.items() if isinstance(value, dict)}
    return {
        "model_type": config.get("model_type"),
        "architecture": config.get("architectures"),
        "mtp_keys_in_index": len(mtp),
        "mtp_key_sample": mtp[:3],
        "quantization_default": {key: value for key, value in qmap.items() if not isinstance(value, dict)},
        "per_module_quantization_entries": len(overrides),
        "per_module_widths": sorted({value.get("bits") for value in overrides.values()}),
    }


def install_sanitize_observer(workaround: bool) -> dict[str, Any]:
    """Observe mlx-lm's MTP handling and optionally apply the minimal shim."""
    from mlx_lm.models import qwen3_5

    original = qwen3_5.TextModel.sanitize
    observation: dict[str, Any] = {}

    def sanitize(self, weights):
        observation["mtp_keys_before_sanitize"] = sum("mtp." in key for key in weights)
        observation["converted_conv1d"] = not any(
            key.endswith("conv1d.weight") and value.shape[-1] != 1
            for key, value in weights.items()
        )
        if workaround and observation["converted_conv1d"]:
            # Converted packs need neither raw-checkpoint transform.  Remove the
            # unsupported head *before* mlx-lm decides whether to add +1.
            weights = {key: value for key, value in weights.items() if "mtp." not in key}
        sanitized = original(self, weights)
        observation["mtp_keys_after_sanitize"] = sum("mtp." in key for key in sanitized)
        return sanitized

    qwen3_5.TextModel.sanitize = sanitize
    return observation


def observed_quantization(model, config: dict[str, Any]) -> dict[str, Any]:
    """Verify instantiated QuantizedLinear bit widths against config.json."""
    from mlx import nn
    from mlx.utils import tree_flatten

    configured = {key: value for key, value in config["quantization"].items() if isinstance(value, dict)}
    default_bits = config["quantization"]["bits"]
    actual: dict[str, int] = {}
    for path, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            actual[path] = module.bits
    mismatches = {
        path: {"expected": configured.get(path, {}).get("bits", default_bits), "actual": bits}
        for path, bits in actual.items()
        if configured.get(path, {}).get("bits", default_bits) != bits
    }
    expected_paths = set(configured)
    return {
        "quantized_linear_or_embedding_modules": len(actual),
        "configured_override_modules": len(expected_paths),
        "override_paths_missing_from_model": sorted(expected_paths - set(actual)),
        "width_mismatches": mismatches,
        "actual_width_histogram": {str(bits): list(actual.values()).count(bits) for bits in sorted(set(actual.values()))},
    }


def greedy_generate(model, tokenizer) -> dict[str, Any]:
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    generated: list[int] = []
    first_logit_top5: list[dict[str, Any]] = []
    started = time.perf_counter()
    for token, logprobs in generate_step(mx.array(prompt_ids), model, max_tokens=128, sampler=make_sampler(0.0)):
        mx.eval(token, logprobs)
        # mlx-lm 0.31.3 yields a Python int here; older releases yield a
        # scalar mx.array.  Keep the probe version-tolerant.
        token_id = int(token.item()) if hasattr(token, "item") else int(token)
        if not generated:
            top = mx.argpartition(-logprobs, kth=4)[-5:]
            mx.eval(top)
            top_ids = [int(item) for item in top.tolist()]
            first_logit_top5 = [
                {"id": item, "logprob": float(logprobs[item].item())}
                for item in top_ids
            ]
        if token_id in tokenizer.eos_token_ids:
            break
        generated.append(token_id)
        if len(generated) >= 128:
            break
    elapsed = time.perf_counter() - started
    text = tokenizer.decode(generated)
    return {
        "template_loaded": tokenizer.has_chat_template,
        "rendered_prompt": rendered,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated,
        "generated_text": text,
        "expected_text_match": text == EXPECTED_TEXT,
        "generation_seconds": elapsed,
        "generation_tokens_per_second": len(generated) / elapsed if elapsed else None,
        "first_position_top5_logprobs": first_logit_top5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=os.environ.get("MACCELERATE_MLX_LM_MODEL"))
    parser.add_argument("--sanitize-workaround", action="store_true")
    parser.add_argument(
        "--reference-token-ids",
        type=Path,
        help="JSON array (or mlx-serve /tokenize response) captured before this run",
    )
    parser.add_argument("--report", type=Path, help="write the JSON result here")
    args = parser.parse_args()
    if not args.model:
        parser.error("pass --model or set MACCELERATE_MLX_LM_MODEL")
    if not args.model.is_dir():
        parser.error(f"not a model directory: {args.model}")

    result: dict[str, Any] = {
        "model": str(args.model.resolve()),
        "versions": installed_versions(),
        "workaround": args.sanitize_workaround,
        "inventory": pack_inventory(args.model),
    }
    observation = install_sanitize_observer(args.sanitize_workaround)
    try:
        from mlx_lm import load

        started = time.perf_counter()
        model, tokenizer = load(str(args.model))
        result["load"] = {"outcome": "success", "seconds": time.perf_counter() - started}
        result["sanitize"] = observation
        config = json.loads((args.model / "config.json").read_text())
        result["quantization"] = observed_quantization(model, config)
        result["generation"] = greedy_generate(model, tokenizer)
        if args.reference_token_ids:
            reference_json = json.loads(args.reference_token_ids.read_text())
            reference_ids = reference_json.get("tokens", reference_json)
            actual_ids = result["generation"]["generated_token_ids"]
            result["generation"]["reference_token_ids"] = reference_ids
            result["generation"]["reference_token_ids_match"] = actual_ids == reference_ids
            result["generation"]["first_token_id_mismatch"] = next(
                (
                    {
                        "position": pos,
                        "mlx_lm": actual_ids[pos] if pos < len(actual_ids) else None,
                        "reference": reference_ids[pos] if pos < len(reference_ids) else None,
                    }
                    for pos in range(max(len(actual_ids), len(reference_ids)))
                    if (actual_ids[pos] if pos < len(actual_ids) else None)
                    != (reference_ids[pos] if pos < len(reference_ids) else None)
                ),
                None,
            )
        result["verdict"] = (
            "PASS"
            if result["generation"]["expected_text_match"]
            and result["generation"].get("reference_token_ids_match", True)
            else "SILENT_MISLOAD"
        )
    except Exception:
        phase = "generation" if result.get("load", {}).get("outcome") == "success" else "load"
        result[phase] = {"outcome": "error", "traceback": traceback.format_exc()}
        result["sanitize"] = observation
        result["verdict"] = "LOAD_FAILURE" if phase == "load" else "PROBE_FAILURE"
    result["peak_rss_gib"] = mac_peak_rss_gib()
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.write_text(rendered + "\n")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
