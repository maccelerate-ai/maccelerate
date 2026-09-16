# Qwen3.8-27B UD3-Q6_K_M: mlx-lm compatibility finding

## Verdict

**Do not describe this checkpoint as stock-mlx-lm compatible.** `mlx_lm.load()`
in mlx-lm 0.31.3 succeeds, but silently applies the raw-Qwen3.8 norm transform
a second time.  Its greedy output is corrupted.  The release runtime is
**mlx-serve**.  mlx-lm can run the trunk only with the small sanitizer shim in
`tests/mlx_lm_compat.py --sanitize-workaround`; it drops the inline MTP head
and therefore cannot use MTP/speculative decoding.

This is a compatibility finding for this converted layout, not a claim that
Qwen3.8 itself is unsupported by mlx-lm.

## Reproduction

The test is deliberately an opt-in script, never collected by pytest.  It
needs a local model directory and Metal, and does not write to that directory.
Create an isolated environment (the `uv venv` command panicked in this managed
macOS session, so Python's standard `venv` was used here):

```bash
python3.13 -m venv /tmp/mlxlm-qwen38-compat-venv
/tmp/mlxlm-qwen38-compat-venv/bin/python -m pip install mlx-lm==0.31.3

# With mlx-serve stopped, so only one 21 GB model is resident:
MACCELERATE_MLX_LM_MODEL=/path/to/Qwen3.8-27B-UD3-Q6_K_M-MLX \
  /tmp/mlxlm-qwen38-compat-venv/bin/python tests/mlx_lm_compat.py

# The only known working mlx-lm invocation for this pack:
MACCELERATE_MLX_LM_MODEL=/path/to/Qwen3.8-27B-UD3-Q6_K_M-MLX \
  /tmp/mlxlm-qwen38-compat-venv/bin/python tests/mlx_lm_compat.py \
  --sanitize-workaround --reference-token-ids /tmp/mlx-serve-token-ids.json
```

Capture `/tmp/mlx-serve-token-ids.json` while the verified mlx-serve reference
is running, then stop it before the mlx-lm process:

```bash
curl -sS http://127.0.0.1:11235/tokenize \
  -H 'Content-Type: application/json' \
  --data '{"content":"1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40"}' \
  > /tmp/mlx-serve-token-ids.json
```

The probe applies the model's chat template with `enable_thinking=False`, uses
argmax (`temperature=0`), prints generated IDs and a first-position top-5
logprob sample, checks instantiated quantized module widths, and reports peak
RSS.  A nonzero exit means the canonical loader failed or silently misloaded.

## Observed environment and evidence

| item | result |
| --- | --- |
| host | Apple M4 Max, 64 GB unified memory |
| Python / mlx-lm / MLX | 3.13.5 / 0.31.3 / 0.32.2 |
| canonical `from mlx_lm import load; load(model_dir)` | loads successfully in 2.33 s |
| canonical peak process RSS | 17.53 GiB (`/usr/bin/time -l`); Metal peak footprint 23.40 GB |
| canonical MTP handling | sees 31 `language_model.mtp.*` tensors, then drops all 31 |
| quantization handling | correct per-module widths: 3 × 4-bit, 71 × 5-bit, 225 × 6-bit, 199 × 8-bit trunk modules; no width mismatch |
| canonical generation | **silent mis-load**: starts `/ 琭\n Default悉> …`; 30 IDs then EOS; 10.63 generated token/s |
| cause | `qwen3_5.TextModel.sanitize()` treats any MTP key as evidence that trunk norms need `+1`; this pack has already folded +1 into the 161 trunk norms, so the transform is applied twice |
| shim generation | 110 IDs, exactly equal to mlx-serve's reference; 14.73 generated token/s; 16.91 GiB peak RSS |
| reference mlx-serve | 26.9.2-dev / MLX 0.32.2; MTP active; correct 110-token response at 39.91 generated token/s |

The matching 110 IDs begin `[16, 220, 17, 220, …]` and end `[18, 24, 220,
19, 15]`; the shim probe reports `reference_token_ids_match: true` and no
first mismatch.  The unpatched first-position top candidates had nearly flat,
very low logprobs (−40.75 to −41.00), consistent with the corrupted output.

## Root cause and smallest workaround

mlx-lm 0.31.3 already implements `qwen3_5` and understands the pack's
per-module `quantization` map.  It is neither an architecture-support failure
nor a fallback to uniform 6-bit.  The exact bad branch is in
`mlx_lm.models.qwen3_5.TextModel.sanitize`:

1. `has_mtp_weights = any("mtp." in k for k in weights)` is true.
2. It sets `should_shift_norm_weights` true and drops MTP weights.
3. It adds `+1` to trunk norm vectors, although this converted pack already
   stores them in mlx-lm/mlx-serve layout.

The shim detects converted conv1d layout (`[C, K, 1]`), removes MTP weights
*before* the upstream sanitizer makes that decision, then delegates to it.
It is in-process only, does not change weights, and intentionally gives mlx-lm
no MTP capability.  It is a useful diagnostic/workaround, but not a good
first-run instruction for a model card.  Publish mlx-serve as the supported
runtime and state that stock mlx-lm 0.31.3 is unsafe for this pack.
