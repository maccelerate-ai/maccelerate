# maccelerate

Turn a GGUF checkpoint's **per-tensor quantization allocation** into a native,
mixed-width Apple MLX model.

maccelerate reads which bit width the GGUF assigned to each tensor, then
quantizes the matching clean bf16 source weights for MLX. It keeps the MTP head
when the model family supports it and writes a sharded MLX SafeTensors pack with
a provenance manifest.

It is designed for the [mlx-serve](https://github.com/ddalcu/mlx-serve)
runtime. The general method works with GGUF allocation tables; the current
end-to-end implementation supports the Qwen3.5 family and is especially useful
for Unsloth Dynamic GGUFs.

## Published models

Ready-to-use maccelerate MLX packs are available on
[Hugging Face](https://huggingface.co/maccelerate/models).

## Use this when

- You have a supported GGUF checkpoint and its matching clean bf16 source
  model.
- You want a native MLX pack that follows the GGUF's tensor-by-tensor bit
  allocation.
- You serve with `mlx-serve`.
- You want a reproducible conversion with validation and provenance metadata.

## Do not use this when

- You need a byte-for-byte or numerically identical replacement for the GGUF.
  MLX affine quantization cannot reproduce llama.cpp K-quant or IQ codebooks.
- You want to convert arbitrary GGUF weights directly into MLX. maccelerate
  quantizes from the clean bf16 source; it does not dequantize the GGUF.
- You need a vision model. The vision tower is intentionally omitted.
- You require stock `mlx-lm` compatibility for the published Qwen3.8 MTP
  layout. The supported runtime is `mlx-serve`.

## What "reproduce" means

GGUF files store a quantization format for every tensor. For example, one
tensor may be `Q8_0`, another `Q6_K`, and another `IQ4_XS`.

maccelerate preserves that allocation on MLX's affine grid:

| Upstream GGUF type | MLX width |
|---|---:|
| `Q8_0` | 8-bit |
| `Q6_K` | 6-bit |
| `Q5_K` | 5-bit |
| `Q4_K`, `IQ4_*` | 4-bit |

The result has the same per-tensor bit-width recipe, but not the same encoded
numbers.

| Claim | Status |
|---|---|
| Same upstream per-tensor allocation | Yes |
| Quantized from clean bf16 weights, not dequantized GGUF weights | Yes |
| Exact numerical parity with llama.cpp GGUF tensors | No |

Call the output "the same allocation, re-encoded for MLX"—not a lossless GGUF
conversion.

### Why Unsloth Dynamic GGUFs?

The method is not inherently Unsloth-specific: any supported GGUF exposes a
per-tensor type table. Unsloth Dynamic checkpoints are the most useful case
because their allocation is deliberately heterogeneous. A conventional GGUF is
often nearly uniform-width, so preserving its allocation is less distinctive.

The current implementation is not a universal GGUF converter. It requires a
supported architecture, a tensor-name mapping to the matching bf16 source, and
supported ggml types. New architectures need a `Family` implementation in
`names.py`.

## Requirements

- Apple Silicon macOS
- Python 3.13 recommended; other Python versions are not yet verified
- `uv`
- Enough local disk for the bf16 source, GGUF, build files, and output pack
- A matching local copy of:
  - the GGUF checkpoint
  - the clean bf16 Hugging Face model
  - an official imatrix GGUF, if using imatrix-weighted quantization

No model files are downloaded automatically. All CLI paths below are local
paths.

For a 27B conversion, read [Memory safety](#memory-safety) before starting.

## Install

```bash
git clone https://github.com/maccelerate-ai/maccelerate.git
cd maccelerate

uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .

maccelerate --help
```

Install the optional GGUF dequantizer required by `compare`:

```bash
uv pip install -e '.[compare]'
```

## Quick start

Given:

- `Qwen3.8-27B-UD-Q6_K_M.gguf`: the official Unsloth Dynamic GGUF
- `Qwen3.8-27B-bf16/`: the matching clean bf16 source model
- `imatrix_unsloth.gguf`: the matching official imatrix
- `out/...`: a new output directory

run the complete pipeline:

```bash
maccelerate ud \
  --gguf Qwen3.8-27B-UD-Q6_K_M.gguf \
  --variant UD-Q6_K_M \
  --src Qwen3.8-27B-bf16 \
  --imatrix imatrix_unsloth.gguf \
  --out out/Qwen3.8-27B-UD3-Q6_K_M-MLX \
  --gguf-revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 \
  --base-model Qwen/Qwen3.8-27B \
  --hash --finite
```

This:

1. reads the GGUF tensor-type table;
2. writes an allocation manifest;
3. converts the imatrix to HF-named SafeTensors;
4. quantizes the bf16 source into a sharded MLX pack;
5. validates shard closure, hashes, tensor geometry, and finite values;
6. writes `manifest.json` and a model card into the output directory.

The default build workspace is created beside the output directory. Use `--work`
to place it elsewhere.

## CLI overview

| Command | Purpose |
|---|---|
| `maccelerate ud …` | Run allocation, imatrix conversion, MLX conversion, and validation in one command |
| `maccelerate inspect …` | Inspect a GGUF's architecture and tensor-type table |
| `maccelerate alloc …` | Write the GGUF-derived per-tensor allocation JSON |
| `maccelerate imatrix …` | Convert an imatrix GGUF to HF-named SafeTensors |
| `maccelerate convert …` | Quantize bf16 source weights into a sharded MLX pack |
| `maccelerate validate …` | Validate output closure, hashes, geometry, and manifest |
| `maccelerate compare …` | Audit pack/GGUF inventory and allocation correspondence; quantify numeric differences |

## Step-by-step pipeline

Use the individual commands when you want to inspect or retain intermediates:

```bash
maccelerate inspect \
  --gguf Qwen3.8-27B-UD-Q6_K_M.gguf

maccelerate alloc \
  --gguf Qwen3.8-27B-UD-Q6_K_M.gguf \
  --variant UD-Q6_K_M \
  --src Qwen3.8-27B-bf16 \
  --out alloc.json \
  --hash

maccelerate imatrix \
  --imatrix imatrix_unsloth.gguf \
  --out imatrix.safetensors

maccelerate convert \
  --src Qwen3.8-27B-bf16 \
  --dst out/Qwen3.8-27B-UD3-Q6_K_M-MLX \
  --alloc alloc.json \
  --gguf Qwen3.8-27B-UD-Q6_K_M.gguf \
  --imatrix imatrix.safetensors \
  --jobs 3 \
  --row-block-mb 64 \
  --shard-gb 2 \
  --verify

maccelerate validate \
  --model out/Qwen3.8-27B-UD3-Q6_K_M-MLX \
  --alloc alloc.json \
  --src Qwen3.8-27B-bf16 \
  --hash --finite
```

To compare the output with the upstream GGUF:

```bash
maccelerate compare \
  --gguf Qwen3.8-27B-UD-Q6_K_M.gguf \
  --pack out/Qwen3.8-27B-UD3-Q6_K_M-MLX \
  --src Qwen3.8-27B-bf16
```

`compare` can prove matching tensor inventory and per-tensor allocation. It
does not claim equal quantized values, because MLX affine and llama.cpp K-quant
or IQ use different encodings.

## Supported scope

| Area | Current support |
|---|---|
| Runtime | `mlx-serve` |
| Model family | Qwen3.5-family models, including Qwen3.5/3.6/3.8 27B relatives |
| MTP | Preserved when present |
| Vision tower | Not included; text-only output |
| Quantization | Mixed-width MLX affine packs |
| Stock `mlx-lm` | Not supported for the published Qwen3.8 UD3-Q6_K_M layout |

New architectures require a `Family` implementation in `names.py` for GGUF,
source, and MLX-pack tensor-name mapping. The remaining conversion pipeline is
architecture-neutral.

## Quantization details and limitations

MLX's imatrix-weighted packing path supports 2-, 3-, 4-, and 8-bit widths.
Tensors allocated 5 or 6 bits use MLX `mx.quantize` instead.

This means a Q6-family target will typically contain both:

- imatrix-weighted 4- and 8-bit tensors; and
- uncalibrated 5- and 6-bit tensors.

The converter reports the calibrated and RTN tensor counts so this trade-off is
visible in every build.

Additional limitations:

- MLX affine is not numerically equivalent to llama.cpp K-quant or IQ.
- The pack is mixed-width, so mlx-serve's uniform-width NAX MTP profile does
  not apply. MTP still loads through the generic profile.
- `mtp.fc` follows the upstream allocation by default. Use
  `--keep-mtp-fc-dense` to preserve it as bf16.

## Memory safety

Large embeddings and language-model heads are expensive to quantize. For a 27B
model, `embed_tokens` or `lm_head` can be about 5 GB in f32, and
imatrix-weighted search can temporarily require much more memory.

maccelerate is designed to bound this work:

- Quantization runs in row blocks (`--row-block-mb`, default: 64 MB).
- The blocked result is byte-identical to whole-tensor quantization.
- Output shards are flushed at `--shard-gb` (default: 2 GB).
- `--jobs 1` avoids worker forks and large inter-process transfers.

For the Qwen3.8-27B UD-Q6_K_M reference conversion, the measured peak was
7.8 GB RSS and the conversion took 132 seconds. Hardware, MLX version, and
settings will affect your result.

## Reference build

The Qwen3.8-27B UD-Q6_K_M reference build used:

| Item | Value |
|---|---|
| GGUF source | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e01f1bff70c033e0d0028fd02e502` |
| bf16 source | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Output | 22.94 GB in 11 shards |
| Quantized tensors | 506 |
| Effective size | 6.716 bpw |
| Width mix | 201 × 8-bit, 231 × 6-bit, 71 × 5-bit, 3 × 4-bit |
| Calibrated / RTN | 201 / 305 |

The full validation and numerical-comparison findings should live in a separate
reproducibility document rather than define the project's primary interface.

## Tests

```bash
python -m pytest tests/ -q
```

The test suite uses synthetic fixtures and requires no model downloads. It
covers blocked quantization equivalence, packing round trips, a full mixed-width
pipeline smoke test, MTP preservation, name mapping, allocation closure, and
validation.

## License

Apache-2.0. See [LICENSE](./LICENSE).

Third-party attribution, including the mlx-serve quantization core ported by
this project, is in [NOTICE](./NOTICE).
