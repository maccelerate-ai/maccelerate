"""maccelerate — reproduce an Unsloth Dynamic GGUF's per-tensor allocation as a
native Apple MLX pack, from the clean bf16 source, with MTP preserved.

The pipeline, in order:

  `gguf.Gguf`          read a GGUF's metadata + per-tensor type table (stdlib)
  `names.detect_family` map GGUF names to HF source names and pack keys
  `alloc.build_alloc`  the type table -> per-tensor MLX affine allocation
  `imatrix`            official imatrix GGUF -> HF-named activation weights
  `affine`             MLX affine quantization (RTN + imatrix-weighted),
                       streamed in row blocks so a 27B build stays bounded
  `convert`            source + allocation -> sharded MLX SafeTensors pack
  `validate`           structural validation + provenance manifest

What this does and does not claim: the *allocation* (which tensor got how many
bits) is the upstream one; the *numbers* are MLX affine, not llama.cpp k-quant /
IQ codebooks. Read `README.md` before quoting a quality result.
"""

__version__ = "0.1.0"

from .alloc import build_alloc  # noqa: F401
from .convert import convert  # noqa: F401
from .gguf import Gguf  # noqa: F401
from .imatrix import convert_imatrix  # noqa: F401
from .names import detect_family  # noqa: F401
from .validate import validate  # noqa: F401

__all__ = ["build_alloc", "convert", "convert_imatrix", "detect_family",
           "Gguf", "validate", "__version__"]