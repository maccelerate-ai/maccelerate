"""The pack-vs-GGUF alignment classifier.

`compare` only reaches a trustworthy verdict if it can tell a *permutation*
(llama.cpp reorders the GDN/SSM head axis) apart from *quantization noise*.
Those are easy to confuse, because sorting any two vectors makes them look
closer — so the classifier is unit-tested here in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

from maccelerate.compare import _leaf, classify_alignment


def test_identical_is_exact():
    a = np.arange(100, dtype=np.float32)
    assert classify_alignment(a, a)[0] == "exact"


def test_permutation_is_reordered():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(4096).astype(np.float32)
    b = a[rng.permutation(a.size)]
    kind, elem, multi = classify_alignment(a, b)
    assert kind == "reordered"
    assert elem > 0.5          # ~100%: grossly misaligned
    assert multi < 0.01        # identical as a multiset


@pytest.mark.parametrize("scale", [0.01, 0.05, 0.1, 0.2, 0.5])
def test_quantization_noise_is_not_reordered(scale):
    """The trap: noisy-but-unpermuted must NOT read as a permutation.

    Sorting is a smoothing operation, so `multi` is always smaller than `elem`
    for any pair. A ratio-only classifier calls a 5 %-noise tensor "reordered" —
    the absolute guards exist for exactly this.
    """
    rng = np.random.default_rng(1)
    a = rng.standard_normal(4096).astype(np.float32)
    noisy = (a + rng.standard_normal(4096).astype(np.float32) * scale).astype(np.float32)
    kind, elem, multi = classify_alignment(a, noisy)
    assert kind == "different", (scale, elem, multi)


def test_shape_mismatch_is_different():
    assert classify_alignment(np.zeros(4), np.zeros(5))[0] == "different"


@pytest.mark.parametrize("name,expect", [
    ("blk.12.ssm_a", "ssm_a"),
    ("blk.0.ssm_conv1d.weight", "ssm_conv1d.weight"),
    ("blk.63.ffn_down.weight", "ffn_down.weight"),
    ("token_embd.weight", "token_embd.weight"),
    ("output.weight", "output.weight"),
])
def test_leaf(name, expect):
    assert _leaf(name) == expect