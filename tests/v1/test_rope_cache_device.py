# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.v1.compute.positional_encoding import _match_rope_cache_device


class _DummyRope:
    def __init__(self, dtype: torch.dtype):
        self.cos_sin_cache = torch.zeros((2, 2), dtype=dtype)


class _DummyRopeWithMatcher:
    def __init__(self):
        self.called = False

    def _match_cos_sin_cache_dtype(self, reference: torch.Tensor) -> None:
        self.called = True


def test_match_rope_cache_device_updates_dtype():
    rope = _DummyRope(dtype=torch.float16)
    reference = torch.zeros((2, 2), dtype=torch.float32)

    _match_rope_cache_device(rope, reference)

    assert rope.cos_sin_cache.dtype == reference.dtype


def test_match_rope_cache_device_prefers_rope_matcher():
    rope = _DummyRopeWithMatcher()
    reference = torch.zeros((2, 2), dtype=torch.float32)

    _match_rope_cache_device(rope, reference)

    assert rope.called is True
