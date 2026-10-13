# SPDX-License-Identifier: Apache-2.0
"""Precision identity is stable without a serving-engine dependency."""

# Standard
from dataclasses import replace

# Third Party
import pytest

# First Party
from lmcache.v1.cache_format import KVCacheFormat


@pytest.mark.parametrize(
    "precision",
    [
        KVCacheFormat("torch.float16", "auto", (), "auto", "auto"),
        KVCacheFormat("torch.bfloat16", "fp8", (), "auto", "auto"),
        KVCacheFormat("torch.bfloat16", "auto", ("0",), "auto", "auto"),
        KVCacheFormat("torch.bfloat16", "auto", (), "float32", "auto"),
        KVCacheFormat("torch.bfloat16", "auto", (), "auto", "float32"),
    ],
)
def test_precision_fields_change_identity(precision: KVCacheFormat) -> None:
    baseline = KVCacheFormat("torch.bfloat16", "auto", (), "auto", "auto")
    assert baseline.to_cache_format() != precision.to_cache_format()


def test_skip_layers_are_a_set() -> None:
    left = KVCacheFormat("torch.bfloat16", "fp8", ("2", "0", "2"), "auto", "auto")
    right = replace(left, skip_layers=("0", "2"))
    assert left.to_cache_format() == right.to_cache_format()


def test_format_identifier_is_stable() -> None:
    """Pin the namespace contract for clients and persisted cache keys."""
    precision = KVCacheFormat("torch.bfloat16", "auto", (), "auto", "auto")
    assert precision.to_cache_format() == "kv-dtype-v1-62b7f083a2b9e34128fa0d41e63de434"
