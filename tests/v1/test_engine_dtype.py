# SPDX-License-Identifier: Apache-2.0
"""The KV dtype tag value shared by the engine integrations."""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.common.engine_dtype import KV_DTYPE_TAG_NAME, kv_dtype_tag_code


def test_torch_dtypes_get_short_codes():
    assert kv_dtype_tag_code(torch.bfloat16) == "bf16"
    assert kv_dtype_tag_code(torch.float16) == "f16"
    assert kv_dtype_tag_code(torch.float8_e4m3fn) == "f8e4m3"


def test_a_dtype_name_is_accepted_too():
    assert kv_dtype_tag_code("torch.bfloat16") == "bf16"
    assert kv_dtype_tag_code("float16") == "f16"


@pytest.mark.parametrize(
    "declared",
    ["fp8_per_token_head", "int8_per_token_head", "fp8_e4m3", "custom_dtype"],
)
def test_an_engine_declared_dtype_is_kept_verbatim(declared: str):
    """Per-token-head modes share a page size but not a layout."""
    assert kv_dtype_tag_code(declared) == declared


def test_nothing_to_report_is_empty():
    assert kv_dtype_tag_code(None) == ""
    assert KV_DTYPE_TAG_NAME == "lmcache.tag.kv_dtype"
