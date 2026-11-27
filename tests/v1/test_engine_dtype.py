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
    ("declared", "code"),
    [
        ("fp8_per_token_head", "f8pth"),
        ("int8_per_token_head", "i8pth"),
    ],
)
def test_the_per_token_head_modes_get_distinct_short_codes(declared: str, code: str):
    """Their full names overflow the object name next to a long path and salt."""
    assert kv_dtype_tag_code(declared) == code


@pytest.mark.parametrize("declared", ["fp8_e4m3", "custom_dtype", "CUSTOM_DTYPE"])
def test_an_unknown_engine_dtype_is_kept_verbatim(declared: str):
    """Unknown modes must not collide, so their exact spelling survives."""
    assert kv_dtype_tag_code(declared) == declared


def test_unknown_modes_differ_by_case():
    """Lowercasing would merge two modes the engine treats as different."""
    assert kv_dtype_tag_code("CUSTOM_DTYPE") != kv_dtype_tag_code("custom_dtype")


@pytest.mark.parametrize("declared", ["fp8_per_token_head", "int8_per_token_head"])
def test_a_real_key_with_a_pinned_revision_fits_name_max(declared: str):
    """A 32-byte hash, a 64-byte salt and a revision tag must still fit."""
    # First Party
    from lmcache.v1.distributed.api import ObjectKey, ObjectKeyIdentity
    from lmcache.v1.distributed.key_codec import encode_fs_filename

    key = ObjectKey(
        identity=ObjectKeyIdentity(
            chunk_hash=bytes(range(32)),
            kv_rank=1,
            # 40 characters: long enough to matter, short enough that the
            # legacy name format still fits with a real hash.
            model_name="models--Qwen--Qwen3-1.7B/snapshots/" + "a" * 5,
            cache_salt="s" * 64,
            tags={"kv_dtype": kv_dtype_tag_code(declared), "revision": "abc123"},
        )
    )

    name = encode_fs_filename(key).encode()
    assert len(name) < 255


def test_nothing_to_report_is_empty():
    assert kv_dtype_tag_code(None) == ""
    assert KV_DTYPE_TAG_NAME == "lmcache.tag.kv_dtype"
