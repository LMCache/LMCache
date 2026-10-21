# SPDX-License-Identifier: Apache-2.0
"""Cache format isolation and storage round trips."""

# Standard
from dataclasses import replace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.distributed.l2_adapters.bigtable_key_encoder import BigtableL2KeyEncoder
from lmcache.v1.distributed.serde.utils import make_temp_key
from lmcache.v1.mp_coordinator.utils.encoding import decode_key, encode_key
from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry
from lmcache.v1.storage_backend.raw_block.key_codec import (
    decode_object_key,
    encode_object_key,
)


@pytest.mark.parametrize("salt", ["", "alice"])
def test_key_projections_preserve_format_identity(salt: str) -> None:
    legacy = ObjectKey(b"\xab", "org/model", 1, cache_salt=salt)
    fp8 = replace(legacy, cache_format="fp8-e4m3")
    bf16 = replace(legacy, cache_format="bfloat16")
    assert len({legacy, fp8, bf16}) == 3
    assert decode_key((b"\xab", "org/model", 1, 0, salt)) == legacy
    keys = (legacy, fp8, bf16)
    assert len({encode_object_key(key).slot_identity for key in keys}) == 3
    for key in keys:
        assert key.to_encoded_object_key().to_object_key() == key
        assert decode_key(encode_key(key)) == key
        assert decode_object_key(encode_object_key(key).encoded) == key
        assert make_temp_key(key).cache_format == key.cache_format


def test_coexisting_formats_keep_their_own_registered_layouts() -> None:
    registry = LayoutDescRegistry()
    fp8 = MemoryLayoutDesc([torch.Size([16])], [torch.uint8])
    bf16 = MemoryLayoutDesc([torch.Size([16])], [torch.bfloat16])
    registry.register("m", 1, fp8, cache_format="fp8")
    registry.register("m", 1, bf16, cache_format="bf16")
    assert registry.find("m", 1, "fp8") == fp8
    assert registry.find("m", 1, "bf16") == bf16
    assert registry.find("m", 1) is None
    assert registry.find_group_layout_descs("m", 1, "fp8") == {0: fp8}
    registry.unregister("m", 1, "fp8")
    assert registry.find("m", 1, "fp8") is None
    assert registry.find("m", 1, "bf16") == bf16


def test_bigtable_format_cannot_alias_legacy_salt() -> None:
    encoder = BigtableL2KeyEncoder()
    base = ObjectKey(b"ab", "m", 1)
    formatted = replace(base, cache_format="fp8")
    salted = replace(base, cache_salt="format:fp8")
    assert encoder.encode_row_key(formatted) != encoder.encode_row_key(salted)
