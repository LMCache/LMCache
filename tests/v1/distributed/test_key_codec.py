# SPDX-License-Identifier: Apache-2.0
"""Shared codecs preserve established backend names and escaping."""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.key_codec import (
    decode_fs_filename,
    decode_storage_key,
    encode_fs_filename,
    encode_storage_key,
    format_storage_key,
    storage_shard,
)
from lmcache.v1.distributed.l2_adapters.bigtable_key_encoder import BigtableL2KeyEncoder
from lmcache.v1.distributed.l2_adapters.raw_block.key_codec import (
    decode_object_key,
    object_key_to_string,
)


@pytest.mark.parametrize("salt", ["", "tenant"])
def test_legacy_backend_bytes_and_recovery(salt: str) -> None:
    key = ObjectKey(b"\xde\xad\xbe\xef", "org/model", 42, 3, salt)
    suffix = f"@{salt}" if salt else ""
    assert encode_storage_key(key) == f"org/model@0000002a@3@deadbeef{suffix}"
    assert decode_storage_key(encode_storage_key(key)) == key
    assert (
        encode_fs_filename(key) == f"org-SEP-model@0x0000002a@3@deadbeef{suffix}.data"
    )
    assert decode_fs_filename(encode_fs_filename(key)) == key
    assert object_key_to_string(key) == f"org%2Fmodel@0x0000002a@3@deadbeef{suffix}"
    assert decode_object_key(object_key_to_string(key)) == key
    assert storage_shard(key) == ("de", "ad")
    assert (
        format_storage_key(
            key, "{model}_{rank}_{group}_{hash}{suffix}.bin", model_separator="--"
        )
        == f"org--model_0000002a_3_deadbeef{suffix}.bin"
    )
    assert BigtableL2KeyEncoder().encode_row_key(key) == (
        f"dead@org/model@0000002a@3@deadbeef{suffix}".encode()
    )
    assert BigtableL2KeyEncoder(layer_group_size=4).encode_row_key(key) == (
        f"dead@org/model@lg4@0000002a@3@deadbeef{suffix}".encode()
    )
