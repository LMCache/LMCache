# SPDX-License-Identifier: Apache-2.0
"""Cache identity: the KV storage dtype, derived on the server from the layout.

``float16`` and ``bfloat16`` produce objects of the same size but different
bytes, so size cannot separate them: an object written by one engine is exactly
what the other expects to find. The dtype is therefore part of the identity, and
it is derived where the dtype actually lives -- the layout registered with the
server -- so every connector is covered without sending anything new over the
wire.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.common.attn_window import AttnWindowDesc
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.distributed.api import (
    DEFAULT_SHARD_REPLICATION_DESC,
    KV_DTYPE_TAG_NAME,
    MemoryLayoutDesc,
    ObjectKey,
    get_request_tags,
    ipc_key_to_grouped_object_keys,
    ipc_key_to_object_keys,
    layout_dtype_tags,
)
from lmcache.v1.distributed.key_codec import format_storage_key

_HASH = bytes(range(32))
_MODEL = "org/model"


def _layout(*dtypes: torch.dtype) -> MemoryLayoutDesc:
    return MemoryLayoutDesc(
        shapes=[torch.Size([2, 4]) for _ in dtypes], dtypes=list(dtypes)
    )


def _key(salt: str = "", configs: dict[str, object] | None = None) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name=_MODEL,
        world_size=1,
        worker_id=0,
        token_bytes=bytes(16),
        start=0,
        end=256,
        request_id="req",
        cache_salt=salt,
        request_configs=configs,
    )


class TestLayoutTag:
    def test_single_dtype_uses_its_short_code(self) -> None:
        assert layout_dtype_tags(_layout(torch.bfloat16)) == {KV_DTYPE_TAG_NAME: "bf16"}

    def test_mixed_precision_folds_into_one_sorted_token(self) -> None:
        assert layout_dtype_tags(
            _layout(torch.float16, torch.bfloat16, torch.float16)
        ) == {KV_DTYPE_TAG_NAME: "bf16,f16"}

    def test_an_unknown_dtype_keeps_its_name(self) -> None:
        """Two unknown dtypes must still not collide."""
        assert layout_dtype_tags(_layout(torch.complex64)) == {
            KV_DTYPE_TAG_NAME: "complex64"
        }

    def test_a_layout_without_dtypes_adds_nothing(self) -> None:
        assert layout_dtype_tags(MemoryLayoutDesc(shapes=[], dtypes=[])) == {}


class TestKeysCarryTheTag:
    def test_flat_helper_tags_every_object_group(self) -> None:
        keys = ipc_key_to_object_keys(
            _key(),
            [_HASH],
            [0, 1],
            DEFAULT_SHARD_REPLICATION_DESC,
            group_layout_descs={0: _layout(torch.float16), 1: _layout(torch.bfloat16)},
        )

        assert keys[0][0].identity.tags[KV_DTYPE_TAG_NAME] == "f16"
        assert keys[1][0].identity.tags[KV_DTYPE_TAG_NAME] == "bf16"

    def test_the_flat_helper_requires_layouts(self) -> None:
        """A caller that forgets them must fail, not build untagged names."""
        with pytest.raises(TypeError):
            ipc_key_to_object_keys(  # type: ignore[call-arg]
                _key(), [_HASH], [0], DEFAULT_SHARD_REPLICATION_DESC
            )

    def test_grouped_helper_tags_the_rows(self) -> None:
        rows = ipc_key_to_grouped_object_keys(
            _key(),
            [_HASH],
            [0],
            {0: _layout(torch.bfloat16)},
            AttnWindowDesc(num_chunks_in_sw=[-1]),
            DEFAULT_SHARD_REPLICATION_DESC,
        )

        assert rows[0].keys[0].identity.tags[KV_DTYPE_TAG_NAME] == "bf16"

    def test_request_tags_are_preserved(self) -> None:
        rows = ipc_key_to_grouped_object_keys(
            _key(configs={"lmcache.tag.revision": "abc"}),
            [_HASH],
            [0],
            {0: _layout(torch.float16)},
            AttnWindowDesc(num_chunks_in_sw=[-1]),
            DEFAULT_SHARD_REPLICATION_DESC,
        )

        tags = rows[0].keys[0].identity.tags
        assert tags["revision"] == "abc"
        assert tags[KV_DTYPE_TAG_NAME] == "f16"


class TestIdentityEffect:
    """Tags have to change the storage name, or the reuse still happens."""

    @staticmethod
    def _storage_key(layout: MemoryLayoutDesc) -> str:
        keys = ipc_key_to_grouped_object_keys(
            _key(),
            [_HASH],
            [0],
            {0: layout},
            AttnWindowDesc(num_chunks_in_sw=[-1]),
            DEFAULT_SHARD_REPLICATION_DESC,
        )
        key: ObjectKey = keys[0].keys[0]
        return format_storage_key(key, "{model}/{rank}-{hash}@{salt}")

    def test_two_two_byte_dtypes_produce_different_storage_keys(self) -> None:
        assert self._storage_key(_layout(torch.float16)) != self._storage_key(
            _layout(torch.bfloat16)
        )

    def test_the_same_dtype_is_stable(self) -> None:
        assert self._storage_key(_layout(torch.bfloat16)) == self._storage_key(
            _layout(torch.bfloat16)
        )


def test_tags_reach_get_request_tags() -> None:
    """The helper strips ``lmcache.tag.``; the name is what the server sees."""
    assert get_request_tags({"lmcache.tag.kv_dtype": "bfloat16"})["kv_dtype"] == (
        "bfloat16"
    )
