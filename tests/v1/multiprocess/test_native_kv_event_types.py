# SPDX-License-Identifier: Apache-2.0
"""Native identities preserve their representation across both wire codecs."""

# Standard
from dataclasses import FrozenInstanceError, replace

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.custom_types import (
    EventBinding,
    IPCCacheServerKey,
    KVEventProfile,
    get_customized_decoder,
    get_customized_encoder,
)
from lmcache.v1.multiprocess.kv_event_types import KVEventBatch, KVEventRecord
from lmcache.v1.multiprocess.transport.grpc_impl.method_registry import (
    get_method_codec_registry,
)


@pytest.mark.parametrize(
    "hashes,parent",
    [
        ((b"a", b"b"), None),
        ((-(2**63), 2**63 - 1), -1),
        ((2**64 - 1, 2**63), 2**64 - 2),
    ],
)
def test_native_metadata_wire_representations(hashes, parent):
    profile = KVEventProfile("native-test", 2)
    binding = EventBinding(profile.profile_id, hashes, parent, (0, 1, 2, 3), 2)
    key = IPCCacheServerKey(
        "model",
        2,
        0,
        (0, 1, 2, 3),
        0,
        4,
        "request",
        event_profile=profile,
        event_bindings=(binding,),
    )
    record = KVEventRecord(
        1,
        "stored",
        "CPU",
        "model",
        hashes,
        parent,
        binding.token_ids,
        2,
        profile.profile_id,
        2,
    )
    batch = KVEventBatch(True, 1, 1, False, [record])
    for value in (key, batch):
        decoded = get_customized_decoder(type(value)).decode(
            get_customized_encoder(type(value)).encode(value)
        )
        assert decoded == value
        if isinstance(value, IPCCacheServerKey):
            assert (
                decoded.event_bindings == (binding,)
                and decoded.event_profile == profile
            )
    store = get_method_codec_registry().by_full_name[
        "lmcache.mp.LMCacheDrivenService.Store"
    ]
    decoded = store.request_decoder(
        store.request_encoder((key, 42, [[1]], b"event"), {})
    )[0]
    assert decoded.event_bindings == (binding,) and decoded.event_profile == profile
    assert key == replace(key, event_profile=None, event_bindings=())
    with pytest.raises(FrozenInstanceError):
        record.block_size = 4


@pytest.mark.parametrize(
    "changes",
    [
        {"block_hashes": [1, 2]},
        {"block_hashes": (1, b"b")},
        {"token_ids": (1,)},
        {"block_size": 0},
    ],
)
def test_invalid_native_bindings_are_rejected(changes):
    with pytest.raises(ValueError):
        replace(EventBinding("native-test", (1, 2), None, (0, 1, 2, 3), 2), **changes)
