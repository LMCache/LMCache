# SPDX-License-Identifier: Apache-2.0
"""Wire and coverage contract used by the MP Holemiss integration."""

import msgspec
import pytest
from lmcache.v1.common.custom_types import RetrieveCoverage


@pytest.mark.parametrize(
    "loaded,failed", [([0, 2], [1]), ([1, 2], [0]), ([0, 1], [2]), ([0, 1, 2], [])]
)
def test_coverage_roundtrip(loaded: list[int], failed: list[int]) -> None:
    response = RetrieveCoverage(3, loaded, failed)
    restored = msgspec.msgpack.decode(
        msgspec.msgpack.encode((b"event", response)),
        type=tuple[bytes, bool | RetrieveCoverage],
    )
    assert restored[0] == b"event"
    assert restored[1] == response
    assert bool(response) == (not failed)
    assert response.failed_block_ids([20, 21, 22], 1) == [20 + i for i in failed]


def test_legacy_boolean_reply_remains_supported() -> None:
    assert msgspec.msgpack.decode(
        msgspec.msgpack.encode((b"", False)), type=tuple[bytes, bool | RetrieveCoverage]
    ) == (b"", False)


def test_hybrid_coverage_roundtrip_certifies_the_required_checkpoint() -> None:
    """A historical attention hole must not imply a lost recurrent checkpoint."""
    from lmcache.v1.common.custom_types import RetrieveGroupCoverage

    response = RetrieveCoverage(
        3,
        [0, 2],
        [1],
        groups=[
            RetrieveGroupCoverage(0, [2], "attention", 0, [0, 2], [1]),
            RetrieveGroupCoverage(1, [0, 1], "recurrent", 2, [2], []),
        ],
    )
    response.validate()
    restored = msgspec.msgpack.decode(
        msgspec.msgpack.encode((b"event", response)),
        type=tuple[bytes, bool | RetrieveCoverage],
    )
    assert restored == (b"event", response)
    assert response.attention_only_failure
    assert not bool(response)


def test_missing_recurrent_checkpoint_is_never_attention_only_failure() -> None:
    from lmcache.v1.common.custom_types import RetrieveGroupCoverage

    response = RetrieveCoverage(
        3,
        [0],
        [1, 2],
        groups=[
            RetrieveGroupCoverage(0, [0], "attention", 0, [0, 2], [1]),
            RetrieveGroupCoverage(1, [1], "recurrent", 2, [], [2]),
        ],
    )
    response.validate()
    assert not response.attention_only_failure


@pytest.mark.parametrize("invalid", ["aggregate", "checkpoint", "identity"])
def test_hybrid_coverage_rejects_inconsistent_group_evidence(invalid: str) -> None:
    from lmcache.v1.common.custom_types import RetrieveGroupCoverage

    attention = RetrieveGroupCoverage(0, [0], "attention", 0, [0, 2], [1])
    checkpoint = RetrieveGroupCoverage(1, [1], "recurrent", 2, [2], [])
    if invalid == "aggregate":
        response = RetrieveCoverage(3, [0, 1, 2], [], [attention, checkpoint])
    elif invalid == "checkpoint":
        response = RetrieveCoverage(
            3,
            [0, 2],
            [1],
            [
                attention,
                RetrieveGroupCoverage(1, [1], "recurrent", 2, [], []),
            ],
        )
    else:
        response = RetrieveCoverage(3, [0, 2], [1], [attention, attention])
    with pytest.raises(ValueError):
        response.validate()


def test_incomplete_coverage_is_rejected() -> None:
    with pytest.raises(ValueError):
        RetrieveCoverage(3, [0], [1]).validate()
    with pytest.raises(ValueError):
        RetrieveCoverage(3, [0, 1], [1, 2]).validate()


@pytest.mark.parametrize("hole", [0, 1, 2])
def test_missing_chunk_leaves_its_gpu_destination_untouched(
    monkeypatch, hole: int
) -> None:
    from types import SimpleNamespace
    import lmcache.lmcache_native as native
    from lmcache.v1.multiprocess.object_group_transfer import dispatch

    objects = [object(), object(), object()]
    destinations = {20: -1, 21: -1, 22: -1}

    def copy(
        context,
        blocks,
        memory_objs,
        group,
        batch,
        skip,
        direction,
        *,
        transfer_key,
        chunk_range,
    ):
        lo, hi = chunk_range
        for index in range(lo, hi):
            destinations[blocks[0][index]] = objects.index(memory_objs[index]) + 10

    monkeypatch.setattr(dispatch, "batch_memcpy_available", lambda: False)
    monkeypatch.setattr(dispatch, "native_plan_available", lambda: True)
    monkeypatch.setattr(dispatch, "run_object_group_transfer_plan", copy)
    dispatch.transfer_kv_per_object_group(
        SimpleNamespace(),
        [[20, 21, 22]],
        objects,
        0,
        8,
        0,
        native.TransferDirection.H2D,
        transfer_key="test",
        omitted_chunks=[hole],
    )
    assert destinations == {20 + i: (-1 if i == hole else i + 10) for i in range(3)}


def test_coverage_is_not_complete_until_gpu_event_is_ready() -> None:
    from unittest.mock import MagicMock
    from lmcache.v1.transport.futures import MessagingFuture

    backend = MagicMock()
    backend.query_event.return_value = False
    raw = MessagingFuture()
    future = raw.to_device_future(device=0, event_backend=backend)
    coverage = RetrieveCoverage(3, [0, 2], [1])
    raw.set_result((b"gpu-completion", coverage))
    assert not future.query()
    backend.query_event.return_value = True
    assert future.query()
    assert future.result() == coverage
    backend.import_event.assert_called_once_with(b"gpu-completion", 0)


@pytest.mark.parametrize("outcome", [True, False, RetrieveCoverage(3, [0, 2], [1])])
def test_grpc_retrieve_wire_roundtrip(outcome) -> None:
    from lmcache.v1.common.custom_types import RetrieveReply
    from lmcache.v1.transport.grpc._proto_gen.lmcache_driven_service_pb2 import (
        RetrieveResponse,
    )
    from lmcache.v1.transport.grpc.proto_codec import (
        compile_response_encoder_for_type,
        compile_response_decoder_for_type,
    )

    encode = compile_response_encoder_for_type(RetrieveResponse, RetrieveReply)
    decode = compile_response_decoder_for_type(RetrieveResponse, RetrieveReply)
    wire = encode((b"event", outcome))
    restored = RetrieveResponse.FromString(wire.SerializeToString())
    assert decode(restored) == (b"event", outcome)
