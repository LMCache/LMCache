# SPDX-License-Identifier: Apache-2.0
"""Focused unit tests for SGLang's fused raw-block MP retrieve client."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, call
import threading

# Third Party
import pytest
import torch

pytest.importorskip("sglang", reason="SGLang is required for its MP adapter tests")

# First Party
from lmcache.integration.sglang import multi_process_adapter as adapter_mod
from lmcache.integration.sglang.multi_process_adapter import (
    CompletionEvent,
    FusedRestoreUndrainedError,
    LMCacheMPConnector,
    _PendingLookup,
)
from lmcache.integration.sglang.sglang_adapter import LoadMetadata
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.engine import (
    FUSED_RAW_BLOCK_RETRIEVE_CAPABILITY,
)


@pytest.fixture
def fused_connector(monkeypatch):
    connector = object.__new__(LMCacheMPConnector)
    connector.tp_size = 1
    connector.worker_id = 0
    connector.page_size = 1
    connector.num_layers = 2
    connector.device = torch.device("cpu")
    connector.model_name = "test-model"
    connector.instance_id = 123
    connector.tp_group = "tp-group"
    connector._mq_timeout = 5.0
    connector._lmcache_chunk_size = 4
    connector._supports_fused_raw_block_retrieve = True
    connector._pending_lookups = {}
    connector._daemon_session_ids = set()
    connector._pending_lookups_lock = threading.Lock()
    connector._fused_final_events = {}
    connector._undrained_fused_requests = set()
    connector._health_event = threading.Event()
    connector._health_event.set()
    connector._event_backend = MagicMock()
    connector._test_final_event = object()
    connector._event_backend.import_event.return_value = connector._test_final_event
    connector._test_current_stream = object()
    monkeypatch.setattr(
        adapter_mod.torch_dev,
        "current_stream",
        MagicMock(return_value=connector._test_current_stream),
    )
    return connector


def _metadata(request_id: str = "request-0") -> LoadMetadata:
    return LoadMetadata(
        token_ids=list(range(10)),
        slot_mapping=torch.arange(8, dtype=torch.int64),
        offset=0,
        prefix_pad=0,
        request_id=request_id,
    )


@pytest.mark.parametrize("retrieved_tokens", [8, 4, 0])
def test_fused_retrieve_waits_completion_on_current_load_stream(
    fused_connector,
    retrieved_tokens,
):
    future = MagicMock()
    future.result.return_value = (
        b"final",
        (retrieved_tokens, True),
    )
    fused_connector._submit_fused_raw_block_retrieve = MagicMock(return_value=future)

    assert fused_connector.retrieve_kv(_metadata()) == retrieved_tokens

    fused_connector._event_backend.wait_event.assert_called_once_with(
        fused_connector._test_final_event,
        fused_connector._test_current_stream,
    )
    fused_connector._event_backend.import_event.assert_called_once_with(
        b"final",
        fused_connector.device,
    )
    fused_connector._event_backend.synchronize_event.assert_not_called()
    submitted = fused_connector._submit_fused_raw_block_retrieve.call_args.kwargs
    assert submitted["offset"] == 0
    assert submitted["aligned_end"] == 8
    assert submitted["block_ids"] == list(range(8))
    assert submitted["prefix_pad"] == 0


def test_fused_retrieve_agrees_tp_before_retaining_and_waiting_completion(
    fused_connector,
):
    trace: list[str] = []
    future = MagicMock()
    future.result.return_value = (
        b"final",
        (8, True),
    )
    fused_connector._submit_fused_raw_block_retrieve = MagicMock(return_value=future)
    fused_connector._global_fused_result = MagicMock(
        side_effect=lambda succeeded, tokens: (
            trace.append("collective") or (succeeded, tokens)
        )
    )
    fused_connector._event_backend.wait_event.side_effect = lambda _event, _stream: (
        trace.append("wait")
    )

    assert fused_connector.retrieve_kv(_metadata()) == 8

    assert trace == ["collective", "wait"]
    fused_connector._event_backend.synchronize_event.assert_not_called()
    retained = fused_connector._fused_final_events["request-0"]
    assert len(retained) == 1
    assert isinstance(retained[0], CompletionEvent)
    assert retained[0]._event is fused_connector._test_final_event  # noqa: SLF001


def test_end_session_synchronizes_final_before_releasing_exporters(
    fused_connector,
    monkeypatch,
):
    trace: list[tuple] = []
    future = MagicMock()
    future.result.return_value = (
        b"final",
        (8, True),
    )
    fused_connector._submit_fused_raw_block_retrieve = MagicMock(return_value=future)
    fused_connector.mq_client = MagicMock()
    fused_connector._event_backend.synchronize_event.side_effect = (
        lambda event, device: trace.append(("synchronize", event, device))
    )

    def send_request(_client, request_type, payload):
        trace.append(("send", request_type, payload))
        return MagicMock()

    monkeypatch.setattr(adapter_mod, "send_lmcache_request", send_request)

    assert fused_connector.retrieve_kv(_metadata()) == 8
    assert "request-0" in fused_connector._fused_final_events

    fused_connector.end_session("request-0")

    assert trace == [
        (
            "synchronize",
            fused_connector._test_final_event,
            fused_connector.device,
        ),
        ("send", RequestType.END_SESSION, ["request-0"]),
    ]
    assert "request-0" not in fused_connector._fused_final_events


def test_tp_failure_synchronizes_local_completion(fused_connector):
    future = MagicMock()
    future.result.return_value = (
        b"final",
        (8, True),
    )
    fused_connector._submit_fused_raw_block_retrieve = MagicMock(return_value=future)
    fused_connector._global_fused_result = MagicMock(return_value=(False, 0))

    with pytest.raises(RuntimeError, match="another TP rank"):
        fused_connector.retrieve_kv(_metadata())

    fused_connector._event_backend.synchronize_event.assert_called_once_with(
        fused_connector._test_final_event,
        fused_connector.device,
    )


def test_completion_event_delegates_to_backend():
    backend = MagicMock()
    native_event = object()
    device = torch.device("cpu")
    stream = object()
    event = CompletionEvent(backend, native_event, device)

    event.wait_on_stream(stream)
    event.synchronize()

    backend.wait_event.assert_called_once_with(native_event, stream)
    backend.synchronize_event.assert_called_once_with(native_event, device)


def test_global_fused_result_reduces_success_and_token_bounds(
    fused_connector,
    monkeypatch,
):
    fused_connector.tp_size = 2

    def reduce_result(result, *, op, group):
        assert result.tolist() == [1, 8, -8]
        assert op == torch.distributed.ReduceOp.MIN
        assert group == "tp-group"
        result.copy_(torch.tensor([1, 4, -8], dtype=torch.int32))

    monkeypatch.setattr(adapter_mod.dist, "all_reduce", reduce_result)

    assert fused_connector._global_fused_result(True, 8) == (False, 4)


def test_asymmetric_capability_is_disabled_by_tp_min_agreement(
    fused_connector,
    monkeypatch,
):
    fused_connector.tp_size = 2

    def reduce_capability(supported, *, op, group):
        assert supported.tolist() == [1]
        assert op == torch.distributed.ReduceOp.MIN
        assert group == "tp-group"
        # This process reached the exact capability, while a peer did not.
        supported.zero_()

    monkeypatch.setattr(adapter_mod.dist, "all_reduce", reduce_capability)

    assert fused_connector._agree_fused_raw_block_capability(True) is False


def test_timeout_drains_response_final_before_collective_and_raise(
    fused_connector,
):
    trace: list[tuple] = []
    future = MagicMock()

    def get_result(timeout=None):
        trace.append(("future", timeout))
        raise TimeoutError("fused timed out")

    future.result.side_effect = get_result
    fused_connector._submit_fused_raw_block_retrieve = MagicMock(return_value=future)
    fused_connector._global_fused_result = MagicMock(
        side_effect=lambda succeeded, tokens: (
            trace.append(("collective", succeeded, tokens)) or (False, 0)
        )
    )
    fused_connector._drain_fused_raw_block_retrieve = MagicMock(
        side_effect=lambda _request_id: trace.append(("drain",))
    )

    with pytest.raises(TimeoutError, match="fused timed out"):
        fused_connector.retrieve_kv(_metadata())

    assert future.result.call_args_list == [
        call(timeout=fused_connector._mq_timeout),
    ]
    assert trace == [
        ("future", fused_connector._mq_timeout),
        ("drain",),
        ("collective", False, 0),
    ]


def test_drain_timeout_surfaces_unsafe_error_and_tracks_request(
    fused_connector,
):
    future = MagicMock()
    future.result.side_effect = TimeoutError("fused timed out")
    fused_connector._submit_fused_raw_block_retrieve = MagicMock(return_value=future)
    fused_connector._drain_fused_raw_block_retrieve = MagicMock(
        side_effect=TimeoutError("drain timed out")
    )

    with pytest.raises(FusedRestoreUndrainedError, match="server-side drain failed"):
        fused_connector.retrieve_kv(_metadata())

    assert fused_connector._undrained_fused_requests == {"request-0"}


def test_unimportable_final_uses_rank_specific_server_drain_before_collective(
    fused_connector,
):
    trace: list[str] = []
    future = MagicMock()
    future.result.return_value = (
        b"final",
        (8, True),
    )
    fused_connector._submit_fused_raw_block_retrieve = MagicMock(return_value=future)
    fused_connector._event_backend.import_event.side_effect = RuntimeError(
        "final import failed"
    )
    fused_connector._drain_fused_raw_block_retrieve = MagicMock(
        side_effect=lambda _request_id: trace.append("drain")
    )
    fused_connector._global_fused_result = MagicMock(
        side_effect=lambda *_args: trace.append("collective") or (False, 0)
    )

    with pytest.raises(RuntimeError, match="final import failed"):
        fused_connector.retrieve_kv(_metadata())

    assert trace == ["drain", "collective"]
    fused_connector._drain_fused_raw_block_retrieve.assert_called_once_with("request-0")


def test_rank_specific_server_drain_rpc_waits_without_timeout(
    fused_connector,
    monkeypatch,
):
    fused_connector.mq_client = MagicMock()
    drain_future = MagicMock()
    drain_future.result.return_value = True
    send_mock = MagicMock(return_value=drain_future)
    monkeypatch.setattr(adapter_mod, "send_lmcache_request", send_mock)

    fused_connector._drain_fused_raw_block_retrieve("request-0")

    send_mock.assert_called_once_with(
        fused_connector.mq_client,
        RequestType.FUSED_RAW_BLOCK_DRAIN,
        ["request-0", fused_connector.worker_id],
    )
    drain_future.result.assert_called_once_with(timeout=fused_connector._mq_timeout)


def test_submit_fused_retrieve_retains_producer_export_event(
    fused_connector,
    monkeypatch,
):
    fused_connector.mq_client = MagicMock()
    producer_event = object()
    fused_connector._event_backend.create_event.return_value = producer_event
    fused_connector._event_backend.export_event.return_value = b"producer"
    messaging_future = MagicMock()
    send_mock = MagicMock(return_value=messaging_future)
    monkeypatch.setattr(adapter_mod, "send_lmcache_request", send_mock)

    result = fused_connector._submit_fused_raw_block_retrieve(
        request_id="request-0",
        token_ids=list(range(10)),
        offset=0,
        aligned_end=8,
        block_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        prefix_pad=2,
    )

    assert result is messaging_future
    assert messaging_future._export_event is producer_event
    request_type, payload = send_mock.call_args.args[1:]
    assert request_type is RequestType.FUSED_RAW_BLOCK_RETRIEVE
    key, instance_id, block_ids, event_handle, prefix_pad = payload
    assert (key.start, key.end, key.request_id) == (0, 8, "request-0")
    assert instance_id == fused_connector.instance_id
    assert block_ids == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert event_handle == b"producer"
    assert prefix_pad == 2


def test_reset_drains_all_final_events_and_ends_every_known_session(
    fused_connector,
    monkeypatch,
):
    fused_connector.mq_client = MagicMock()
    fused_connector._pending_lookups["lookup"] = _PendingLookup(
        token_ids=list(range(8)),
        matched_token_num=8,
        locks_held=True,
    )
    fused_connector._daemon_session_ids.update({"lookup", "store"})
    final_events = [
        CompletionEvent(
            fused_connector._event_backend,
            object(),
            fused_connector.device,
        ),
        CompletionEvent(
            fused_connector._event_backend,
            object(),
            fused_connector.device,
        ),
    ]
    fused_connector._fused_final_events["fused"] = final_events
    fused_connector._daemon_session_ids.add("fused")
    fused_connector._free_lookup_locks = MagicMock()
    sends: list[tuple[RequestType, list]] = []
    monkeypatch.setattr(
        adapter_mod,
        "send_lmcache_request",
        lambda _client, request_type, payload: (
            sends.append((request_type, payload)) or MagicMock()
        ),
    )

    fused_connector.reset()

    assert fused_connector._event_backend.synchronize_event.call_args_list == [
        call(event._event, fused_connector.device)  # noqa: SLF001
        for event in final_events
    ]
    fused_connector._free_lookup_locks.assert_called_once_with(
        list(range(8)),
        0,
        8,
        "lookup",
    )
    assert sends == [
        (RequestType.END_SESSION, ["fused"]),
        (RequestType.END_SESSION, ["lookup"]),
        (RequestType.END_SESSION, ["store"]),
    ]
    assert fused_connector._pending_lookups == {}
    assert fused_connector._daemon_session_ids == set()
    assert fused_connector._fused_final_events == {}
    assert fused_connector._undrained_fused_requests == set()


def test_retaining_policy_same_prompt_hit_uses_generic_retrieve(fused_connector):
    fused_connector._supports_fused_raw_block_retrieve = False
    fused_connector._pending_lookups["request-0"] = _PendingLookup(
        token_ids=list(range(8)),
        matched_token_num=8,
        locks_held=True,
    )
    fused_connector._free_lookup_locks = MagicMock()
    future = MagicMock()
    future.result.return_value = True
    fused_connector._submit_retrieve = MagicMock(return_value=future)
    fused_connector._retrieve_fused_raw_block = MagicMock()
    metadata = _metadata()
    metadata.slot_mapping = torch.tensor(
        [-1, -1, 10, 11, 12, 13, 14, 15],
        dtype=torch.int64,
    )
    metadata.prefix_pad = 2

    assert fused_connector.retrieve_kv(metadata) == 8

    generic_call = fused_connector._submit_retrieve.call_args.kwargs
    assert generic_call["block_ids"] == [0, 0, 10, 11, 12, 13, 14, 15]
    assert generic_call["skip_first_n_tokens"] == 2
    fused_connector._retrieve_fused_raw_block.assert_not_called()


@pytest.mark.parametrize(
    ("capabilities", "expected"),
    [
        ({FUSED_RAW_BLOCK_RETRIEVE_CAPABILITY}, True),
        (set(), False),
        ({"lmcache.unrelated.v1"}, False),
    ],
)
def test_init_enables_fused_only_for_exact_paired_capability(
    monkeypatch,
    capabilities,
    expected,
):
    fake_client = MagicMock()
    monkeypatch.setattr(
        adapter_mod,
        "MessageQueueClient",
        MagicMock(return_value=fake_client),
    )
    monkeypatch.setattr(
        adapter_mod,
        "get_lmcache_chunk_size",
        MagicMock(return_value=4),
    )
    monkeypatch.setattr(
        adapter_mod,
        "get_experimental",
        MagicMock(return_value=capabilities),
    )
    event_backend = MagicMock()
    monkeypatch.setattr(
        adapter_mod,
        "get_event_ipc_backend",
        MagicMock(return_value=event_backend),
    )
    monkeypatch.setattr(
        adapter_mod,
        "_wrap_sglang_kv_caches",
        MagicMock(return_value=[]),
    )
    monkeypatch.setattr(adapter_mod.zmq.Context, "instance", MagicMock())
    register_future = MagicMock()
    register_future.result.return_value = None
    monkeypatch.setattr(
        adapter_mod,
        "send_lmcache_request",
        MagicMock(return_value=register_future),
    )
    heartbeat = MagicMock()
    monkeypatch.setattr(
        adapter_mod,
        "HeartbeatThread",
        MagicMock(return_value=heartbeat),
    )
    kv_tensor = MagicMock()
    kv_tensor.device = torch.device("cpu")

    connector = LMCacheMPConnector(
        sgl_config=SimpleNamespace(model_path="test-model"),
        tp_size=1,
        rank=0,
        page_size=1,
        host="127.0.0.1",
        port=0,
        k_pool=[kv_tensor],
        v_pool=[kv_tensor],
    )

    assert connector.supports_fused_raw_block_retrieve() is expected
    event_backend.check_event_support.assert_called_once_with(torch.device("cpu"))
    heartbeat.start.assert_called_once_with()
