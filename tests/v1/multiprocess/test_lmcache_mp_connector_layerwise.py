# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from typing import Literal, cast

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm import lmcache_mp_connector as mp_connector


class _NullStream:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> Literal[False]:
        return False


class _FakeEvent:
    def __init__(self, interprocess: bool) -> None:
        self.interprocess = interprocess
        self.recorded = False

    def record(self) -> None:
        self.recorded = True


class _FakeFuture:
    def __init__(self, result: bool = True) -> None:
        self._result = result
        self.result_calls = 0
        self.result_on_current_stream_calls = 0
        self.timeouts: list[float | None] = []

    def result(self, timeout: float | None = None) -> bool:
        self.result_calls += 1
        self.timeouts.append(timeout)
        return self._result

    def result_on_current_stream(self, timeout: float | None = None) -> bool:
        self.result_on_current_stream_calls += 1
        self.timeouts.append(timeout)
        return self._result


class _FakeWorkerAdapter:
    def __init__(self) -> None:
        self.use_mla = False
        self.is_first_rank_of_pp_group = True
        self._mq_timeout = 7.0
        self.error_block_ids: set[int] = set()
        self.retrieve_futures: dict[str, tuple[_FakeFuture, list[int]]] = {}
        self.retrieve_batches: list[
            tuple[
                list[str],
                list[mp_connector.LoadStoreOp],
                _FakeEvent,
                list[str] | None,
            ]
        ] = []
        self.store_batches: list[
            tuple[
                list[str],
                list[mp_connector.LoadStoreOp],
                _FakeEvent,
                list[str] | None,
            ]
        ] = []
        self.finished_return: tuple[set[str] | None, set[str] | None] = (
            set(),
            set(),
        )

    def batched_submit_retrieve_requests(
        self,
        request_ids: list[str],
        ops: list[mp_connector.LoadStoreOp],
        event: _FakeEvent,
        cache_salts: list[str] | None = None,
    ) -> None:
        self.retrieve_batches.append((request_ids, ops, event, cache_salts))
        for request_id, op in zip(request_ids, ops, strict=False):
            self.retrieve_futures[request_id] = (
                _FakeFuture(result=True),
                list(op.block_ids),
            )

    def batched_submit_store_requests(
        self,
        request_ids: list[str],
        ops: list[mp_connector.LoadStoreOp],
        event: _FakeEvent,
        cache_salts: list[str] | None = None,
    ) -> None:
        self.store_batches.append((request_ids, ops, event, cache_salts))

    def get_finished(
        self,
        _finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        return self.finished_return


def _patch_torch_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stream(_stream_obj: object) -> _NullStream:
        return _NullStream()

    monkeypatch.setattr(
        mp_connector,
        "torch_dev",
        SimpleNamespace(
            current_stream=lambda: object(),
            stream=_stream,
            Event=_FakeEvent,
        ),
    )


def _make_metadata(
    requests: list[mp_connector.LMCacheMPRequestMetadata],
) -> mp_connector.LMCacheMPConnectorMetadata:
    metadata = mp_connector.LMCacheMPConnectorMetadata()
    for request in requests:
        metadata.add_request_metadata(request)
    return metadata


def _make_request_metadata(
    request_id: str,
    direction: Literal["STORE", "RETRIEVE"],
    op: mp_connector.LoadStoreOp,
    cache_salt: str = "",
) -> mp_connector.LMCacheMPRequestMetadata:
    return mp_connector.LMCacheMPRequestMetadata(
        request_id=request_id,
        direction=direction,
        op=op,
        cache_salt=cache_salt,
    )


def _make_connector(
    metadata: mp_connector.LMCacheMPConnectorMetadata,
    worker: _FakeWorkerAdapter,
) -> mp_connector.LMCacheMPConnector:
    connector = mp_connector.LMCacheMPConnector.__new__(mp_connector.LMCacheMPConnector)
    connector._connector_metadata = metadata
    connector.worker_adapter = cast(mp_connector.LMCacheMPWorkerAdapter, worker)
    connector._layerwise_waited_retrieve_ids = set()
    return connector


@pytest.mark.parametrize(
    ("use_mla", "expected"),
    [
        (False, (8, 5)),
        (True, (2, 1)),
    ],
)
def test_mp_connector_mla_rank_normalization(
    use_mla: bool,
    expected: tuple[int, int],
) -> None:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(use_mla=use_mla),
        parallel_config=SimpleNamespace(
            world_size=8,
            rank=5,
            tensor_parallel_size=4,
        ),
    )

    assert (
        mp_connector.extract_world_size_and_kv_rank(
            world_size=8,
            rank=5,
            vllm_config=cast(mp_connector.VllmConfig, vllm_config),
        )
        == expected
    )


def test_mp_connector_layerwise_lifecycle_waits_once_and_finalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_torch_dev(monkeypatch)
    timing_logs: list[tuple[str, tuple[object, ...]]] = []

    def _record_info(message: str, *args: object, **_kwargs: object) -> None:
        timing_logs.append((message, args))

    monkeypatch.setenv("LMCACHE_MP_CONNECTOR_TIMING", "1")
    monkeypatch.setattr(mp_connector.logger, "info", _record_info)
    worker = _FakeWorkerAdapter()
    retrieve_op = mp_connector.LoadStoreOp(
        token_ids=[1, 2, 3, 4],
        block_ids=[2, 3],
        start=0,
        end=4,
    )
    store_op = mp_connector.LoadStoreOp(
        token_ids=[5, 6, 7, 8],
        block_ids=[4, 5],
        start=0,
        end=4,
    )
    metadata = _make_metadata(
        [
            _make_request_metadata(
                "req-load",
                "RETRIEVE",
                retrieve_op,
                cache_salt="load-salt",
            ),
            _make_request_metadata(
                "req-store",
                "STORE",
                store_op,
                cache_salt="store-salt",
            ),
        ]
    )
    connector = _make_connector(metadata, worker)

    connector.start_load_kv(SimpleNamespace())
    assert len(worker.retrieve_batches) == 1
    retrieve_ids, retrieve_ops, retrieve_event, retrieve_salts = (
        worker.retrieve_batches[0]
    )
    assert retrieve_ids == ["req-load"]
    assert retrieve_ops == [retrieve_op]
    assert retrieve_event.recorded is True
    assert retrieve_salts == ["load-salt"]

    future = worker.retrieve_futures["req-load"][0]
    connector.wait_for_layer_load("layer.0")
    connector.wait_for_layer_load("layer.1")
    assert future.result_calls == 0
    assert future.result_on_current_stream_calls == 1
    assert future.timeouts == [worker._mq_timeout]
    assert connector._layerwise_waited_retrieve_ids == {"req-load"}
    assert len(timing_logs) == 1
    message, args = timing_logs[0]
    assert "LMCache MP layerwise retrieve wait" in message
    assert args[0] == "req-load"
    assert args[1] == "layer.0"
    assert isinstance(args[2], int)
    assert args[3] == len(retrieve_op.block_ids)
    assert args[4] is True
    assert args[5] is True

    worker.finished_return = (set(), {"req-load"})
    assert connector.get_finished({"req-load"}) == (set(), {"req-load"})
    assert connector._layerwise_waited_retrieve_ids == set()

    connector.save_kv_layer("layer.0", torch.zeros(1), None)
    assert worker.store_batches == []

    connector.wait_for_save()
    assert len(worker.store_batches) == 1
    store_ids, store_ops, store_event, store_salts = worker.store_batches[0]
    assert store_ids == ["req-store"]
    assert store_ops == [store_op]
    assert store_event.recorded is True
    assert store_salts == ["store-salt"]


def test_mp_connector_layerwise_failed_retrieve_records_error_blocks() -> None:
    worker = _FakeWorkerAdapter()
    worker.retrieve_futures["req-load"] = (_FakeFuture(result=False), [7, 8])
    connector = _make_connector(_make_metadata([]), worker)

    connector.wait_for_layer_load("layer.0")

    future = worker.retrieve_futures["req-load"][0]
    assert future.result_calls == 0
    assert future.result_on_current_stream_calls == 1
    assert worker.error_block_ids == {7, 8}
