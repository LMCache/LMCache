# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the native ATOM multiprocess integration."""

# Standard
from typing import Any, cast
from unittest.mock import MagicMock
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.atom import multi_process_adapter as atom_adapter
from lmcache.integration.atom.multi_process_adapter import (
    AtomMPParallelConfig,
    AtomMPSchedulerAdapter,
    AtomMPTransferSpec,
    AtomMPWorkerAdapter,
)
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format import detect_format
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocol import RequestType
import lmcache.lmcache_native as lmcache_native


class _FakeHeartbeatThread:
    """No-thread heartbeat double used by worker adapter tests."""

    instances: list["_FakeHeartbeatThread"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.health_event = args[1]
        self.started = False
        self.stop_requested = False
        self.recover_callback = lambda: True
        self.unhealthy_callback = lambda: None
        self.healthy_callback = self._set_healthy
        self.instances.append(self)

    def _set_healthy(self) -> bool:
        self.health_event.set()
        return True

    def register_recover_callback(self, callback: Any) -> None:
        self.recover_callback = callback

    def register_unhealthy_callback(self, callback: Any) -> None:
        self.unhealthy_callback = callback

    def register_healthy_callback(self, callback: Any) -> None:
        self.healthy_callback = callback

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stop_requested = True
        self.started = False

    def stop_and_wait(self) -> None:
        self.stop()

    def mark_unhealthy(self) -> None:
        self.unhealthy_callback()
        self.health_event.clear()

    def recover(self) -> bool:
        recovered = self.recover_callback()
        if recovered:
            recovered = self.healthy_callback()
        return recovered


class _FakeEvent:
    """Minimal event accepted by the transfer-context protocol."""

    def ipc_handle(self) -> bytes:
        return b"atom-event"

    def wait(self, stream: object | None = None) -> None:
        del stream


@pytest.fixture
def worker_with_transfer_context(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[AtomMPWorkerAdapter, MagicMock, dict[str, torch.Tensor]]:
    """Construct a worker with its MQ and transfer boundaries stubbed."""
    client = MagicMock(name="mq_client")
    transfer_context = MagicMock(name="transfer_context")
    monkeypatch.setattr(atom_adapter, "MessageQueueClient", lambda *args: client)
    monkeypatch.setattr(atom_adapter, "_get_chunk_size", lambda *args: 256)
    monkeypatch.setattr(
        atom_adapter,
        "create_transfer_context",
        lambda *args, **kwargs: transfer_context,
    )
    monkeypatch.setattr(atom_adapter, "_HeartbeatThread", _FakeHeartbeatThread)
    _FakeHeartbeatThread.instances.clear()

    worker = AtomMPWorkerAdapter(
        server_url="tcp://127.0.0.1:5555",
        context=MagicMock(name="zmq_context"),
        model_name="atom-test-model",
        block_size=64,
        parallel_config=AtomMPParallelConfig(
            world_size=2,
            worker_id=1,
            tp_size=2,
        ),
        transfer_mode="lmcache_driven",
    )
    caches = {
        "layer.0.latent": torch.empty(4, 64, 576),
        "layer.0.index": torch.empty(4, 64, 144),
    }
    return worker, transfer_context, caches


def _atom_groups() -> list[EngineGroupInfo]:
    """Return ATOM latent/index kernel groups sharing one block namespace."""
    return [
        EngineGroupInfo(
            engine_group_id=0,
            layer_indices=(0,),
            tokens_per_block=64,
        ),
        EngineGroupInfo(
            engine_group_id=0,
            layer_indices=(1,),
            tokens_per_block=64,
        ),
    ]


def _make_adapter(
    adapter_kind: str,
) -> AtomMPSchedulerAdapter | AtomMPWorkerAdapter:
    """Construct the requested ATOM adapter with shared test parameters."""
    if adapter_kind == "scheduler":
        return AtomMPSchedulerAdapter(
            server_url="tcp://127.0.0.1:5555",
            context=MagicMock(name="zmq_context"),
            model_name="atom-test-model",
            block_size=64,
            parallel_config=AtomMPParallelConfig(2, 1, 2),
        )
    return AtomMPWorkerAdapter(
        server_url="tcp://127.0.0.1:5555",
        context=MagicMock(name="zmq_context"),
        model_name="atom-test-model",
        block_size=64,
        parallel_config=AtomMPParallelConfig(2, 1, 2),
    )


def _mock_client(worker: AtomMPWorkerAdapter) -> MagicMock:
    """Expose the fixture-provided MQ mock with its assertion helpers."""
    return cast(MagicMock, worker._client)


def test_atom_worker_registers_native_engine_type(
    worker_with_transfer_context: tuple[
        AtomMPWorkerAdapter, MagicMock, dict[str, torch.Tensor]
    ],
) -> None:
    """Registration identifies ATOM and forwards its physical cache groups."""
    worker, transfer_context, caches = worker_with_transfer_context
    groups = _atom_groups()

    worker.register_kv_caches(caches, engine_group_infos=groups)

    register_call = transfer_context.register.call_args
    assert register_call.kwargs["engine_type"] is EngineType.ATOM
    assert register_call.kwargs["layout_hints"] == {}
    assert register_call.kwargs["engine_group_infos"] == groups


@pytest.mark.parametrize(
    ("method_name", "submit_name"),
    [
        ("submit_store_request", "submit_store"),
        ("submit_retrieve_request", "submit_retrieve"),
    ],
)
def test_atom_submit_returns_future_and_expands_physical_groups(
    worker_with_transfer_context: tuple[
        AtomMPWorkerAdapter, MagicMock, dict[str, torch.Tensor]
    ],
    method_name: str,
    submit_name: str,
) -> None:
    """One ATOM block list fans out to latent/index and returns its future."""
    worker, transfer_context, caches = worker_with_transfer_context
    worker.register_kv_caches(caches, engine_group_infos=_atom_groups())
    future: MessagingFuture[bool] = MessagingFuture()
    getattr(transfer_context, submit_name).return_value = future
    spec = AtomMPTransferSpec(
        token_ids=list(range(256)),
        block_ids=[[7, 8, 9, 10]],
        start=0,
        end=256,
    )
    event = _FakeEvent()

    returned = getattr(worker, method_name)(
        "request-1",
        spec,
        event,
    )

    assert returned is future
    assert event in returned._retained_references
    submit_call = getattr(transfer_context, submit_name).call_args
    assert submit_call.args[4] == [[7, 8, 9, 10], [7, 8, 9, 10]]

    future.set_result(True)
    assert returned.result(timeout=0) is True


def test_atom_worker_reregisters_without_wrapping_inflight_future(
    worker_with_transfer_context: tuple[
        AtomMPWorkerAdapter, MagicMock, dict[str, torch.Tensor]
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery republishes cache registration before health becomes visible."""
    worker, transfer_context, caches = worker_with_transfer_context
    worker.register_kv_caches(caches, engine_group_infos=_atom_groups())
    future: MessagingFuture[bool] = MessagingFuture()
    transfer_context.submit_store.return_value = future
    returned = worker.submit_store_request(
        "request-1",
        AtomMPTransferSpec(
            token_ids=list(range(256)),
            block_ids=[[0, 1, 2, 3]],
            end=256,
        ),
        _FakeEvent(),
    )
    heartbeat = _FakeHeartbeatThread.instances[0]
    recovery_context = MagicMock(name="recovery_transfer_context")
    monkeypatch.setattr(
        atom_adapter,
        "create_transfer_context",
        lambda *args, **kwargs: recovery_context,
    )

    heartbeat.mark_unhealthy()

    assert worker.is_healthy is False
    assert returned is future
    assert returned.query() is False
    assert heartbeat.recover() is True
    assert worker.is_healthy is True
    recovery_context.register.assert_called_once()
    transfer_context.close.assert_called_once_with()
    assert worker._transfer_context is recovery_context

    future.set_result(True)
    assert returned.result(timeout=0) is True


def test_atom_heartbeat_runs_recovery_before_publishing_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful ping only restores health after re-registration succeeds."""
    ping_results = iter([False, True])

    def send_request(
        _client: Any,
        request_type: RequestType,
        _payloads: list[Any],
    ) -> MessagingFuture[Any]:
        future: MessagingFuture[Any] = MessagingFuture()
        result = next(ping_results) if request_type is RequestType.PING else True
        future.set_result(result)
        return future

    monkeypatch.setattr(atom_adapter, "_send_request", send_request)
    health_event = threading.Event()
    health_event.set()
    recover = MagicMock(return_value=True)
    unhealthy = MagicMock()
    heartbeat = atom_adapter._HeartbeatThread(
        MagicMock(),
        health_event,
        instance_id=7,
        interval=1.0,
    )
    heartbeat.register_recover_callback(recover)
    heartbeat.register_unhealthy_callback(unhealthy)

    heartbeat._execute()
    assert health_event.is_set() is False
    unhealthy.assert_called_once_with()

    heartbeat._execute()
    assert health_event.is_set() is True
    recover.assert_called_once_with()


@pytest.mark.parametrize("adapter_kind", ["scheduler", "worker"])
@pytest.mark.parametrize("failure_kind", ["chunk_query", "misaligned_chunk"])
def test_atom_adapter_constructor_closes_client_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    adapter_kind: str,
    failure_kind: str,
) -> None:
    """GET_CHUNK_SIZE and validation failures do not leak the MQ client."""
    client = MagicMock(name="mq_client")
    monkeypatch.setattr(atom_adapter, "MessageQueueClient", lambda *args: client)
    if failure_kind == "chunk_query":
        monkeypatch.setattr(
            atom_adapter,
            "_get_chunk_size",
            MagicMock(side_effect=RuntimeError("chunk query failed")),
        )
        error: type[Exception] = RuntimeError
    else:
        monkeypatch.setattr(atom_adapter, "_get_chunk_size", lambda *args: 255)
        error = ValueError

    with pytest.raises(error):
        _make_adapter(adapter_kind)

    client.close.assert_called_once_with()


@pytest.mark.parametrize("adapter_kind", ["scheduler", "worker"])
def test_atom_constructor_preserves_error_when_client_close_fails(
    monkeypatch: pytest.MonkeyPatch,
    adapter_kind: str,
) -> None:
    """Cleanup errors cannot replace the constructor's original failure."""
    client = MagicMock(name="mq_client")
    client.close.side_effect = RuntimeError("close failed")
    monkeypatch.setattr(atom_adapter, "MessageQueueClient", lambda *args: client)
    monkeypatch.setattr(
        atom_adapter,
        "_get_chunk_size",
        MagicMock(side_effect=ValueError("original init failure")),
    )
    with pytest.raises(ValueError, match="original init failure"):
        _make_adapter(adapter_kind)


def test_atom_initial_registration_failure_rolls_back_candidate(
    worker_with_transfer_context: tuple[
        AtomMPWorkerAdapter, MagicMock, dict[str, torch.Tensor]
    ],
) -> None:
    """An ambiguous registration failure removes the GPU candidate."""
    worker, transfer_context, caches = worker_with_transfer_context
    client = _mock_client(worker)
    register_error = TimeoutError("server accepted but registration response timed out")
    transfer_context.register.side_effect = register_error
    rollback_future: MessagingFuture[None] = MessagingFuture()
    rollback_future.set_result(None)
    client.submit_request.return_value = rollback_future

    with pytest.raises(type(register_error)) as exc_info:
        worker.register_kv_caches(caches, engine_group_infos=_atom_groups())

    assert exc_info.value is register_error
    rollback_call = client.submit_request.call_args
    assert rollback_call.args[0] is RequestType.UNREGISTER_KV_CACHE
    assert rollback_call.args[1] == [worker.instance_id]
    transfer_context.close.assert_called_once_with()
    client.close.assert_called_once_with()
    assert worker.is_healthy is False


def test_atom_initial_registration_preserves_error_across_cleanup_failures(
    worker_with_transfer_context: tuple[
        AtomMPWorkerAdapter, MagicMock, dict[str, torch.Tensor]
    ],
) -> None:
    """Context/client close failures cannot replace a registration error."""
    worker, transfer_context, caches = worker_with_transfer_context
    client = _mock_client(worker)
    transfer_context.register.side_effect = ValueError("original register failure")
    client.submit_request.side_effect = RuntimeError("rollback failed")
    transfer_context.close.side_effect = RuntimeError("context close failed")
    client.close.side_effect = RuntimeError("client close failed")

    with pytest.raises(ValueError, match="original register failure"):
        worker.register_kv_caches(caches, engine_group_infos=_atom_groups())

    assert worker._shutdown_complete.is_set()


def test_atom_failed_recovery_rolls_back_candidate_and_keeps_old_context(
    worker_with_transfer_context: tuple[
        AtomMPWorkerAdapter, MagicMock, dict[str, torch.Tensor]
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed recovery cleans its candidate and can retry the old registration."""
    worker, old_context, caches = worker_with_transfer_context
    client = _mock_client(worker)
    worker.register_kv_caches(caches, engine_group_infos=_atom_groups())
    heartbeat = _FakeHeartbeatThread.instances[0]
    heartbeat.mark_unhealthy()

    failed_candidate = MagicMock(name="failed_candidate")
    failed_candidate.register.side_effect = TimeoutError(
        "server accepted but recovery response timed out"
    )
    retry_context = MagicMock(name="retry_context")
    candidates = iter([failed_candidate, retry_context])
    monkeypatch.setattr(
        atom_adapter,
        "create_transfer_context",
        lambda *args, **kwargs: next(candidates),
    )
    rollback_future: MessagingFuture[None] = MessagingFuture()
    rollback_future.set_result(None)
    client.submit_request.return_value = rollback_future

    assert heartbeat.recover() is False
    failed_candidate.close.assert_called_once_with()
    old_context.close.assert_not_called()
    assert worker._transfer_context is old_context
    assert worker._registered is True
    assert worker.is_healthy is False
    rollback_call = client.submit_request.call_args
    assert rollback_call.args[0] is RequestType.UNREGISTER_KV_CACHE

    assert heartbeat.recover() is True
    old_context.close.assert_called_once_with()
    assert worker._transfer_context is retry_context
    assert worker.is_healthy is True


@pytest.mark.parametrize(
    ("method_name", "submit_name"),
    [
        ("submit_store_request", "submit_store"),
        ("submit_retrieve_request", "submit_retrieve"),
    ],
)
def test_atom_submit_while_unhealthy_is_dropped_without_enqueue(
    worker_with_transfer_context: tuple[
        AtomMPWorkerAdapter, MagicMock, dict[str, torch.Tensor]
    ],
    method_name: str,
    submit_name: str,
) -> None:
    """An unhealthy generation cannot enqueue new transfer work."""
    worker, transfer_context, caches = worker_with_transfer_context
    worker.register_kv_caches(caches, engine_group_infos=_atom_groups())
    _FakeHeartbeatThread.instances[0].mark_unhealthy()

    returned = getattr(worker, method_name)(
        "request-1",
        AtomMPTransferSpec(
            token_ids=list(range(256)),
            block_ids=[[0, 1, 2, 3]],
            end=256,
        ),
        _FakeEvent(),
    )

    assert returned is None
    getattr(transfer_context, submit_name).assert_not_called()


def test_atom_submit_and_health_transition_are_atomic(
    worker_with_transfer_context: tuple[
        AtomMPWorkerAdapter, MagicMock, dict[str, torch.Tensor]
    ],
) -> None:
    """A failure transition cannot slip inside a healthy request enqueue."""
    worker, transfer_context, caches = worker_with_transfer_context
    worker.register_kv_caches(caches, engine_group_infos=_atom_groups())
    submit_entered = threading.Event()
    release_submit = threading.Event()
    failure_published = threading.Event()
    future: MessagingFuture[bool] = MessagingFuture()

    def delayed_submit(*args: Any, **kwargs: Any) -> MessagingFuture[bool]:
        del args, kwargs
        submit_entered.set()
        assert release_submit.wait(timeout=5.0)
        return future

    transfer_context.submit_store.side_effect = delayed_submit
    futures: list[MessagingFuture[bool] | None] = []
    submit_thread = threading.Thread(
        target=lambda: futures.append(
            worker.submit_store_request(
                "request-1",
                AtomMPTransferSpec(
                    token_ids=list(range(256)),
                    block_ids=[[0, 1, 2, 3]],
                    end=256,
                ),
                _FakeEvent(),
            )
        )
    )
    submit_thread.start()
    assert submit_entered.wait(timeout=5.0)

    def publish_failure() -> None:
        worker._mark_unhealthy()
        failure_published.set()

    failure_thread = threading.Thread(target=publish_failure)
    failure_thread.start()
    # The submit lease keeps the context alive without holding the state lock;
    # heartbeat failure must remain publishable while engine-driven submit blocks.
    assert failure_published.wait(timeout=5.0)
    release_submit.set()
    submit_thread.join(timeout=5.0)
    failure_thread.join(timeout=5.0)

    assert not submit_thread.is_alive()
    assert not failure_thread.is_alive()
    assert futures == [future]
    assert worker.is_healthy is False


@pytest.mark.parametrize(
    ("method_name", "submit_name"),
    [
        ("submit_store_request", "submit_store"),
        ("submit_retrieve_request", "submit_retrieve"),
    ],
)
def test_atom_recovery_waits_for_old_context_submission_lease(
    worker_with_transfer_context: tuple[
        AtomMPWorkerAdapter, MagicMock, dict[str, torch.Tensor]
    ],
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    submit_name: str,
) -> None:
    """Recovery cannot close an old context during its synchronous submit."""
    worker, old_context, caches = worker_with_transfer_context
    worker.register_kv_caches(caches, engine_group_infos=_atom_groups())
    heartbeat = _FakeHeartbeatThread.instances[0]
    recovery_context = MagicMock(name="recovery_context")
    monkeypatch.setattr(
        atom_adapter,
        "create_transfer_context",
        lambda *args, **kwargs: recovery_context,
    )

    submit_entered = threading.Event()
    release_submit = threading.Event()
    context_submit_returning = threading.Event()
    waiting_for_old_lease = threading.Event()
    old_context_closed = threading.Event()
    future: MessagingFuture[bool] = MessagingFuture()

    def delayed_submit(*args: Any, **kwargs: Any) -> MessagingFuture[bool]:
        del args, kwargs
        submit_entered.set()
        assert release_submit.wait(timeout=5.0)
        context_submit_returning.set()
        return future

    def close_old_context() -> None:
        assert context_submit_returning.is_set()
        old_context_closed.set()

    original_wait = worker._wait_for_context_leases

    def observed_wait(transfer_context: Any) -> None:
        assert transfer_context is old_context
        waiting_for_old_lease.set()
        original_wait(transfer_context)

    getattr(old_context, submit_name).side_effect = delayed_submit
    old_context.close.side_effect = close_old_context
    monkeypatch.setattr(worker, "_wait_for_context_leases", observed_wait)

    futures: list[MessagingFuture[bool] | None] = []
    submit_errors: list[BaseException] = []

    def submit() -> None:
        try:
            futures.append(
                getattr(worker, method_name)(
                    "request-1",
                    AtomMPTransferSpec(
                        token_ids=list(range(256)),
                        block_ids=[[0, 1, 2, 3]],
                        end=256,
                    ),
                    _FakeEvent(),
                )
            )
        except BaseException as error:
            submit_errors.append(error)

    submit_thread = threading.Thread(target=submit)
    submit_thread.start()
    assert submit_entered.wait(timeout=5.0)

    heartbeat.mark_unhealthy()
    recovery_results: list[bool] = []
    recovery_thread = threading.Thread(
        target=lambda: recovery_results.append(heartbeat.recover())
    )
    recovery_thread.start()

    assert waiting_for_old_lease.wait(timeout=5.0)
    assert worker._transfer_context is recovery_context
    assert worker.is_healthy is False
    assert old_context_closed.is_set() is False
    assert recovery_thread.is_alive()

    release_submit.set()
    submit_thread.join(timeout=5.0)
    recovery_thread.join(timeout=5.0)

    assert not submit_thread.is_alive()
    assert not recovery_thread.is_alive()
    assert submit_errors == []
    assert futures == [future]
    assert recovery_results == [True]
    assert old_context_closed.is_set()
    old_context.close.assert_called_once_with()
    assert worker.is_healthy is True


def test_atom_shutdown_discards_late_recovery_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A callback returning after shutdown cannot republish context or health."""
    client = MagicMock(name="mq_client")
    initial_context = MagicMock(name="initial_context")
    recovery_context = MagicMock(name="recovery_context")
    recovery_entered = threading.Event()
    release_recovery = threading.Event()

    def delayed_register(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        recovery_entered.set()
        assert release_recovery.wait(timeout=5.0)

    recovery_context.register.side_effect = delayed_register
    contexts = iter([initial_context, recovery_context])
    monkeypatch.setattr(atom_adapter, "MessageQueueClient", lambda *args: client)
    monkeypatch.setattr(atom_adapter, "_get_chunk_size", lambda *args: 256)
    monkeypatch.setattr(
        atom_adapter,
        "create_transfer_context",
        lambda *args, **kwargs: next(contexts),
    )
    monkeypatch.setattr(atom_adapter, "_HeartbeatThread", _FakeHeartbeatThread)
    _FakeHeartbeatThread.instances.clear()
    worker = AtomMPWorkerAdapter(
        server_url="tcp://127.0.0.1:5555",
        context=MagicMock(name="zmq_context"),
        model_name="atom-test-model",
        block_size=64,
        parallel_config=AtomMPParallelConfig(2, 1, 2),
    )
    caches = {"layer.0": torch.empty(4, 64, 576)}
    worker.register_kv_caches(caches, engine_group_infos=_atom_groups()[:1])
    heartbeat = _FakeHeartbeatThread.instances[0]
    heartbeat.mark_unhealthy()
    recovery_results: list[bool] = []
    recovery_thread = threading.Thread(
        target=lambda: recovery_results.append(heartbeat.recover())
    )
    recovery_thread.start()
    assert recovery_entered.wait(timeout=5.0)

    shutdown_returned = threading.Event()

    def shutdown() -> None:
        worker.shutdown()
        shutdown_returned.set()

    shutdown_thread = threading.Thread(target=shutdown)
    shutdown_thread.start()
    assert not shutdown_returned.wait(timeout=0.05)
    assert worker._transfer_context is initial_context
    assert worker.is_healthy is False
    client.close.assert_not_called()

    release_recovery.set()
    recovery_thread.join(timeout=5.0)
    shutdown_thread.join(timeout=5.0)
    assert not recovery_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert shutdown_returned.is_set()
    assert recovery_results == [False]
    recovery_context.close.assert_called_once_with()
    initial_context.close.assert_called_once_with()
    client.close.assert_called_once_with()
    assert worker._transfer_context is None
    assert worker._registered is False
    assert worker.is_healthy is False


def test_atom_heartbeat_publish_and_start_are_one_lifecycle_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """shutdown cannot detach a heartbeat in the publish-before-start window."""
    start_entered = threading.Event()
    release_start = threading.Event()

    class _BlockingStartHeartbeat(_FakeHeartbeatThread):
        def start(self) -> None:
            start_entered.set()
            assert release_start.wait(timeout=5.0)
            super().start()

    client = MagicMock(name="mq_client")
    transfer_context = MagicMock(name="transfer_context")
    monkeypatch.setattr(atom_adapter, "MessageQueueClient", lambda *args: client)
    monkeypatch.setattr(atom_adapter, "_get_chunk_size", lambda *args: 256)
    monkeypatch.setattr(
        atom_adapter,
        "create_transfer_context",
        lambda *args, **kwargs: transfer_context,
    )
    monkeypatch.setattr(atom_adapter, "_HeartbeatThread", _BlockingStartHeartbeat)
    _BlockingStartHeartbeat.instances.clear()
    worker = AtomMPWorkerAdapter(
        server_url="tcp://127.0.0.1:5555",
        context=MagicMock(name="zmq_context"),
        model_name="atom-test-model",
        block_size=64,
        parallel_config=AtomMPParallelConfig(2, 1, 2),
    )
    caches = {"layer.0": torch.empty(4, 64, 576)}
    register_errors: list[Exception] = []

    def register() -> None:
        try:
            worker.register_kv_caches(caches, engine_group_infos=_atom_groups()[:1])
        except Exception as error:
            register_errors.append(error)

    register_thread = threading.Thread(target=register)
    register_thread.start()
    assert start_entered.wait(timeout=5.0)
    shutdown_returned = threading.Event()

    def shutdown() -> None:
        worker.shutdown()
        shutdown_returned.set()

    shutdown_thread = threading.Thread(target=shutdown)
    shutdown_thread.start()
    assert not shutdown_returned.wait(timeout=0.05)

    release_start.set()
    register_thread.join(timeout=5.0)
    shutdown_thread.join(timeout=5.0)

    assert not register_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert register_errors == []
    heartbeat = _BlockingStartHeartbeat.instances[0]
    assert heartbeat.stop_requested is True
    assert heartbeat.started is False
    client.close.assert_called_once_with()


def test_atom_repeated_public_registration_fails_without_overwrite(
    worker_with_transfer_context: tuple[
        AtomMPWorkerAdapter, MagicMock, dict[str, torch.Tensor]
    ],
) -> None:
    """A second public register cannot overwrite the live local context."""
    worker, transfer_context, caches = worker_with_transfer_context
    worker.register_kv_caches(caches, engine_group_infos=_atom_groups())

    with pytest.raises(RuntimeError, match="already being or were registered"):
        worker.register_kv_caches(caches, engine_group_infos=_atom_groups())

    transfer_context.register.assert_called_once()
    assert worker._transfer_context is transfer_context


def test_atom_shutdown_unregisters_gpu_context(
    worker_with_transfer_context: tuple[
        AtomMPWorkerAdapter, MagicMock, dict[str, torch.Tensor]
    ],
) -> None:
    """ATOM removes its LMCache-driven GPU registration during shutdown."""
    worker, transfer_context, caches = worker_with_transfer_context
    client = _mock_client(worker)
    worker.register_kv_caches(caches, engine_group_infos=_atom_groups())

    worker.shutdown()

    request_types = [call.args[0] for call in client.submit_request.call_args_list]
    assert request_types == [RequestType.UNREGISTER_KV_CACHE]


@pytest.mark.parametrize(
    ("method_name", "submit_name"),
    [
        ("submit_store_request", "submit_store"),
        ("submit_retrieve_request", "submit_retrieve"),
    ],
)
def test_atom_shutdown_drains_unresolved_operation_before_unregister(
    worker_with_transfer_context: tuple[
        AtomMPWorkerAdapter, MagicMock, dict[str, torch.Tensor]
    ],
    method_name: str,
    submit_name: str,
) -> None:
    """Shutdown preserves registration and context until the future finishes."""
    worker, transfer_context, caches = worker_with_transfer_context
    client = _mock_client(worker)
    worker.register_kv_caches(caches, engine_group_infos=_atom_groups())
    wait_started = threading.Event()

    class _ObservedFuture(MessagingFuture[bool]):
        def wait(self, timeout: float | None = None) -> bool:
            wait_started.set()
            return super().wait(timeout)

    future = _ObservedFuture()
    getattr(transfer_context, submit_name).return_value = future
    returned = getattr(worker, method_name)(
        "request-1",
        AtomMPTransferSpec(
            token_ids=list(range(256)),
            block_ids=[[0, 1, 2, 3]],
            end=256,
        ),
        _FakeEvent(),
    )

    shutdown_thread = threading.Thread(target=worker.shutdown)
    shutdown_thread.start()
    assert wait_started.wait(timeout=5.0)
    transfer_context.close.assert_not_called()
    client.close.assert_not_called()
    client.submit_request.assert_not_called()

    future.set_result(True)
    shutdown_thread.join(timeout=5.0)

    assert not shutdown_thread.is_alive()
    assert returned is future
    assert returned.result(timeout=0) is True
    transfer_context.close.assert_called_once_with()
    client.close.assert_called_once_with()
    unregister_call = client.submit_request.call_args
    assert unregister_call.args[0] is RequestType.UNREGISTER_KV_CACHE


@pytest.mark.parametrize(
    ("method_name", "submit_name"),
    [
        ("submit_store_request", "submit_store"),
        ("submit_retrieve_request", "submit_retrieve"),
    ],
)
def test_atom_restart_window_transfer_and_shutdown_terminate(
    worker_with_transfer_context: tuple[
        AtomMPWorkerAdapter, MagicMock, dict[str, torch.Tensor]
    ],
    method_name: str,
    submit_name: str,
) -> None:
    """A missing-registration response cannot strand an ATOM operation."""
    worker, transfer_context, caches = worker_with_transfer_context
    client = _mock_client(worker)
    worker.register_kv_caches(caches, engine_group_infos=_atom_groups())
    raw_wait_started = threading.Event()

    class _ObservedRawFuture(MessagingFuture[tuple[bytes, bool]]):
        def wait(self, timeout: float | None = None) -> bool:
            raw_wait_started.set()
            return super().wait(timeout)

    # The server has restarted since the last heartbeat, but the worker still
    # considers its registration healthy. #4709 makes the replacement server
    # answer the admitted request with an empty event handle and False.
    raw_future = _ObservedRawFuture()
    device_future = raw_future.to_device_future(device="cpu")
    getattr(transfer_context, submit_name).return_value = device_future
    assert worker.is_healthy is True

    returned = getattr(worker, method_name)(
        "request-1",
        AtomMPTransferSpec(
            token_ids=list(range(256)),
            block_ids=[[0, 1, 2, 3]],
            end=256,
        ),
        _FakeEvent(),
    )
    assert returned is device_future

    shutdown_thread = threading.Thread(target=worker.shutdown)
    shutdown_thread.start()
    assert raw_wait_started.wait(timeout=5.0)
    transfer_context.close.assert_not_called()
    client.close.assert_not_called()
    client.submit_request.assert_not_called()

    raw_future.set_result((b"", False))
    shutdown_thread.join(timeout=5.0)

    assert not shutdown_thread.is_alive()
    assert returned.result(timeout=0) is False
    transfer_context.close.assert_called_once_with()
    client.close.assert_called_once_with()
    unregister_call = client.submit_request.call_args
    assert unregister_call.args[0] is RequestType.UNREGISTER_KV_CACHE


def test_atom_shutdown_finally_closes_client_when_context_close_fails(
    worker_with_transfer_context: tuple[
        AtomMPWorkerAdapter, MagicMock, dict[str, torch.Tensor]
    ],
) -> None:
    """Normal shutdown completes all cleanup stages despite close failures."""
    worker, transfer_context, caches = worker_with_transfer_context
    client = _mock_client(worker)
    worker.register_kv_caches(caches, engine_group_infos=_atom_groups())
    transfer_context.close.side_effect = RuntimeError("context close failed")
    client.close.side_effect = RuntimeError("client close failed")

    worker.shutdown()

    transfer_context.close.assert_called_once_with()
    client.close.assert_called_once_with()
    assert worker._shutdown_complete.is_set()


def test_atom_detector_recognizes_paged_3d_cache_views() -> None:
    """ATOM's per-layer ``[NB, BS, width]`` views use the MLA-like format."""
    cache_tensors = [
        torch.empty(4, 64, 576),
        torch.empty(4, 64, 144),
    ]
    caches: DiscoverableKVCache = cache_tensors

    detected, normalized = detect_format(caches, EngineType.ATOM, {})
    normalized_tensors = cast(list[torch.Tensor], normalized)

    assert EngineType.ATOM.value == "atom"
    assert detected is lmcache_native.EngineKVFormat.NL_X_NB_BS_HS
    assert [tuple(tensor.shape) for tensor in normalized_tensors] == [
        (4, 64, 576),
        (4, 64, 144),
    ]
    assert [tensor.data_ptr() for tensor in normalized_tensors] == [
        tensor.data_ptr() for tensor in cache_tensors
    ]
