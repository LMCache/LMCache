# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MooncakeTeTransferChannelClient batch completion tracking."""

# Standard
from dataclasses import dataclass
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.common.transfer_channel.api import TransferChannelAddress
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.transfer_channel.impl.mooncake_te_impl import (
    INVALID_BATCH_ID,
    MooncakeTeTransferChannelClient,
)

_LOCAL_BASE = 0x1000_0000
_REMOTE_BASE = 0x2000_0000


@dataclass
class _SubmittedBatch:
    operation: str
    target_hostname: str
    local_ptrs: list[int]
    remote_ptrs: list[int]
    lengths: list[int]
    task_statuses: list[int]
    freed: bool = False


class _FakeLegacyMooncakeEngine:
    """Stand-in for Mooncake TE prior to PR #5674.

    Models the released Mooncake binding semantics:
    - No ``batch_transfer_check_status`` method.
    - ``transfer_check_status(batch_id)`` inspects ONLY task index 0 and frees
      the batch handle as soon as task 0 reaches a terminal status.
    - Re-polling a freed ``batch_id`` raises an AssertionError to catch
      use-after-free bugs.
    """

    def __init__(self, *, support_free_batch_id: bool = False) -> None:
        self._next_batch_id = 100
        self.batches: dict[int, _SubmittedBatch] = {}
        self.submit_order: list[int] = []
        self.check_calls: list[int] = []
        self.freed_batches: list[int] = []
        self.fail_on_submit_index: int | None = None
        self.fail_return_value: int = 0
        self._submit_count = 0
        if support_free_batch_id:
            self.free_batch_id = self._free_batch_id  # type: ignore[method-assign]

    def _submit(
        self,
        operation: str,
        target_hostname: str,
        local_ptrs: list[int],
        remote_ptrs: list[int],
        lengths: list[int],
    ) -> int:
        idx = self._submit_count
        self._submit_count += 1
        if self.fail_on_submit_index is not None and idx == self.fail_on_submit_index:
            return self.fail_return_value

        batch_id = self._next_batch_id
        self._next_batch_id += 1
        self.batches[batch_id] = _SubmittedBatch(
            operation=operation,
            target_hostname=target_hostname,
            local_ptrs=list(local_ptrs),
            remote_ptrs=list(remote_ptrs),
            lengths=list(lengths),
            task_statuses=[0] * len(local_ptrs),
        )
        self.submit_order.append(batch_id)
        return batch_id

    def batch_transfer_async_read(
        self,
        target_hostname: str,
        local_ptrs: list[int],
        remote_ptrs: list[int],
        lengths: list[int],
    ) -> int:
        return self._submit("READ", target_hostname, local_ptrs, remote_ptrs, lengths)

    def batch_transfer_async_write(
        self,
        target_hostname: str,
        local_ptrs: list[int],
        remote_ptrs: list[int],
        lengths: list[int],
    ) -> int:
        return self._submit("WRITE", target_hostname, local_ptrs, remote_ptrs, lengths)

    def transfer_check_status(self, batch_id: int) -> int:
        self.check_calls.append(batch_id)
        batch = self.batches[batch_id]
        assert not batch.freed, f"batch_id {batch_id} was polled after being freed"
        # Legacy Mooncake checks task index 0 only.
        status = batch.task_statuses[0]
        if status != 0:
            batch.freed = True
            self.freed_batches.append(batch_id)
        return status

    def _free_batch_id(self, batch_id: int) -> int:
        batch = self.batches[batch_id]
        assert not batch.freed, f"batch_id {batch_id} was double-freed"
        batch.freed = True
        self.freed_batches.append(batch_id)
        return 0

    def set_object_status(self, object_index: int, status: int) -> None:
        """Set status for the object at ``object_index`` across submitted batches."""
        seen = 0
        for batch_id in self.submit_order:
            batch = self.batches[batch_id]
            count = len(batch.task_statuses)
            if object_index < seen + count:
                batch.task_statuses[object_index - seen] = status
                return
            seen += count
        raise IndexError(f"object_index {object_index} out of range (total={seen})")


class _FakeModernMooncakeEngine(_FakeLegacyMooncakeEngine):
    """Stand-in for Mooncake TE with PR #5674 whole-batch status polling.

    Exposes ``batch_transfer_check_status(batch_ids) -> list[int]`` which
    checks all tasks in each batch (non-blocking) and frees a batch handle once
    all tasks reach a terminal state.
    """

    def __init__(self) -> None:
        super().__init__(support_free_batch_id=True)
        self.batch_check_calls: list[list[int]] = []
        self.override_batch_check_result: list[int] | None = None

    def _evaluate_whole_batch(self, batch_id: int) -> int:
        batch = self.batches[batch_id]
        assert not batch.freed, f"batch_id {batch_id} was polled after being freed"
        statuses = batch.task_statuses
        for status in statuses:
            if status not in (0, 1, -1, -2):
                batch.freed = True
                self.freed_batches.append(batch_id)
                return status
        if any(status == 0 for status in statuses):
            return 0
        batch.freed = True
        self.freed_batches.append(batch_id)
        if all(status == 1 for status in statuses):
            return 1
        if any(status == -2 for status in statuses):
            return -2
        return -1

    def batch_transfer_check_status(self, batch_ids: list[int]) -> list[int]:
        self.batch_check_calls.append(list(batch_ids))
        if self.override_batch_check_result is not None:
            return list(self.override_batch_check_result)
        return [self._evaluate_whole_batch(bid) for bid in batch_ids]

    def transfer_check_status(self, batch_id: int) -> int:
        self.check_calls.append(batch_id)
        return self._evaluate_whole_batch(batch_id)


class _StubContext:
    """Minimal context exposing the public properties used by the client."""

    def __init__(self, engine: Any) -> None:
        self._l1_memory_desc = L1MemoryDesc(
            ptr=_LOCAL_BASE, size=65536, align_bytes=256
        )
        self._engine = engine

    @property
    def l1_memory_desc(self) -> L1MemoryDesc:
        return self._l1_memory_desc

    @property
    def mooncake_te_engine(self) -> Any:
        return self._engine


def _make_client(engine: Any) -> MooncakeTeTransferChannelClient:
    return MooncakeTeTransferChannelClient(
        context=_StubContext(engine),  # type: ignore[arg-type]
        remote_session_id="127.0.0.1:18591",
        remote_buffer_ptr=_REMOTE_BASE,
    )


def _sample_addresses(
    count: int,
) -> tuple[list[TransferChannelAddress], list[TransferChannelAddress]]:
    local = [TransferChannelAddress(offset=i * 256, size=256) for i in range(count)]
    remote = [
        TransferChannelAddress(offset=(i + 1) * 512, size=256) for i in range(count)
    ]
    return local, remote


# =========================================================
# Legacy Mooncake TE (without batch_transfer_check_status)
# =========================================================
@pytest.mark.parametrize("operation", ["READ", "WRITE"])
def test_legacy_engine_partial_completion_waits_for_all_buffers(operation: str) -> None:
    """Task 0 finishing must not report the transfer done while other buffers run."""
    engine = _FakeLegacyMooncakeEngine()
    client = _make_client(engine)
    local, remote = _sample_addresses(3)

    task_id = (
        client.submit_read(local, remote)
        if operation == "READ"
        else client.submit_write(local, remote)
    )

    # Legacy engine must split multi-buffer transfers so every buffer is tracked.
    assert len(engine.submit_order) == 3
    for idx, batch_id in enumerate(engine.submit_order):
        batch = engine.batches[batch_id]
        assert batch.operation == operation
        assert batch.local_ptrs == [_LOCAL_BASE + local[idx].offset]
        assert batch.remote_ptrs == [_REMOTE_BASE + remote[idx].offset]
        assert batch.lengths == [local[idx].size]

    # Buffer 0 finishes while buffers 1 and 2 are still in flight.
    engine.set_object_status(0, 1)
    result = client.query_status(task_id)
    assert result.is_finished() is False
    assert result.succeeded_mask == []

    # Buffer 0's batch handle was freed on terminal status and must not be re-polled.
    batch0, batch1, batch2 = engine.submit_order
    assert engine.check_calls == [batch0, batch1, batch2]
    engine.check_calls.clear()

    # Buffer 1 finishes; buffer 2 is still pending.
    engine.set_object_status(1, 1)
    result = client.query_status(task_id)
    assert result.is_finished() is False
    assert engine.check_calls == [batch1, batch2]
    engine.check_calls.clear()

    # Buffer 2 finishes -> whole transfer completes.
    engine.set_object_status(2, 1)
    result = client.query_status(task_id)
    assert result.is_finished() is True
    assert result.succeeded_mask == [True, True, True]
    assert engine.check_calls == [batch2]

    # Task is reported finished at most once.
    with pytest.raises(KeyError):
        client.query_status(task_id)


@pytest.mark.parametrize("operation", ["READ", "WRITE"])
def test_legacy_engine_partial_failure_and_timeout_reported_per_buffer(
    operation: str,
) -> None:
    """A failing/timed-out later buffer must not be masked by buffer 0 succeeding."""
    engine = _FakeLegacyMooncakeEngine()
    client = _make_client(engine)
    local, remote = _sample_addresses(3)

    task_id = (
        client.submit_read(local, remote)
        if operation == "READ"
        else client.submit_write(local, remote)
    )

    # Buffer 0 succeeds, buffer 1 fails, buffer 2 is still pending -> must wait for 2.
    engine.set_object_status(0, 1)
    engine.set_object_status(1, -1)
    result = client.query_status(task_id)
    assert result.is_finished() is False

    # Buffer 2 times out -> now all buffers have drained; report exact per-buffer mask.
    engine.set_object_status(2, -2)
    result = client.query_status(task_id)
    assert result.is_finished() is True
    assert result.succeeded_mask == [True, False, False]

    with pytest.raises(KeyError):
        client.query_status(task_id)


def test_legacy_engine_single_buffer_submits_one_batch() -> None:
    engine = _FakeLegacyMooncakeEngine()
    client = _make_client(engine)
    local, remote = _sample_addresses(1)

    task_id = client.submit_read(local, remote)
    assert len(engine.submit_order) == 1

    assert client.query_status(task_id).is_finished() is False
    engine.set_object_status(0, 1)
    result = client.query_status(task_id)
    assert result.is_finished() is True
    assert result.succeeded_mask == [True]


@pytest.mark.parametrize(
    "support_free_batch_id", [False, True], ids=["no_free_batch_id", "free_batch_id"]
)
@pytest.mark.parametrize("operation", ["READ", "WRITE"])
def test_legacy_engine_partial_submit_failure_waits_for_submitted_buffers(
    operation: str, support_free_batch_id: bool
) -> None:
    """Buffers already submitted when a later submit fails keep being tracked.

    They cannot be cancelled, so the task must stay unfinished until they drain
    instead of raising while they still write into the caller's buffers.
    """
    engine = _FakeLegacyMooncakeEngine(support_free_batch_id=support_free_batch_id)
    engine.fail_on_submit_index = 2
    client = _make_client(engine)
    local, remote = _sample_addresses(3)

    task_id = (
        client.submit_read(local, remote)
        if operation == "READ"
        else client.submit_write(local, remote)
    )

    batch0, batch1 = engine.submit_order
    # In-flight batch handles must not be released behind the engine's back.
    assert engine.freed_batches == []

    assert client.query_status(task_id).is_finished() is False
    engine.set_object_status(0, 1)
    assert client.query_status(task_id).is_finished() is False

    engine.set_object_status(1, 1)
    result = client.query_status(task_id)
    assert result.is_finished() is True
    assert result.succeeded_mask == [True, True, False]
    # Both handles were reclaimed through the normal polling path.
    assert engine.freed_batches == [batch0, batch1]

    with pytest.raises(KeyError):
        client.query_status(task_id)


@pytest.mark.parametrize(
    "fail_return_value", [0, INVALID_BATCH_ID], ids=["zero", "invalid"]
)
def test_legacy_engine_first_submit_failure_raises(fail_return_value: int) -> None:
    """With nothing in flight, a submit failure still raises and registers no task."""
    engine = _FakeLegacyMooncakeEngine(support_free_batch_id=True)
    engine.fail_on_submit_index = 0
    engine.fail_return_value = fail_return_value
    client = _make_client(engine)
    local, remote = _sample_addresses(3)

    with pytest.raises(RuntimeError, match="Failed to submit async read"):
        client.submit_read(local, remote)

    assert engine.submit_order == []
    with pytest.raises(KeyError):
        client.query_status(0)


# =========================================================
# Modern Mooncake TE (with batch_transfer_check_status)
# =========================================================
@pytest.mark.parametrize("operation", ["READ", "WRITE"])
def test_modern_engine_submits_single_batch_and_waits_for_all_tasks(
    operation: str,
) -> None:
    """Modern engines keep single-batch submission and poll whole-batch status."""
    engine = _FakeModernMooncakeEngine()
    client = _make_client(engine)
    local, remote = _sample_addresses(3)

    task_id = (
        client.submit_read(local, remote)
        if operation == "READ"
        else client.submit_write(local, remote)
    )

    # Submitted as a single multi-buffer batch.
    assert len(engine.submit_order) == 1
    batch_id = engine.submit_order[0]
    assert len(engine.batches[batch_id].local_ptrs) == 3

    # Task 0 finishes while tasks 1 and 2 are pending -> still in flight.
    engine.set_object_status(0, 1)
    result = client.query_status(task_id)
    assert result.is_finished() is False
    assert engine.batch_check_calls == [[batch_id]]
    assert engine.check_calls == []

    # All tasks finish -> whole batch completes.
    engine.set_object_status(1, 1)
    engine.set_object_status(2, 1)
    result = client.query_status(task_id)
    assert result.is_finished() is True
    assert result.succeeded_mask == [True, True, True]
    assert engine.batch_check_calls == [[batch_id], [batch_id]]

    with pytest.raises(KeyError):
        client.query_status(task_id)


@pytest.mark.parametrize("error_code", [-1, -2], ids=["failed", "timeout"])
def test_modern_engine_partial_failure_surfaces_as_unsuccessful(
    error_code: int,
) -> None:
    """When task 0 succeeds but task 1 fails/times out, the batch is not successful."""
    engine = _FakeModernMooncakeEngine()
    client = _make_client(engine)
    local, remote = _sample_addresses(3)

    task_id = client.submit_read(local, remote)
    engine.set_object_status(0, 1)
    engine.set_object_status(1, error_code)
    engine.set_object_status(2, 1)

    result = client.query_status(task_id)
    assert result.is_finished() is True
    assert result.succeeded_mask == [False, False, False]

    with pytest.raises(KeyError):
        client.query_status(task_id)


def test_modern_engine_unexpected_status_raises_and_removes_task() -> None:
    engine = _FakeModernMooncakeEngine()
    client = _make_client(engine)
    local, remote = _sample_addresses(2)

    task_id = client.submit_read(local, remote)
    engine.set_object_status(0, 42)

    with pytest.raises(
        RuntimeError, match="unexpected mooncake transfer engine status"
    ):
        client.query_status(task_id)

    with pytest.raises(KeyError):
        client.query_status(task_id)


def test_modern_engine_mismatched_status_list_length_raises() -> None:
    engine = _FakeModernMooncakeEngine()
    client = _make_client(engine)
    local, remote = _sample_addresses(2)

    task_id = client.submit_read(local, remote)
    engine.override_batch_check_result = [1, 1]

    with pytest.raises(RuntimeError, match="expected 1"):
        client.query_status(task_id)

    with pytest.raises(KeyError):
        client.query_status(task_id)


# =========================================================
# Validation & lifecycle edge cases
# =========================================================
def test_empty_transfer_completes_immediately_without_engine_submit() -> None:
    engine = _FakeLegacyMooncakeEngine()
    client = _make_client(engine)

    task_id = client.submit_read([], [])
    assert engine.submit_order == []

    result = client.query_status(task_id)
    assert result.is_finished() is True
    assert result.succeeded_mask == []

    with pytest.raises(KeyError):
        client.query_status(task_id)


def test_validation_errors_and_close_invalidates_pending_tasks() -> None:
    engine = _FakeModernMooncakeEngine()
    client = _make_client(engine)
    local, remote = _sample_addresses(2)

    with pytest.raises(ValueError, match="only reads into the L1 buffer"):
        client.submit_read(local, remote, local_region=1)

    with pytest.raises(ValueError, match="equal length"):
        client.submit_read(local[:1], remote)

    with pytest.raises(ValueError, match="sizes must match"):
        client.submit_read(
            [TransferChannelAddress(offset=0, size=256)],
            [TransferChannelAddress(offset=0, size=512)],
        )

    task_id = client.submit_read(local, remote)
    batch_id = engine.submit_order[0]
    client.close()
    assert engine.freed_batches == [batch_id]

    with pytest.raises(KeyError):
        client.query_status(task_id)
