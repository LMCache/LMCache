# SPDX-License-Identifier: Apache-2.0
"""Receiver verification with real CPU tensors and an in-memory transport."""

# Standard
from collections.abc import Callable
from unittest.mock import MagicMock

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache.cli.commands.bench.server_bench import helpers
from lmcache.cli.commands.bench.server_bench.cases.base import BenchResult
from lmcache.cli.commands.bench.server_bench.cases.baseline import BaselineBenchCase
from lmcache.cli.commands.bench.server_bench.client import ServerBenchClient
from lmcache.cli.commands.bench.server_bench.config import BenchConfig
from lmcache.v1.multiprocess import transfer_context
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory


@pytest.fixture
def run_bench(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., tuple[BenchResult, list[str], MagicMock]]:
    """Run the public baseline case while controlling transfer results."""

    def run(
        *,
        mla: bool = True,
        tp_size: int = 2,
        mode: str = "engine_driven",
        fault: str = "none",
    ) -> tuple[BenchResult, list[str], MagicMock]:
        allocated: list[list[torch.Tensor]] = []
        stored: dict[int, dict[str, torch.Tensor]] = {}
        block_dim = 0 if mla else 1
        shape = (8, 2, 4) if mla else (2, 8, 2, 1, 4)
        ready: MessagingFuture[bool] = MessagingFuture()
        ready.set_result(True)

        def allocate(**_kwargs: object) -> list[torch.Tensor]:
            rank = len(allocated)
            tensors = [
                torch.full(shape, 10 * rank + layer + 1, dtype=torch.float16)
                for layer in range(2)
            ]
            allocated.append(tensors)
            return tensors

        class MemoryTransferContext:
            def __init__(self, instance_id: int, *_args: object) -> None:
                self.rank = instance_id - 1000
                self.register = MagicMock()
                self.unregister = MagicMock(return_value=None)
                self.create_recorded_event = MagicMock(return_value=None)
                self.flush_inflight_stores = MagicMock()
                self.close = MagicMock()

            def submit_store(
                self,
                request_id: str,
                key: IPCCacheServerKey,
                kv_caches: dict[str, torch.Tensor],
                block_ids: list[list[int]],
                event: object,
                blocks_in_chunk: int,
            ) -> MessagingFuture[bool]:
                assert key.worker_id is not None
                stored[key.worker_id] = {
                    name: tensor.index_select(block_dim, torch.tensor(block_ids[0]))
                    for name, tensor in kv_caches.items()
                }
                return ready

            def submit_retrieve(
                self,
                request_id: str,
                key: IPCCacheServerKey,
                kv_caches: dict[str, torch.Tensor],
                block_ids: list[list[int]],
                event: object,
                blocks_in_chunk: int,
                *,
                skip_first_n_tokens: int,
            ) -> MessagingFuture[bool]:
                assert key.worker_id is not None
                if fault == "no_write" and self.rank == tp_size - 1:
                    return ready
                for name, tensor in kv_caches.items():
                    tensor.index_copy_(
                        block_dim,
                        torch.tensor(block_ids[0]),
                        stored[key.worker_id][name],
                    )
                if fault == "corrupt" and self.rank == tp_size - 1:
                    next(iter(kv_caches.values())).select(
                        block_dim, block_ids[0][-1]
                    ).fill_(-1)
                return ready

        def query_checksum(
            http_url: str,
            block_offset: int,
            num_blocks: int,
            block_size: int,
            chunk_size: int,
            instance_id: int,
        ) -> list[str] | None:
            if fault == "unavailable" and instance_id == 1000 + tp_size - 1:
                return None
            return helpers._compute_client_checksums(
                allocated[instance_id - 1000],
                block_offset,
                num_blocks,
                block_size,
                chunk_size,
            )

        checksum_query = MagicMock(side_effect=query_checksum)
        monkeypatch.setattr(zmq, "Context", MagicMock())
        monkeypatch.setattr(RequestClientFactory, "create", MagicMock())
        monkeypatch.setattr(helpers, "_get_chunk_size", lambda _client: 2)
        monkeypatch.setattr(helpers, "_allocate_kv_cache", allocate)
        monkeypatch.setattr(helpers, "_send_lookup", MagicMock(return_value=True))
        monkeypatch.setattr(
            helpers, "_poll_prefetch_status", MagicMock(side_effect=[0, 2])
        )
        monkeypatch.setattr(helpers, "_send_end_session", MagicMock())
        monkeypatch.setattr(helpers, "_query_checksum", checksum_query)
        monkeypatch.setattr(
            transfer_context, "EngineDrivenTransferContext", MemoryTransferContext
        )
        monkeypatch.setattr(
            transfer_context,
            "create_transfer_context",
            lambda _caches, *, instance_id, **_kwargs: MemoryTransferContext(
                instance_id
            ),
        )
        messages: list[str] = []
        client = ServerBenchClient(
            BenchConfig(
                rpc_url="ipc:///tmp/test-receiver-checksums",
                http_url="http://localhost:8000",
                mode="cpu",
                transfer_mode=mode,
                tp_size=tp_size,
                use_mla=mla,
                num_tokens=3,
                kvcache_shape_spec=f"({1 if mla else 2},8,2,1,4):float16:2",
                num_blocks=8,
                block_size=2,
            ),
            messages.append,
        )
        try:
            client.start()
            result = BaselineBenchCase(
                sequence_count=1, interval_seconds=0, sequence_id_offset=1
            ).run(client, messages.append)
        finally:
            client.close()
        return result, messages, checksum_query

    return run


@pytest.mark.parametrize("mode", ["auto", "engine_driven", "lmcache_driven"])
@pytest.mark.parametrize(
    ("mla", "tp_size"), [(True, 1), (True, 2), (True, 4), (False, 2)]
)
def test_all_receivers_match_their_writer(
    run_bench: Callable[..., tuple[BenchResult, list[str], MagicMock]],
    mode: str,
    mla: bool,
    tp_size: int,
) -> None:
    """MLA receivers share rank 0's reference; ordinary TP keeps distinct shards."""
    result, _, query = run_bench(mla=mla, tp_size=tp_size, mode=mode)

    assert result.succeeded
    assert result.checks["checksum_match"] == [True]
    if mode == "lmcache_driven":
        writers = [1000] if mla else list(range(1000, 1000 + tp_size))
        receivers = list(range(1000, 1000 + tp_size))
        assert [c.kwargs["instance_id"] for c in query.call_args_list] == (
            writers + receivers
        )
    else:
        query.assert_not_called()


@pytest.mark.parametrize("mode", ["auto", "engine_driven", "lmcache_driven"])
@pytest.mark.parametrize(
    ("mla", "fault"), [(True, "corrupt"), (True, "no_write"), (False, "corrupt")]
)
def test_secondary_receiver_failure_fails_verification(
    run_bench: Callable[..., tuple[BenchResult, list[str], MagicMock]],
    mode: str,
    mla: bool,
    fault: str,
) -> None:
    """A successful transfer reply must not hide bad data on rank 1."""
    result, messages, _ = run_bench(mla=mla, mode=mode, fault=fault)

    assert result.checks["warm_retrieve_succeeded"] == [True]
    assert result.checks["checksum_match"] == [False]
    assert not result.succeeded
    assert any("rank 1" in message and "FAIL" in message for message in messages)


def test_missing_receiver_checksum_fails_verification(
    run_bench: Callable[..., tuple[BenchResult, list[str], MagicMock]],
) -> None:
    """An unavailable HTTP checksum on rank 1 cannot pass verification."""
    result, _, _ = run_bench(mode="lmcache_driven", fault="unavailable")

    assert result.checks["checksum_available"] == [False]
    assert "checksum_match" not in result.checks
    assert not result.succeeded
