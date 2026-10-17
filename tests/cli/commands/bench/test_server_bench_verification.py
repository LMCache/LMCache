# SPDX-License-Identifier: Apache-2.0
"""Verify server-bench payload checks follow the created transfer context."""

# Standard
from collections.abc import Generator
from unittest.mock import MagicMock
import hashlib

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache import torch_dev
from lmcache.cli.commands.bench.server_bench import helpers
from lmcache.cli.commands.bench.server_bench.client import ServerBenchClient
from lmcache.cli.commands.bench.server_bench.config import BenchConfig
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo
from lmcache.v1.multiprocess import transfer_context
from lmcache.v1.multiprocess.transfer_context import (
    AsyncEngineDrivenTransferContext,
    EngineDrivenTransferContext,
    LMCacheDrivenTransferContext,
)
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory


@pytest.fixture(
    params=[
        ("gpu", "auto", EngineDrivenTransferContext, "cpu"),
        ("gpu", "auto", AsyncEngineDrivenTransferContext, "cpu"),
        ("gpu", "auto", LMCacheDrivenTransferContext, "cpu"),
        ("gpu", "engine_driven", EngineDrivenTransferContext, "cpu"),
        ("gpu", "engine_driven", AsyncEngineDrivenTransferContext, "cpu"),
        ("gpu", "lmcache_driven", LMCacheDrivenTransferContext, "cpu"),
        ("cpu", "auto", EngineDrivenTransferContext, "cpu"),
        ("cpu", "engine_driven", EngineDrivenTransferContext, "cpu"),
        ("cpu", "lmcache_driven", LMCacheDrivenTransferContext, "cpu"),
        pytest.param(
            ("gpu", "auto", EngineDrivenTransferContext, "cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is not available"
            ),
        ),
        pytest.param(
            ("gpu", "auto", EngineDrivenTransferContext, "mps"),
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS is not available"
            ),
        ),
        pytest.param(
            ("gpu", "engine_driven", EngineDrivenTransferContext, "mps"),
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS is not available"
            ),
        ),
    ],
    ids=[
        "gpu-auto-engine",
        "gpu-auto-async-engine",
        "gpu-auto-handles",
        "gpu-engine",
        "gpu-async-engine",
        "gpu-handles",
        "cpu-auto",
        "cpu-engine",
        "cpu-handles",
        "cuda-tensor-auto-engine",
        "mps-tensor-auto-engine",
        "mps-tensor-engine",
    ],
)
def verification_client(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[
    tuple[ServerBenchClient, list[torch.Tensor], MagicMock, bool], None, None
]:
    """Start a two-rank client with real tensors and mocked transfer I/O."""
    mode, transfer_mode, context_type, tensor_device = request.param
    tensors: list[torch.Tensor] = []

    def allocate(
        *, groups: list[KVLayerGroupInfo], device: str | None
    ) -> list[torch.Tensor]:
        tensor = torch.ones(2, 8, 2, 1, 4, dtype=torch.float16, device=tensor_device)
        tensors.append(tensor)
        return [tensor]

    def create_context(*args: object, **kwargs: object) -> MagicMock:
        return MagicMock(spec=context_type)

    query_checksum = MagicMock(return_value=["server-checksum"])
    monkeypatch.setattr(torch_dev, "is_available", lambda: True)
    monkeypatch.setattr(zmq, "Context", MagicMock)
    monkeypatch.setattr(RequestClientFactory, "create", MagicMock())
    monkeypatch.setattr(helpers, "_get_chunk_size", lambda client: 2)
    monkeypatch.setattr(helpers, "_allocate_kv_cache", allocate)
    monkeypatch.setattr(helpers, "_query_checksum", query_checksum)
    monkeypatch.setattr(transfer_context, "create_transfer_context", create_context)
    monkeypatch.setattr(EngineDrivenTransferContext, "register", MagicMock())
    bench = ServerBenchClient(
        BenchConfig(
            rpc_url="ipc:///tmp/test-bench-verification",
            http_url="http://localhost:8080",
            mode=mode,
            transfer_mode=transfer_mode,
            tp_size=2,
            use_mla=False,
            num_tokens=3,
            kvcache_shape_spec="(2,8,2,1,4):float16:1",
            num_blocks=8,
            block_size=2,
        ),
        lambda message: None,
    )
    try:
        bench.start()
        yield (
            bench,
            tensors,
            query_checksum,
            context_type is LMCacheDrivenTransferContext,
        )
    finally:
        bench.close()


def test_zero_destination_follows_transfer_context(
    verification_client: tuple[ServerBenchClient, list[torch.Tensor], MagicMock, bool],
) -> None:
    """Clear only the requested blocks for engine-driven contexts."""
    bench, tensors, _, uses_handles = verification_client
    request = bench.create_request(1, "verification-zero", "warm")
    assert request is not None

    bench.zero_destination(request, start_token=2, token_count=2)

    for tensor in tensors:
        expected = torch.ones_like(tensor)
        if not uses_handles:
            expected[:, 3:4].zero_()
        torch.testing.assert_close(tensor, expected)


def test_checksums_follow_transfer_context(
    verification_client: tuple[ServerBenchClient, list[torch.Tensor], MagicMock, bool],
) -> None:
    """Hash local bytes for engine-driven contexts and query HTTP for handles."""
    bench, tensors, query_checksum, uses_handles = verification_client
    request = bench.create_request(1, "verification-checksum", "cold")
    assert request is not None

    checksums = bench.compute_checksums(request, start_token=2, token_count=2)

    if uses_handles:
        assert checksums == ["server-checksum"]
        query_checksum.assert_called_once_with(
            "http://localhost:8080", 3, 1, 2, 2, instance_id=1000
        )
    else:
        expected = [
            hashlib.md5(tensor[:, 3:4].contiguous().cpu().numpy().tobytes()).hexdigest()
            for tensor in tensors
        ]
        assert checksums == expected
        query_checksum.assert_not_called()
