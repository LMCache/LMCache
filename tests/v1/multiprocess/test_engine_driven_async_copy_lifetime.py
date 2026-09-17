# SPDX-License-Identifier: Apache-2.0
"""Async device<->CPU copies must complete before their buffers are reused.

Both engine-driven paths launch async copies and then hand the buffers to
something that reads them. Two places got this wrong, and both failed
silently: the KV was already committed or already scattered, so the only
symptom was corrupted content much later.

These tests assert the ordering contract without a GPU, so they run in the
CPU-only unit CI. The hardware reproductions live alongside them in
``test_engine_driven_transfer.py`` and skip without CUDA.
"""

# Standard
from typing import cast
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch


def test_scatter_syncs_before_releasing_dynamically_pinned_chunks() -> None:
    """Unpinned input must be synced before scatter returns.

    ``scatter_cpu_to_paged_kv`` pins unpinned chunks into temporaries and
    launches async H2D reads on them through raw pointers, which torch's
    stream tracking cannot see. Returning drops the last reference, so the
    caching host allocator can hand that memory to the next caller while the
    copies are in flight. The documented caller-side synchronize cannot cover
    it -- by then the temporaries are gone -- so scatter must sync itself.
    """
    # First Party
    from lmcache.v1.multiprocess.transfer_context import base

    kv = {f"layer_{i}": torch.zeros(2, 4, 4, 2, 8) for i in range(2)}

    # Mock chunks, not real tensors: pin_memory() needs an accelerator, so the
    # ptr-only branch cannot execute for real on the CPU-only unit CI. What we
    # are pinning down is the ordering contract, not the copy itself.
    def _unpinned_chunk() -> MagicMock:
        c = MagicMock()
        c.is_pinned.return_value = False
        c.pin_memory.return_value = c
        c.data_ptr.return_value = 0
        return c

    chunks = [_unpinned_chunk()]

    # An unannotated mock op models the ptr-only backend. Keep the capability
    # check real, and substitute only the platform resolver's selected ops.
    ops = MagicMock()
    with (
        patch.object(base, "synchronize_device") as synchronize,
        patch.object(base, "resolve_device_ops", return_value=ops),
    ):
        # cast: the mocks stand in for tensors on purpose (see above).
        base.scatter_cpu_to_paged_kv(
            kv, list(range(4)), cast(list[torch.Tensor], chunks), 4
        )
        assert ops.multi_layer_block_kv_transfer.called, (
            "fixture must reach the async H2D launches"
        )
        assert synchronize.call_args.args == (torch.device("cpu"),), (
            "scatter must complete async H2D before releasing the temporaries "
            "it pinned; otherwise the host allocator reuses them mid-copy"
        )


@pytest.mark.parametrize("operation", ["store", "retrieve"])
def test_transfer_syncs_kv_device_before_commit(operation: str) -> None:
    """Transfers finish on the KV device before serialization or SHM release."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import worker_transfer

    order: list[str] = []
    transport = MagicMock()
    transport.prepare_store.return_value = None  # pickle mode
    transport.prepare_retrieve.return_value = [torch.zeros(1)]

    def commit(*args: object, **kwargs: object) -> bool:
        order.append("commit")
        return True

    getattr(transport, f"commit_{operation}").side_effect = commit
    ctx = worker_transfer.EngineDrivenTransferContext(1, MagicMock())
    with patch.object(
        worker_transfer, "create_engine_driven_context", return_value=transport
    ):
        ctx.register(
            {"layer_0": torch.zeros(2, 4, 4, 2, 8)},
            "test",
            1,
            4,
            1.0,
            layout_hints={"kv_layout": "NHD"},
        )

    # Mock device copies so non-default CUDA-device ordering is testable on CPU CI.
    device = torch.device("cuda:1")
    kv = {"layer_0": MagicMock(spec=torch.Tensor, device=device)}
    transfer_name = (
        "gather_paged_kv_to_cpu" if operation == "store" else "scatter_cpu_to_paged_kv"
    )

    def transfer(*args: object, **kwargs: object) -> list[torch.Tensor]:
        order.append("copy")
        return [torch.zeros(1)]

    def synchronize(actual_device: torch.device) -> None:
        assert actual_device == device
        order.append("sync")

    try:
        with (
            patch.object(torch.cuda, "synchronize", side_effect=synchronize),
            patch.object(worker_transfer, transfer_name, side_effect=transfer),
        ):
            submit = getattr(ctx, f"submit_{operation}")
            assert submit("req", "key", kv, [[0, 1, 2, 3]], None, 4).result()
    finally:
        ctx.close()

    expected = ["copy", "sync", "commit"]
    if operation == "store":
        expected.insert(0, "sync")
    assert order == expected
