# SPDX-License-Identifier: Apache-2.0
"""Async device<->CPU copies must complete before their buffers are reused.

Both engine-driven paths launch async copies and then hand the buffers to
something that reads them. Two places got this wrong, and both failed
silently: the KV was already committed or already scattered, so the only
symptom was corrupted content much later.

These tests assert the ordering contract without a GPU, so they run in the
CPU-only unit CI. The hardware reproductions live alongside them in
``test_engine_driven_transfer.py`` and skip without CUDA. The scatter case is
owned by the native block-transfer backend and is covered in
``tests/v1/platform/test_native_kernel_device_ops.py``.
"""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
import torch


def test_pickle_store_syncs_before_commit_serializes() -> None:
    """The pickle path must sync before commit_store reads the buffers.

    Gather issues async device->CPU copies into fresh buffers, and the pickle
    transport serializes them immediately in ``commit_store``. Syncing only
    when ``out_buffers`` is given (the SHM path) leaves pickle serializing a
    buffer that is still being written.
    """
    # First Party
    from lmcache.v1.multiprocess.transfer_context import worker_transfer

    order: list[str] = []
    ctx = worker_transfer.EngineDrivenTransferContext(1, MagicMock())
    transport = MagicMock()
    transport.prepare_store.return_value = None  # pickle mode

    def _commit(*_a: object, **_k: object) -> bool:
        order.append("commit")
        return True

    transport.commit_store.side_effect = _commit
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

    def _gather(*_a: object, **_k: object) -> list[torch.Tensor]:
        order.append("gather")
        return [torch.zeros(1)]

    device = torch.device("cuda:1")
    with (
        patch.object(worker_transfer, "synchronize_device") as synchronize,
        patch.object(worker_transfer, "gather_paged_kv_to_cpu", side_effect=_gather),
    ):
        synchronize.side_effect = lambda *a, **k: order.append("sync")
        ctx.submit_store(
            "req",
            MagicMock(),  # key
            {"layer_0": MagicMock(spec=torch.Tensor, device=device)},
            [[0, 1, 2, 3]],
            MagicMock(),  # event (unused on this transport)
            4,  # blocks_in_chunk
        )

        synchronize.assert_called_with(device)
    ctx.close()

    # A sync must fall BETWEEN gather and commit. submit_store also syncs
    # before prepare_store, so merely finding a "sync" proves nothing -- that
    # earlier one is why guarding this on out_buffers went unnoticed.
    gathered, committed = order.index("gather"), order.index("commit")
    assert any(
        i for i, step in enumerate(order) if step == "sync" and gathered < i < committed
    ), (
        "pickle store must synchronize after gather and before commit_store "
        f"serializes the buffers, got {order}"
    )
