# SPDX-License-Identifier: Apache-2.0
"""Retrieve steps, and the two ways an engine runs one.

A retrieve moves a group's chunks in steps: one kernel batch of full
chunks, or the unfull chunk. Each is a :class:`TransferStep`: the objects,
where they land in a staging, and the kernels that copy the landed bytes
into the paged KV cache.

- An **on-stream** engine queues its copy on the context's stream, so the
  step is staged and its kernels are queued right away: the stream orders
  the kernels after the copy.
- An **off-stream** engine's bytes arrive by a path the stream does not wait
  for (a NIC writing into GPU memory), so the kernels cannot be queued yet.
  The step is handed to the caller's :func:`collect_off_stream_steps` block,
  and the caller gives the collected steps to an :class:`OffStreamEngine`,
  which stages them and queues their kernels once the bytes landed.

``engines.run_step`` picks between the two.
"""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Callable, Protocol, Sequence

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform.base.cache_context import BaseCacheContext


class Staging(Protocol):
    """Where one transfer step is staged, indexed by its chunks: chunk ``i``
    of the step is in slot ``i``. A cache context is one (its shared
    buffer); an off-stream engine passes a view of its own buffer."""

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        """Flat uint8 region of one object group of chunk ``batch_idx``."""
        ...

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """One kernel group of chunk ``batch_idx``, shaped for the kernel."""
        ...

    def get_temp_staging_buffer(self, num_bytes: int) -> torch.Tensor | None:
        """The first ``num_bytes`` bytes from chunk 0 on, as one flat region;
        None if they do not fit."""
        ...


@dataclass
class TransferStep:
    """One retrieve step: objects to stage and the kernels that follow.

    Attributes:
        memory_objs: The objects to stage, in chunk order; all of one engine.
        block_ids: The block ids the kernels use.
        targets: ``staging -> regions``: where each object lands, one region
            per object.
        launch: ``(staging, block_ids) -> None``: queues the kernels that
            copy the landed bytes into the paged KV cache.
        skipped: Chunks of the step before the first object; the kernels
            skip them, so an off-stream engine gives them no slot.
        spans_slots: The object may extend past its slot into the following
            ones (an unfull chunk with lead blocks), so an off-stream engine
            must stage it where the slots after it are free.
    """

    memory_objs: Sequence[MemoryObj]
    block_ids: list[torch.Tensor]
    targets: Callable[[Staging], list[torch.Tensor]]
    launch: Callable[[Staging, list[torch.Tensor]], None]
    skipped: int = 0
    spans_slots: bool = False


_step_sink: ContextVar[list[TransferStep] | None] = ContextVar(
    "lmcache_off_stream_step_sink", default=None
)


# The block ids last handed off in the current collect block and the copy made
# of them. Holding the staged list keeps its identity from being reused.
_block_id_copy: ContextVar[tuple[list[torch.Tensor], list[torch.Tensor]] | None] = (
    ContextVar("lmcache_off_stream_block_id_copy", default=None)
)


@contextmanager
def collect_off_stream_steps(steps: list[TransferStep]) -> Iterator[None]:
    """Append to ``steps`` the off-stream steps of the transfers called
    inside the block, on this thread.

    The steps reach the caller this way and not as a return value because a
    transfer runs as the cache context's ``transfer_object_group`` callback,
    which returns nothing on every platform.

    Args:
        steps: Receives the steps, in transfer order.
    """
    token = _step_sink.set(steps)
    copy_token = _block_id_copy.set(None)
    try:
        yield
    finally:
        _block_id_copy.reset(copy_token)
        _step_sink.reset(token)


def hand_off_step(step: TransferStep) -> None:
    """Hand one step to the caller's :func:`collect_off_stream_steps` block,
    with a copy of the block ids: the next transfer restages the buffer they
    live in.

    Consecutive steps that pass the same ``block_ids`` list (the batches of
    one staging) share one copy, so a transfer must not reuse a staged list
    after restaging the buffer; the engine must not change the copy.

    Args:
        step: The step a transfer did not stage.

    Raises:
        NotImplementedError: If the caller collects none: it cannot run
            off-stream steps.
    """
    sink = _step_sink.get()
    if sink is None:
        raise NotImplementedError(
            "This transfer has off-stream steps: call it inside "
            "collect_off_stream_steps()"
        )
    copied = _block_id_copy.get()
    if copied is None or copied[0] is not step.block_ids:
        copied = (step.block_ids, [ids.clone() for ids in step.block_ids])
        _block_id_copy.set(copied)
    sink.append(replace(step, block_ids=copied[1]))


class OffStreamEngine(Protocol):
    """Runs the off-stream steps of retrieves without blocking the caller."""

    def submit(
        self,
        cache_context: BaseCacheContext,
        steps: Sequence[TransferStep],
        on_done: Callable[[bool], None],
    ) -> None:
        """Stage ``steps`` and queue their kernels; return at once. Call on
        the thread that owns the context's stream, after the retrieve queued
        its on-stream work.

        Args:
            cache_context: The retrieve's context.
            steps: The steps its transfers handed off.
            on_done: Called once, on the thread that submitted and under the
                context's device and stream: True after the last step's
                kernels are queued, False if the steps could not all be
                staged (no kernel is queued after the failure).
        """
        ...

    def drain(self, cache_context: BaseCacheContext) -> bool:
        """Wait until every submission of ``cache_context`` is done. Call
        before closing the context, and not on the thread that owns its
        stream. From this call on, a submission for the context fails.

        Args:
            cache_context: The context about to be closed.

        Returns:
            False if a submission is still running after the engine's
            timeout: the context must then stay open.
        """
        ...

    def close(self) -> None:
        """Stop the engine; steps in flight are abandoned."""
        ...
