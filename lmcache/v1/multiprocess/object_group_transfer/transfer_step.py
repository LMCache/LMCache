# SPDX-License-Identifier: Apache-2.0
"""Retrieve steps and when their kernels run.

A retrieve copies chunks into a staging buffer, then runs kernels that move
them into the paged KV cache. One copy with its kernels is a
:class:`TransferStep`.

When the kernels can be queued depends on how the bytes reach the staging:

- **On-stream**: the copy is queued on the GPU stream, so the kernels are
  queued right behind it.
- **Off-stream**: the bytes arrive in GPU memory outside the stream, so the
  kernels wait until an :class:`OffStreamEngine` says the bytes landed.

A caller does not tell the two apart: it runs its transfers inside
:func:`retrieve_transfer` and gets one callback when everything is queued.
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
from lmcache.v1.common.logging import init_logger
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform.base.cache_context import BaseCacheContext

logger = init_logger(__name__)


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


class RetrieveTransfer:
    """The device work of one retrieve, whichever engines move its objects.

    The transfers of the retrieve run inside :func:`retrieve_transfer`. Steps
    of on-stream engines are queued as they run; steps of off-stream engines
    are kept here. :meth:`when_queued` then reports, once, when all of it is
    queued on the stream.
    """

    def __init__(self) -> None:
        self._off_stream: list[tuple[OffStreamEngine | None, TransferStep]] = []
        # The block ids last handed off and the copy made of them. Holding
        # the staged list keeps its identity from being reused.
        self._block_id_copy: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None

    def hand_off(self, engine: "OffStreamEngine | None", step: TransferStep) -> None:
        """Keep an off-stream step for :meth:`when_queued`, with a copy of
        its block ids: the next transfer restages the buffer they live in.

        Consecutive steps that pass the same ``block_ids`` list (the batches
        of one staging) share one copy, so a transfer must not reuse a staged
        list after restaging the buffer; the engine must not change the copy.

        Args:
            engine: The engine that runs the step; None if there is none for
                the step's objects, which fails the retrieve.
            step: The step a transfer did not stage.
        """
        copied = self._block_id_copy
        if copied is None or copied[0] is not step.block_ids:
            copied = (step.block_ids, [ids.clone() for ids in step.block_ids])
            self._block_id_copy = copied
        self._off_stream.append((engine, replace(step, block_ids=copied[1])))

    def when_queued(
        self,
        cache_context: BaseCacheContext,
        succeeded: bool,
        on_done: Callable[[bool], None],
    ) -> None:
        """Call ``on_done`` once all device work of the retrieve is queued on
        the context's stream. Call once, after the last transfer, on the
        thread that owns the stream.

        With only on-stream steps that is now. Off-stream steps are given to
        their engines first, and ``on_done`` runs when the last engine is
        done, on this thread and under the context's device and stream.

        Args:
            cache_context: The retrieve's context.
            succeeded: Whether every transfer of the retrieve ran; if not,
                its off-stream steps are dropped.
            on_done: Receives whether the retrieve's blocks will hold the
                cached bytes: False if a transfer failed, an off-stream step
                has no engine, or an engine could not stage its steps.
        """
        if not succeeded or not self._off_stream:
            on_done(succeeded)
            return
        by_engine: dict[int, tuple[OffStreamEngine, list[TransferStep]]] = {}
        for engine, step in self._off_stream:
            if engine is None:
                logger.error(
                    "Cannot retrieve: no engine runs %d of its steps, whose "
                    "bytes land off the stream",
                    len(self._off_stream),
                )
                on_done(False)
                return
            by_engine.setdefault(id(engine), (engine, []))[1].append(step)
        results: list[bool] = []

        def engine_done(result: bool) -> None:
            results.append(result)
            if len(results) == len(by_engine):
                on_done(all(results))

        for engine, steps in by_engine.values():
            engine.submit(cache_context, steps, engine_done)


_current_retrieve: ContextVar[RetrieveTransfer | None] = ContextVar(
    "lmcache_current_retrieve_transfer", default=None
)


@contextmanager
def retrieve_transfer() -> Iterator[RetrieveTransfer]:
    """Run the transfers of one retrieve: the block's transfers, on this
    thread, belong to the :class:`RetrieveTransfer` it yields.

    The steps reach it this way and not as a return value because a transfer
    runs as the cache context's ``transfer_object_group`` callback, which
    returns nothing on every platform.
    """
    transfer = RetrieveTransfer()
    token = _current_retrieve.set(transfer)
    try:
        yield transfer
    finally:
        _current_retrieve.reset(token)


def hand_off_step(engine: "OffStreamEngine | None", step: TransferStep) -> None:
    """Hand an off-stream step to the retrieve the calling transfer belongs
    to (see :meth:`RetrieveTransfer.hand_off`).

    Args:
        engine: The engine that runs the step, or None if there is none.
        step: The step a transfer did not stage.

    Raises:
        NotImplementedError: If the transfer does not run inside
            :func:`retrieve_transfer`: its caller cannot run off-stream
            steps.
    """
    transfer = _current_retrieve.get()
    if transfer is None:
        raise NotImplementedError(
            "This transfer has off-stream steps: call it inside retrieve_transfer()"
        )
    transfer.hand_off(engine, step)


class OffStreamEngine(Protocol):
    """Runs the off-stream steps of retrieves without blocking the caller."""

    def submit(
        self,
        cache_context: BaseCacheContext,
        steps: Sequence[TransferStep],
        on_done: Callable[[bool], None],
    ) -> None:
        """Stage ``steps`` and queue their kernels; return at once. Called by
        :meth:`RetrieveTransfer.when_queued` on the thread that owns the
        context's stream, after the retrieve queued its on-stream work.

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
