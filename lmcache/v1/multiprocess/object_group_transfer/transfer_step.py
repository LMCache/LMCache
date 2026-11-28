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
    """The buffer a step is staged in; chunk ``i`` of the step is in slot ``i``.

    A cache context is one; an off-stream engine passes a view of its own.
    """

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
        """The first ``num_bytes`` bytes as one flat region; None if too many."""
        ...


@dataclass
class TransferStep:
    """One retrieve step: objects to stage and the kernels that follow.

    Attributes:
        memory_objs: The objects to stage, in chunk order; all of one engine.
        block_ids: The block ids the kernels use.
        targets: ``staging -> regions``: where each object lands.
        launch: ``(staging, block_ids) -> None``: queues the kernels.
        skipped: Leading chunks the kernels skip; they need no slot.
        spans_slots: The object may run past its slot into the next ones.
    """

    memory_objs: Sequence[MemoryObj]
    block_ids: list[torch.Tensor]
    targets: Callable[[Staging], list[torch.Tensor]]
    launch: Callable[[Staging, list[torch.Tensor]], None]
    skipped: int = 0
    spans_slots: bool = False


class RetrieveTransfer:
    """The device work of one retrieve.

    On-stream steps are queued as they run; off-stream steps are kept here
    until :meth:`when_queued` gives them to their engines.
    """

    def __init__(self) -> None:
        self._off_stream: list[tuple[OffStreamEngine | None, TransferStep]] = []
        # The staged block ids last handed off, and their copy.
        self._block_id_copy: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None

    def hand_off(self, engine: "OffStreamEngine | None", step: TransferStep) -> None:
        """Keep an off-stream step, with its own copy of the block ids.

        The next transfer restages the buffer the ids live in. Steps that
        pass the same ``block_ids`` list share one copy.

        Args:
            engine: The engine that runs the step; None fails the retrieve.
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
        """Call ``on_done`` once all the retrieve's device work is queued.

        With only on-stream steps that is now; otherwise when the last
        off-stream engine is done. Call once, after the last transfer.

        Args:
            cache_context: The retrieve's context.
            succeeded: Whether every transfer ran; if not, the off-stream
                steps are dropped.
            on_done: Receives True if the blocks will hold the cached bytes.
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
    """Scope the transfers of one retrieve, on this thread.

    A context variable carries the steps because ``transfer_object_group``
    returns nothing on every platform.
    """
    transfer = RetrieveTransfer()
    token = _current_retrieve.set(transfer)
    try:
        yield transfer
    finally:
        _current_retrieve.reset(token)


def hand_off_step(engine: "OffStreamEngine | None", step: TransferStep) -> None:
    """Hand an off-stream step to the current retrieve.

    Args:
        engine: The engine that runs the step, or None if there is none.
        step: The step a transfer did not stage.

    Raises:
        NotImplementedError: Outside :func:`retrieve_transfer`.
    """
    transfer = _current_retrieve.get()
    if transfer is None:
        raise NotImplementedError(
            "This transfer has off-stream steps: call it inside retrieve_transfer()"
        )
    transfer.hand_off(engine, step)


class OffStreamEngine(Protocol):
    """Runs off-stream retrieve steps without blocking the caller."""

    def submit(
        self,
        cache_context: BaseCacheContext,
        steps: Sequence[TransferStep],
        on_done: Callable[[bool], None],
    ) -> None:
        """Stage ``steps`` and queue their kernels; return at once.

        Args:
            cache_context: The retrieve's context.
            steps: The steps its transfers handed off.
            on_done: Called once on the submitting thread: True when the
                last kernels are queued, False if staging failed.
        """
        ...

    def drain(self, cache_context: BaseCacheContext) -> bool:
        """Wait for the context's submissions; call before closing it.

        Args:
            cache_context: The context about to be closed.

        Returns:
            False if one is still running after the engine's timeout.
        """
        ...

    def close(self) -> None:
        """Stop the engine; steps in flight are abandoned."""
        ...
