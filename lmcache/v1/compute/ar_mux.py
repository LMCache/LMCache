# SPDX-License-Identifier: Apache-2.0
"""Serialize TP all-reduces onto one CUDA stream (AR multiplexer).

Under TP>1, prefill and blend recompute share the same NCCL/custom communicator.
Launching collectives concurrently from two compute streams deadlocks (Fix C
hangs: jobs 15002378, 15099974).

This module monkeypatches ``GroupCoordinator._all_reduce_out_place`` so every
TP all-reduce:

1. Waits for the producer (current) stream on ``ar_stream``
2. Runs the collective on ``ar_stream`` (total order across prefill + blend)
3. Records ``producer.wait_stream(ar_stream)`` so the caller may use the result

Local GEMMs on other streams can therefore overlap, while ARs stay ordered.

Enable with ``LMCACHE_AR_MUX=1`` (adapter also turns on recompute-on-blend-stream
when combined with ``LMCACHE_BATCHED_BLEND_OVERLAP=1``).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from lmcache.logging import init_logger

logger = init_logger(__name__)


class ARMultiplexer:
    """Process-wide AR mux. One instance per worker process."""

    __slots__ = (
        "enabled",
        "ar_stream",
        "_installed",
        "_orig_out_place",
        "_force_pynccl",
        "count",
    )

    def __init__(self) -> None:
        self.enabled = False
        self.ar_stream: Optional[torch.cuda.Stream] = None
        self._installed = False
        self._orig_out_place = None
        self._force_pynccl = True
        self.count = 0

    @property
    def active(self) -> bool:
        return self.enabled and self._installed and self.ar_stream is not None

    def ensure_installed(self, force_pynccl: bool = True) -> None:
        """Patch vLLM TP all-reduce. Idempotent. Safe to call before CUDA init
        of the stream (stream is created lazily on first use / enable)."""
        if self._installed:
            return
        if not torch.cuda.is_available():
            logger.warning("[ar-mux] CUDA unavailable; AR mux not installed")
            return

        # Third Party
        from vllm.distributed.parallel_state import (
            GroupCoordinator,
            set_custom_all_reduce,
        )

        self._force_pynccl = bool(force_pynccl)
        if self._force_pynccl:
            # Custom AR has no stream arg and uses IPC assumptions that are
            # harder to reason about under remuxing. Prefer pynccl, which
            # honors the current CUDA stream (our ar_stream context).
            set_custom_all_reduce(False)
            logger.info(
                "[ar-mux] disabled vLLM custom all-reduce (pynccl path for stream control)"
            )

        if self.ar_stream is None:
            self.ar_stream = torch.cuda.Stream()

        self._orig_out_place = GroupCoordinator._all_reduce_out_place
        mux = self

        def _muxed_all_reduce_out_place(gc_self, input_: torch.Tensor) -> torch.Tensor:
            if not mux.active:
                return mux._orig_out_place(gc_self, input_)

            producer = torch.cuda.current_stream()
            ar = mux.ar_stream
            assert ar is not None
            # Already on the AR stream (nested / direct call): do not re-enter.
            if producer.cuda_stream == ar.cuda_stream:
                return mux._orig_out_place(gc_self, input_)

            ar.wait_stream(producer)
            with torch.cuda.stream(ar):
                out = mux._orig_out_place(gc_self, input_)
            # Caller continues on producer; result is ready after this wait.
            producer.wait_stream(ar)
            mux.count += 1
            return out

        GroupCoordinator._all_reduce_out_place = _muxed_all_reduce_out_place  # type: ignore[method-assign]
        self._installed = True
        logger.info("[ar-mux] installed (GroupCoordinator._all_reduce_out_place patched)")

    def enable(self) -> None:
        self.ensure_installed(force_pynccl=self._force_pynccl)
        if not self._installed:
            return
        if self.ar_stream is None:
            self.ar_stream = torch.cuda.Stream()
        self.enabled = True
        logger.info("[ar-mux] enabled (ar_stream=%s)", self.ar_stream)

    def disable(self) -> None:
        self.enabled = False

    def wait_for_ar(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Make ``stream`` (default: current) wait for outstanding muxed ARs."""
        if self.ar_stream is None:
            return
        s = stream if stream is not None else torch.cuda.current_stream()
        s.wait_stream(self.ar_stream)


_MUX: Optional[ARMultiplexer] = None


def get_ar_mux() -> ARMultiplexer:
    global _MUX
    if _MUX is None:
        _MUX = ARMultiplexer()
    return _MUX


def ar_mux_env_requested() -> bool:
    return os.environ.get("LMCACHE_AR_MUX", "0") == "1"
