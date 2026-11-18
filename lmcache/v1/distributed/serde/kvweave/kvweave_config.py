# SPDX-License-Identifier: Apache-2.0
"""Configuration and preconditioner support for the KVWeave L2 serde.

Everything here is resolved from the serde's JSON kwargs (the ``"serde"`` dict
of an L2 adapter spec); nothing reads environment variables or a model
``config.json``.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from enum import Enum
from itertools import count
from typing import ClassVar
import threading

# Third Party
import numpy as np
import torch

# First Party
from lmcache.v1.common.logging import init_logger

logger = init_logger(__name__)


class AttentionPlaneLayout(Enum):
    """How one attention group's KV object plane(s) are physically laid out.

    ``SPLIT_KV``: the standard two-plane layout (K and V as separate
    tensors/axis), quantized via the existing K/V-paired codec path.
    ``FUSED_KV``: K and V packed into one tensor's trailing content axis
    (e.g. vLLM's non-MLA blocks-first fused backends) -- still per-head K/V
    data, just packed; quantized via the single-tensor fused codec path.
    ``MLA``: a true compressed latent vector (Multi-head Latent Attention)
    with no per-head K/V structure for the codec to exploit -- never
    quantized.
    ``UNKNOWN``: single-plane (``kv_size == 1``) but the engine KV format
    could not be classified -- never quantized.
    ``AUTO``: decided per chunk from its shape: ``[2, L, T, H]`` is split K/V,
    ``[L, T, H]`` fused K/V, anything else is stored unchanged. A shape cannot
    tell fused K/V from other single-plane formats (MLA, indexer caches), so
    such groups must be declared explicitly.
    """

    SPLIT_KV = "split_kv"
    FUSED_KV = "fused_kv"
    MLA = "mla"
    UNKNOWN = "unknown"
    AUTO = "auto"


@dataclass
class KVWeaveCodecConfig:
    """Typed settings shared by the KVWeave codec and preconditioner."""

    qbit: int = 4
    scaling_method: str = "per_channel"
    rh: bool = True
    asym: bool = True
    precond: bool = False
    num_threads: int = 8
    precond_seed: int = 42
    MAGIC_QUANT: ClassVar[bytes] = b"KVW3"
    MAGIC_QUANT_FUSED: ClassVar[bytes] = b"KVW4"
    DTYPE_TO_CODE: ClassVar[dict[torch.dtype, int]] = {
        torch.float16: 0,
        torch.bfloat16: 1,
        torch.float32: 2,
        # fp8 KV cache (vLLM --kv-cache-dtype fp8); appended so existing payload
        # codes are unchanged.
        torch.float8_e4m3fn: 3,
        torch.float8_e5m2: 4,
    }
    CODE_TO_DTYPE: ClassVar[dict[int, torch.dtype]] = {
        0: torch.float16,
        1: torch.bfloat16,
        2: torch.float32,
        3: torch.float8_e4m3fn,
        4: torch.float8_e5m2,
    }
    SCALING_TO_CODE: ClassVar[dict[str, int]] = {
        "per_tensor": 0,
        "per_token": 1,
        "per_channel": 2,
    }
    SCALE_IDS: ClassVar = count(1)

    def __post_init__(self) -> None:
        self._pd_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._pd_lock = threading.Lock()
        self.scaling_method = str(self.scaling_method)

    @classmethod
    def next_scale_id(cls) -> int:
        """Allocate a process-wide scale id for native KVWeave state."""
        return next(cls.SCALE_IDS) & 0xFFFFFFFF

    @classmethod
    def next_scale_ids(cls, count: int) -> list[int]:
        return [cls.next_scale_id() for _ in range(count)]

    @staticmethod
    def quantized_bytes(elements: int, qbit: int) -> int:
        return (
            (elements + 1) // 2
            if qbit == 4
            else elements
            if qbit <= 8
            else elements * 2
        )

    def rh_precond_tensors(self, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if size <= 0 or size & (size - 1):
            raise ValueError(f"rh requires a power-of-2 transform length, got {size}")
        signs, perm = self.get_pd_matrix(size)
        return (
            torch.as_tensor(signs, dtype=torch.float32).contiguous(),
            torch.as_tensor(perm, dtype=torch.int32).contiguous(),
        )

    def get_pd_matrix(self, hadamard_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the cached deterministic P/D matrices for ``hadamard_size``."""
        cached = self._pd_cache.get(hadamard_size)
        if cached is not None:
            return cached
        with self._pd_lock:
            cached = self._pd_cache.get(hadamard_size)
            if cached is None:
                cached = self._generate_pd(hadamard_size)
                self._pd_cache[hadamard_size] = cached
        return cached

    def _generate_pd(self, hadamard_size: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.precond_seed)
        signs = rng.choice([-1.0, 1.0], size=hadamard_size).astype(np.float32)
        perm = rng.permutation(hadamard_size).astype(np.int32)
        return signs, perm
