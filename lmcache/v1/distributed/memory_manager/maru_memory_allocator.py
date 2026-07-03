# SPDX-License-Identifier: Apache-2.0

"""CXL-backed L1 allocator wrapping the external maru runtime.

A thin wrapper over ``maru_lmcache.CxlMemoryAdapter``. The maru packages are
imported lazily in :meth:`init_layout` so this module stays importable without
the maru runtime. Two-phase init: the CXL pool is typed by the KV layout, so
the ``MaruHandler`` / ``CxlMemoryAdapter`` are built on the first layout
registration, not at construction. ``free`` is a no-op -- page lifecycle is
owned by MaruServer (pin/unpin/delete), not LMCache refcounts.

Single-model per instance: the pool is fixed to the first layout; a different
layout is rejected. TODO(maru-multi-model): partition the pool per layout key.
"""

# Standard
from typing import TYPE_CHECKING, List, Optional, Union

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import MaruL1Config
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)

if TYPE_CHECKING:
    # Third Party
    from maru import MaruHandler
    from maru_handler.memory import AllocHandle
    from maru_lmcache import CxlMemoryAdapter

logger = init_logger(__name__)


def _to_tcp(url: str) -> str:
    """Rewrite a ``maru://`` URL to the ``tcp://`` scheme maru expects."""
    if url.startswith("maru://"):
        return "tcp://" + url[len("maru://") :]
    return url


class MaruMemoryAllocator(MemoryAllocatorInterface):
    """L1 allocator over a shared CXL pool (maru). See module docstring."""

    def __init__(self, config: MaruL1Config) -> None:
        self._config = config
        self._handler: "MaruHandler | None" = None
        self._cxl_adapter: "CxlMemoryAdapter | None" = None
        self._shapes: list[torch.Size] | None = None
        self._dtypes: list[torch.dtype] | None = None
        self._fmt: MemoryFormat | None = None
        self._chunk_size_in_tokens: int = 0
        self._single_token_size: int = 0

    def init_layout(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        fmt: MemoryFormat,
        chunk_size_in_tokens: int,
    ) -> None:
        """Bring up the CXL pool for the KV layout (idempotent per layout).

        Connects the MaruHandler and builds the CxlMemoryAdapter on the first
        call. A later call with the same layout is a no-op; a different layout
        is rejected (single-model; TODO(maru-multi-model)). Called serially at
        startup -- not thread-safe.
        """
        if chunk_size_in_tokens <= 0:
            raise ValueError("chunk_size_in_tokens must be > 0")

        if self._cxl_adapter is not None:
            if (
                self._shapes != shapes
                or self._dtypes != dtypes
                or self._fmt != fmt
                or self._chunk_size_in_tokens != chunk_size_in_tokens
            ):
                raise ValueError(
                    "MaruMemoryAllocator is single-model: the KV layout changed "
                    "on a later init_layout call (TODO(maru-multi-model))."
                )
            return

        chunk_bytes = get_size_bytes(shapes, dtypes)
        if chunk_bytes <= 0 or chunk_bytes % chunk_size_in_tokens:
            raise ValueError(
                f"chunk size {chunk_bytes} bytes is not a positive multiple of "
                f"{chunk_size_in_tokens} tokens"
            )

        # maru runtime is only needed once a layout is bound; import lazily so
        # the module loads on non-maru deployments.
        # Third Party
        from maru import MaruConfig, MaruHandler
        from maru_lmcache import CxlMemoryAdapter

        maru_config = MaruConfig(
            server_url=_to_tcp(self._config.server_url),
            instance_id=self._config.instance_id,
            pool_size=self._config.pool_size_bytes,
            chunk_size_bytes=chunk_bytes,
            auto_connect=False,
            timeout_ms=self._config.timeout_ms,
            use_async_rpc=self._config.use_async_rpc,
            max_inflight=self._config.max_inflight,
            eager_map=self._config.eager_map,
        )
        handler = MaruHandler(maru_config)
        if not handler.connect():
            raise RuntimeError(
                f"failed to connect MaruHandler to {self._config.server_url}"
            )

        self._handler = handler
        try:
            self._cxl_adapter = CxlMemoryAdapter(
                handler=handler,
                shapes=shapes,
                dtypes=dtypes,
                fmt=fmt,
                chunk_size=handler.get_chunk_size(),
            )
        except Exception:
            handler.close()
            self._handler = None
            raise
        self._shapes = shapes
        self._dtypes = dtypes
        self._fmt = fmt
        self._chunk_size_in_tokens = chunk_size_in_tokens
        self._single_token_size = chunk_bytes // chunk_size_in_tokens

    @property
    def is_initialized(self) -> bool:
        """Whether :meth:`init_layout` has bound the CXL pool."""
        return self._cxl_adapter is not None

    def _require_init(self, op: str) -> "CxlMemoryAdapter":
        if self._cxl_adapter is None:
            raise RuntimeError(f"MaruMemoryAllocator.{op} called before init_layout()")
        return self._cxl_adapter

    # MemoryAllocatorInterface

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        return self._require_init("allocate").allocate(
            shapes, dtypes, fmt, allocator_type
        )

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        return self._require_init("batched_allocate").batched_allocate(
            shapes, dtypes, batch_size, fmt, allocator_type
        )

    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None) -> None:
        """No-op: CXL page lifecycle is owned by MaruServer."""

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        """No-op: CXL page lifecycle is owned by MaruServer."""

    # maru-specific surface (called by MaruL1Manager)

    def get_by_location(
        self,
        region_id: int,
        page_index: int,
        actual_size: int,
        single_token_size: Optional[int] = None,
    ) -> Optional[MemoryObj]:
        """Materialize a zero-copy ``MemoryObj`` for a CXL page (no new alloc)."""
        adapter = self._require_init("get_by_location")
        if single_token_size is None:
            single_token_size = self._single_token_size
        return adapter.get_by_location(
            region_id=region_id,
            page_index=page_index,
            actual_size=actual_size,
            single_token_size=single_token_size,
        )

    def create_store_handle(self, memory_obj: MemoryObj) -> "AllocHandle":
        """Build the maru store handle for an allocated page."""
        return self._require_init("create_store_handle").create_store_handle(memory_obj)

    def abort_alloc(self, memory_obj: MemoryObj) -> None:
        """Discard an allocated-but-unregistered page (return it to the owner)."""
        self._require_init("abort_alloc").free(memory_obj)

    @property
    def handler(self) -> "MaruHandler":
        """The connected MaruHandler (raises before init_layout())."""
        if self._handler is None:
            raise RuntimeError(
                "MaruMemoryAllocator.handler accessed before init_layout()"
            )
        return self._handler

    @property
    def single_token_size(self) -> int:
        """Per-token KV byte size (raises before init_layout())."""
        if self._cxl_adapter is None:
            raise RuntimeError(
                "MaruMemoryAllocator.single_token_size accessed before init_layout()"
            )
        return self._single_token_size

    # lifecycle

    def close(self) -> None:
        if self._cxl_adapter is not None:
            self._cxl_adapter.close()
            self._cxl_adapter = None
        if self._handler is not None:
            self._handler.close()
            self._handler = None

    def memcheck(self) -> bool:
        return True
