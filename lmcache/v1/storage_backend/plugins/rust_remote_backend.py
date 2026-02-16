# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from typing import Any, Callable, List, Optional, Sequence
import asyncio
import json
import os
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StoragePluginInterface,
)

logger = init_logger(__name__)


def _validate_connector_lib_path(connector_lib: str) -> None:
    """Validate the connector library path for security.

    Ensures the library path is either:
    1. An absolute path within a trusted system library directory
    2. An absolute path that has been explicitly resolved (no traversal)

    Raises:
        ValueError: If the path is potentially unsafe
    """
    if not connector_lib:
        raise ValueError("Connector library path cannot be empty")

    # Resolve to absolute path and normalize
    abs_path = os.path.abspath(connector_lib)
    real_path = os.path.realpath(connector_lib)

    # Check for path traversal attempts
    if abs_path != real_path:
        raise ValueError(
            "Connector library path contains symbolic links "
            f"or path traversal: {connector_lib}"
        )

    # Ensure the path doesn't contain path traversal sequences
    if ".." in connector_lib:
        raise ValueError(
            f"Connector library path contains path traversal sequences: {connector_lib}"
        )

    # Additional security: Check if file exists and is a regular file
    if not os.path.isfile(abs_path):
        raise ValueError(
            f"Connector library does not exist or is not a regular file: {abs_path}"
        )

    logger.info(f"Validated connector library path: {abs_path}")


def _load_rust_backend(
    connector_lib: str,
    config_json: str,
):
    """Lazy-import and instantiate the Rust extension."""
    try:
        # Third Party
        from lmcache_rust_remote_backend_io import (
            RustRemoteBackend as _RustBackend,  # type: ignore[import-untyped]
        )
    except ImportError as exc:
        raise RuntimeError(
            "Rust remote-backend extension not installed."
            " Build `rust/remote_backend` and retry."
        ) from exc
    return _RustBackend(connector_lib, config_json)


class RustRemoteBackend(StoragePluginInterface):
    """Remote storage plugin backed by a
    dynamically-loaded Rust/C++ connector.

    The connector shared library is specified via
    ``extra_config["rust_remote.connector_lib"]``.

    Config keys prefixed with ``rust_remote.connector.``
    are forwarded to the connector's
    ``connector_create`` as a JSON object.

    Performance-critical logic (I/O, metadata index,
    put-task dedup) is handled entirely in Rust.
    Python only manages memory allocation and async
    scheduling.
    """

    CONNECTOR_PREFIX = "rust_remote.connector."

    def __init__(
        self,
        config=None,
        metadata=None,
        local_cpu_backend=None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        dst_device: str = "cpu",
    ):
        super().__init__(
            dst_device=dst_device,
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop,
        )
        if self.loop is None:
            raise ValueError("RustRemoteBackend requires an event loop")
        if self.local_cpu_backend is None:
            raise ValueError("RustRemoteBackend requires local_cpu_backend")
        if self.config is None:
            raise ValueError("RustRemoteBackend requires config")

        extra = self.config.extra_config or {}
        connector_lib = str(extra.get("rust_remote.connector_lib", ""))
        if not connector_lib:
            raise ValueError("extra_config['rust_remote.connector_lib'] is required")

        # Validate connector library path for security
        _validate_connector_lib_path(connector_lib)

        # Collect connector-specific config
        conn_cfg: dict[str, Any] = {}
        pfx = self.CONNECTOR_PREFIX
        for k, v in extra.items():
            if k.startswith(pfx):
                conn_cfg[k[len(pfx) :]] = v
        config_json = json.dumps(conn_cfg)

        self._backend = _load_rust_backend(connector_lib, config_json)

        # Shapes/dtypes/fmt cache for get_blocking.
        # Only the *first* put per key records these
        # so get_blocking can reconstruct the MemoryObj
        # with the correct allocation shape.
        self._meta_lock = threading.Lock()
        self._meta_shapes: dict[str, list[torch.Size]] = {}
        self._meta_dtypes: dict[str, list[torch.dtype]] = {}
        self._meta_fmts: dict[str, MemoryFormat] = {}

        logger.info(
            "RustRemoteBackend initialized, connector_lib=%s",
            connector_lib,
        )

    def __str__(self) -> str:
        return "RustRemoteBackend"

    # -- helpers -----------------------------------------

    @staticmethod
    def _key_str(key: CacheEngineKey) -> str:
        return key.to_string()

    # -- StorageBackendInterface -------------------------

    def contains(
        self,
        key: CacheEngineKey,
        pin: bool = False,
    ) -> bool:
        return self._backend.exists(self._key_str(key))

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return self._backend.in_put_tasks(self._key_str(key))

    def pin(self, key: CacheEngineKey) -> bool:
        logger.debug("RustRemoteBackend does not support pin. No-op, returning True.")
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        logger.debug("RustRemoteBackend does not support unpin. No-op, returning True.")
        return True

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        return self._backend.remove(self._key_str(key))

    def get_allocator_backend(
        self,
    ) -> AllocatorBackendInterface:
        assert self.local_cpu_backend is not None
        return self.local_cpu_backend

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        if self.local_cpu_backend is None:
            return None

        key_str = self._key_str(key)

        # Determine allocation shape from cached meta
        # or fall back to default metadata shapes.
        with self._meta_lock:
            shapes = self._meta_shapes.get(key_str)
            dtypes = self._meta_dtypes.get(key_str)
            fmt = self._meta_fmts.get(key_str)

        if shapes is not None and dtypes is not None:
            alloc_fmt = fmt if fmt is not None else MemoryFormat.KV_2LTD
        else:
            assert self.local_cpu_backend.metadata is not None
            md = self.local_cpu_backend.metadata
            shapes = md.get_shapes()
            dtypes = md.get_dtypes()
            alloc_fmt = MemoryFormat.KV_2LTD

        memory_obj = self.local_cpu_backend.allocate(shapes, dtypes, alloc_fmt)
        if memory_obj is None:
            logger.warning("Failed to allocate memory for get")
            return None

        buf = memory_obj.byte_array
        if hasattr(buf, "cast"):
            buf = buf.cast("B")

        # Zero-copy read: Rust reads directly into
        # the tensor's underlying memory.
        num_read = self._backend.get_into(key_str, buf)
        if num_read is None:
            memory_obj.ref_count_down()
            return None
        return memory_obj

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ):
        if self.loop is None:
            return None

        futures = []
        for key, obj in zip(keys, objs, strict=True):
            key_str = self._key_str(key)

            # Dedup: Rust-side atomic check-and-add.
            if not self._backend.try_add_put_task(key_str):
                continue

            # Cache allocation metadata for
            # get_blocking reconstruction.
            data_len = len(obj.byte_array)
            self._backend.record_meta(key_str, data_len)
            with self._meta_lock:
                self._meta_shapes[key_str] = obj.get_shapes()
                self._meta_dtypes[key_str] = obj.get_dtypes()
                self._meta_fmts[key_str] = obj.get_memory_format()

            obj.ref_count_up()
            assert self.loop is not None
            fut = asyncio.run_coroutine_threadsafe(
                self._do_put(
                    key,
                    key_str,
                    obj,
                    on_complete_callback,
                ),
                self.loop,
            )
            futures.append(fut)
        return futures or None

    async def _do_put(
        self,
        key: CacheEngineKey,
        key_str: str,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        try:
            buf = memory_obj.byte_array
            if hasattr(buf, "cast"):
                buf = buf.cast("B")

            backend = self._backend

            # GIL-released blocking write via Rust.
            def _write():
                backend.put_blocking(key_str, buf)

            await asyncio.to_thread(_write)

            if on_complete_callback is not None:
                try:
                    on_complete_callback(key)
                except Exception as exc:
                    logger.warning(
                        "on_complete_callback failed for key %s: %s",
                        key,
                        exc,
                    )
        finally:
            memory_obj.ref_count_down()
            self._backend.remove_put_task(key_str)

    def close(self) -> None:
        self._backend.close()
        logger.info("RustRemoteBackend closed.")
