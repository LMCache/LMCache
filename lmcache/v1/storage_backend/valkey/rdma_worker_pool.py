# SPDX-License-Identifier: Apache-2.0
"""
Valkey worker pool that moves KV cache bytes over RDMA.

``ValkeyRdmaWorkerPool`` is a ``ValkeyWorkerPool`` whose SET and GET never
carry the value in the RESP stream. Each worker thread's glide client is
configured for RDMA and registers the whole L1 arena once; every operation
then names a window of that registration, and the server moves the bytes
itself with the valkey-large-object module's ``BLOB.GET`` / ``BLOB.SET``.

One registration per worker is forced by glide's API: an ``RdmaRegion`` is
bound to the client that registered it and carries one transfer at a time.
The window is derived from the ``memoryview`` the adapter already passes, so
the submit API is unchanged. A buffer outside the arena fails its key; in MP
mode every object lives in the arena.

The keys are the module's own data type, so a plain ``GET`` / ``SET`` cannot
read or replace them. ``EXISTS`` and ``DEL`` are inherited unchanged, and
``ttl_seconds`` is applied with ``EXPIRE`` after each store.

Requires ``valkey-glide-sync`` built with RDMA support and libfabric on the
machine; both are checked when the first worker builds its client.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Callable, Optional, TypeVar
import ctypes
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.valkey.worker_pool import (
    GET_MISS,
    ValkeyWorkerPool,
)

logger = init_logger(__name__)

#: Fabric providers glide can carry RDMA transfers over.
RDMA_PROVIDERS: tuple[str, ...] = ("efa-direct", "tcp")

#: First ``valkey-glide-sync`` release whose sync client exposes the RDMA API.
RDMA_MIN_GLIDE_VERSION: str = "2.6.0"

T = TypeVar("T")


class ValkeyRdmaWorkerPool(ValkeyWorkerPool):
    """``ValkeyWorkerPool`` whose SET and GET move bytes over RDMA.

    Takes the L1 arena (from the storage manager's ``L1MemoryDesc``), the
    fabric provider, and every argument the plain pool takes.
    """

    def __init__(
        self,
        *,
        l1_base: int,
        l1_size: int,
        rdma_provider: str = "efa-direct",
        rdma_interface: Optional[str] = None,
        **pool_kwargs: Any,
    ) -> None:
        """Validate the RDMA settings, then build and warm up the pool.

        Warm-up registers the arena on every worker, so a setup that cannot do
        RDMA fails here rather than on the first transfer.

        Args:
            l1_base: Virtual address where the L1 arena starts (> 0).
            l1_size: Size of the L1 arena in bytes (> 0).
            rdma_provider: ``"efa-direct"`` for EFA hardware, or ``"tcp"``
                for libfabric's software provider (development and tests).
            rdma_interface: Fabric domain to pin to on a multi-card host;
                ``None`` lets the provider choose.
            **pool_kwargs: Forwarded to ``ValkeyWorkerPool``.

        Raises:
            ValueError: If the arena or an RDMA setting is invalid.
            RuntimeError: If RDMA is unavailable on this machine, the fabric
                refuses the arena, or a connection fails.
        """
        if l1_base <= 0:
            raise ValueError(f"l1_base must be a nonzero address, got {l1_base:#x}")
        if l1_size <= 0:
            raise ValueError(f"l1_size must be > 0, got {l1_size}")
        if rdma_provider not in RDMA_PROVIDERS:
            raise ValueError(
                f"rdma_provider must be one of {RDMA_PROVIDERS}, got {rdma_provider!r}"
            )
        if rdma_interface is not None and not rdma_interface:
            raise ValueError("rdma_interface must not be empty if set")

        self._l1_base: int = l1_base
        self._l1_size: int = l1_size
        self._rdma_provider: str = rdma_provider
        self._rdma_interface: Optional[str] = rdma_interface
        # A buffer-protocol view of the arena for registration; the L1 manager
        # owns the memory.
        self._l1_memory = (ctypes.c_ubyte * l1_size).from_address(l1_base)

        super().__init__(**pool_kwargs)

        logger.info(
            "ValkeyRdmaWorkerPool: provider=%s interface=%s l1_arena=[%#x, %#x) "
            "(%d bytes) on %d workers",
            rdma_provider,
            rdma_interface,
            l1_base,
            l1_base + l1_size,
            l1_size,
            self.num_workers,
        )

    @property
    def rdma_provider(self) -> str:
        """The fabric provider each client opened."""
        return self._rdma_provider

    @property
    def rdma_interface(self) -> Optional[str]:
        """The fabric domain each client was pinned to, if any."""
        return self._rdma_interface

    @property
    def l1_arena_bytes(self) -> int:
        """Bytes of L1 arena each worker registered."""
        return self._l1_size

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Return the calling thread's client, with the arena registered on it.

        The RDMA support checks run before the first client is built on a
        thread, so an unusable machine fails with a message naming what is
        missing rather than a fabric error from inside glide.

        Returns:
            The thread's ``GlideClient`` or ``GlideClusterClient``.

        Raises:
            RuntimeError: If RDMA is unavailable here, or the fabric refuses
                to register the arena (see :meth:`_registration_failure_message`).
        """
        if getattr(self._local, "client", None) is None:
            self._require_rdma_support()
        client = super()._get_client()
        if getattr(self._local, "region", None) is None:
            try:
                self._local.region = client.register_rdma_region(self._l1_memory)
            except Exception as exc:
                raise RuntimeError(self._registration_failure_message(exc)) from exc
        return client

    def _client_config_extras(self, glide_sync: Any) -> dict[str, Any]:
        """Add the ``RdmaConfiguration`` for the chosen provider."""
        if self._rdma_provider == "tcp":
            provider = glide_sync.RdmaProvider.Tcp()
        else:
            provider = glide_sync.RdmaProvider.EfaDirect()
        return {
            "rdma": glide_sync.RdmaConfiguration(
                provider=provider, interface=self._rdma_interface
            )
        }

    def _close_local_client(self, barrier: threading.Barrier) -> None:
        """Close the thread's client, then deregister its region.

        Client first: closing it cancels any transfer still in flight.
        """
        super()._close_local_client(barrier)
        self._drop_region()

    # ------------------------------------------------------------------
    # Worker-side primitives (run on a worker thread)
    # ------------------------------------------------------------------

    def _do_set(self, key_str: str, data: Any) -> None:
        """``BLOB.SET`` a key from the arena window ``data`` occupies.

        The server reads the bytes out of L1 memory itself. With
        ``ttl_seconds`` configured the key is then given that expiry.

        Args:
            key_str: Wire key.
            data: Writable buffer inside the L1 arena holding the value.

        Raises:
            ValueError: If ``data`` is empty or outside the arena.
            RdmaError: If the server refuses the write.
        """
        view, offset = self._window(data)
        if view.nbytes == 0:
            raise ValueError("BLOB.SET rejects an empty value")
        client = self._get_client()
        key = key_str.encode()
        self._transfer(
            lambda region: client.rdma_set(key, region.window(offset, view.nbytes))
        )
        if self._ttl_seconds is not None:
            client.expire(key, self._ttl_seconds)

    def _do_get_into(self, key_str: str, buf: Any) -> int:
        """``BLOB.GET`` a key into the arena window ``buf`` occupies.

        The server writes the bytes into L1 memory itself and is told the
        window's length, so a larger value is refused before anything lands.
        A smaller value lands but counts as a miss: fixed-size chunks must
        round-trip exactly, as with the plain pool.

        Args:
            key_str: Wire key.
            buf: Writable buffer inside the L1 arena to land the value in.

        Returns:
            The buffer length on a hit, or :data:`GET_MISS` if the key is
            absent or its value is smaller than the buffer.

        Raises:
            ValueError: If ``buf`` is outside the arena.
            RdmaError: If the value is larger than ``buf`` or the transfer
                fails.
        """
        view, offset = self._window(buf)
        size = view.nbytes
        client = self._get_client()
        key = key_str.encode()
        receipt = self._transfer(
            lambda region: client.rdma_get(key, region.window(offset, size))
        )
        if receipt is None:
            return GET_MISS
        landed = int(receipt.bytes_written)
        if landed != size:
            logger.warning(
                "Valkey RDMA GET size mismatch for key %s: landed %d bytes, "
                "expected %d; treating as miss.",
                key_str,
                landed,
                size,
            )
            return GET_MISS
        return landed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_rdma_support() -> None:
        """Raise ``RuntimeError`` naming what is missing unless RDMA works here.

        ``glide_sync`` is imported here, not at module load, for the same
        reason as in ``ValkeyWorkerPool``: the dependency stays optional.
        """
        try:
            # Third Party
            import glide_sync  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError(
                "Valkey RDMA support requires the glide_sync module. "
                f"Install: pip install 'valkey-glide-sync>={RDMA_MIN_GLIDE_VERSION}' "
                "(note: the plain 'valkey-glide' package is async-only and has "
                "no RDMA API)"
            ) from e

        client_cls = glide_sync.GlideClient
        if not hasattr(client_cls, "rdma_available"):
            raise RuntimeError(
                "The installed valkey-glide-sync has no RDMA API. "
                f"Install: pip install 'valkey-glide-sync>={RDMA_MIN_GLIDE_VERSION}'"
            )
        if not client_cls.rdma_available():
            raise RuntimeError(
                "The installed valkey-glide-sync was built without RDMA support. "
                "Install a published wheel, or build from source with "
                "GLIDE_SYNC_RDMA=1."
            )
        if not client_cls.rdma_usable():
            raise RuntimeError(
                "valkey-glide-sync has RDMA support but libfabric is not available "
                "on this machine. Install libfabric (for example the libfabric "
                "package, or the EFA installer on AWS) so the client can open a "
                "fabric."
            )

    def _registration_failure_message(self, exc: Exception) -> str:
        """Explain a failed arena registration in terms the operator can act on.

        The usual cause is the device's cap on total registered memory: every
        worker registers the whole arena, and on EFA the cap is the device's
        ``max_mr_size``, which libfabric reports only as ENOMEM.
        """
        total_gib = self._l1_size * self.num_workers / (1 << 30)
        return (
            f"Registering the L1 arena ({self._l1_size >> 20} MiB) with the fabric "
            f"failed: {exc}. Each of the {self.num_workers} workers registers the "
            f"whole arena, {total_gib:.1f} GiB in total, and the fabric device caps "
            "total registered memory (on EFA, at its max_mr_size; see "
            "`ibv_devinfo -v`). Lower num_workers or --l1-size-gb so their product "
            "fits, and check `ulimit -l` is unlimited."
        )

    def _region(self) -> Any:
        """Return the calling thread's arena registration, creating it if needed."""
        self._get_client()
        return self._local.region

    def _drop_region(self) -> None:
        """Deregister the calling thread's arena registration, if any."""
        region = getattr(self._local, "region", None)
        if region is None:
            return
        self._local.region = None
        try:
            region.close()
        except Exception as exc:
            logger.debug("Error closing per-thread RDMA region: %s", exc)

    def _window(self, buf: Any) -> tuple[memoryview, int]:
        """Locate ``buf`` inside the registered arena.

        Args:
            buf: A writable buffer.

        Returns:
            ``buf`` as a ``"B"`` memoryview, and its offset from the arena
            base.

        Raises:
            ValueError: If ``buf`` is read-only or outside the arena, which
                nothing registered can reach.
        """
        view = buf if isinstance(buf, memoryview) else memoryview(buf)
        if view.format != "B":
            view = view.cast("B")
        if view.readonly:
            raise ValueError(
                "buffer is read-only, so it is not L1 arena memory and cannot "
                "be transferred over RDMA"
            )
        address = ctypes.addressof(ctypes.c_char.from_buffer(view))
        offset = address - self._l1_base
        if offset < 0 or offset + view.nbytes > self._l1_size:
            raise ValueError(
                f"buffer at {address:#x} ({view.nbytes} bytes) is outside the "
                f"registered L1 arena [{self._l1_base:#x}, "
                f"{self._l1_base + self._l1_size:#x}), so it cannot be "
                "transferred over RDMA"
            )
        return view, offset

    def _transfer(self, operate: Callable[[Any], T]) -> T:
        """Run ``operate`` against the thread's region, re-registering once.

        glide revokes a region when a transfer fails without a server reply,
        and refuses the next transfer through it as revoked. Only that refusal
        triggers a fresh registration and one retry; a server's own error
        reply leaves the region usable.

        Args:
            operate: Called with the region; its result is returned.

        Returns:
            Whatever ``operate`` returns.

        Raises:
            Exception: Whatever ``operate`` raises, other than a single
                revoked-region refusal.
        """
        try:
            return operate(self._region())
        except Exception as exc:
            if "revoked" not in str(exc):
                raise
            logger.warning(
                "RDMA region on %s was revoked (%s); registering the L1 arena again",
                threading.current_thread().name,
                exc,
            )
            self._drop_region()
            return operate(self._region())
