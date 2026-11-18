# SPDX-License-Identifier: Apache-2.0
"""
Valkey worker pool that stores KV cache bytes as valkey-large-object objects
over an ordinary connection.

``ValkeyBlobTcpWorkerPool`` is the ``ValkeyWorkerPool`` behind the ``valkey``
adapter's ``transport: "blob-tcp"``. It sends the valkey-large-object
module's ``BLOB.TCP_SET key <data>`` and ``BLOB.TCP_GET key`` through glide's
``custom_command``, so the value travels in the RESP stream and the host
needs no fabric, no libfabric, and no RDMA build of glide.

The keys are the module's own data type, the same type ``transport: "rdma"``
writes, so hosts on either transport share objects under one ``key_prefix``.
A plain ``GET`` cannot read them and a plain ``SET`` replaces them.
``EXISTS`` and ``DEL`` are inherited unchanged, and ``ttl_seconds`` is
applied with ``EXPIRE`` after each store, because ``BLOB.TCP_SET`` takes no
expiry argument.

Each worker checks the module with ``BLOB.INFO`` when it builds its client,
so a server without the module, or with it disabled, fails adapter
construction rather than every store. In cluster mode the check reaches only
the primary that owns the probe key; a shard elsewhere without the module
fails its keys at runtime instead. ``request_timeout`` bounds each request,
value included.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.l2_adapters.valkey.worker_pool import (
    GET_MISS,
    ValkeyWorkerPool,
)

logger = init_logger(__name__)

#: Key the startup check reads with ``BLOB.INFO``. ``BLOB.INFO`` never
#: creates it, and the pool does not know the configured ``key_prefix``.
PROBE_KEY: bytes = b"__lmcache_blob_tcp_probe__"

_SET: bytes = b"BLOB.TCP_SET"
_GET: bytes = b"BLOB.TCP_GET"
_INFO: bytes = b"BLOB.INFO"


class ValkeyBlobTcpWorkerPool(ValkeyWorkerPool):
    """``ValkeyWorkerPool`` that stores values as valkey-large-object objects.

    Takes every argument the plain pool takes.
    """

    def _get_client(self) -> Any:
        """Return the calling thread's client, checking the module when new.

        The check runs when a worker thread builds its client, normally during
        warm-up, so later requests pay nothing for it. A client that fails the
        check is closed and dropped, so the thread's next request checks again
        and reports the module problem rather than a raw command error.

        Returns:
            The thread's ``GlideClient`` or ``GlideClusterClient``.

        Raises:
            RuntimeError: If the server does not have the valkey-large-object
                module loaded, or has it disabled.
            RequestError: Any other error from the check, unchanged.
        """
        is_new = getattr(self._local, "client", None) is None
        client = super()._get_client()
        if is_new:
            try:
                self._check_module(client)
            except Exception:
                # Clean up and re-raise the check's error unchanged; a close
                # failure must not replace the module diagnosis.
                self._local.client = None
                try:
                    client.close()
                except Exception as close_exc:
                    logger.debug(
                        "Error closing glide client after a failed check: %s",
                        close_exc,
                    )
                raise
        return client

    def _do_set(self, key_str: str, data: Any) -> None:
        """``BLOB.TCP_SET`` a key, then give it ``ttl_seconds`` if configured.

        Args:
            key_str: Wire key.
            data: The value, as a ``memoryview``, ``bytes`` or ``bytearray``.
                glide reads it in place.

        Raises:
            ValueError: If ``data`` is empty; the module rejects empty
                objects, so nothing is sent.
            RequestError: If the server refuses the store.
        """
        view = data if isinstance(data, memoryview) else memoryview(data)
        # glide sends len(arg) as the value length, which for a non-byte view
        # is the element count: a float32 view would be cut to a quarter.
        if view.format != "B":
            view = view.cast("B")
        if view.nbytes == 0:
            raise ValueError("BLOB.TCP_SET rejects an empty value")
        client = self._get_client()
        key = key_str.encode()
        client.custom_command([_SET, key, view])
        if self._ttl_seconds is not None:
            client.expire(key, self._ttl_seconds)

    def _do_get_into(self, key_str: str, buf: Any) -> int:
        """``BLOB.TCP_GET`` a key and copy it into ``buf``.

        Fixed-size chunks must round-trip exactly, so a value of any other
        size is a miss and ``buf`` is left untouched. The reply is a fresh
        ``bytes`` object rather than slab memory, so it is copied into
        ``buf`` directly, without the base pool's scratch buffer.

        Args:
            key_str: Wire key.
            buf: Writable buffer to fill.

        Returns:
            The buffer length on a hit, or :data:`GET_MISS` if the key is
            absent or its value is not exactly the buffer's size.

        Raises:
            RequestError: If the server refuses the load.
        """
        dst = buf if isinstance(buf, memoryview) else memoryview(buf)
        if dst.format != "B":
            dst = dst.cast("B")
        size = dst.nbytes
        reply = self._get_client().custom_command([_GET, key_str.encode()])
        if reply is None:
            return GET_MISS
        if len(reply) != size:
            logger.warning(
                "Valkey BLOB.TCP_GET size mismatch for key %s: read %d bytes, "
                "expected %d; treating as miss.",
                key_str,
                len(reply),
                size,
            )
            return GET_MISS
        dst[:] = reply
        return size

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_module(client: Any) -> None:
        """Fail unless the server can serve the module's commands.

        ``BLOB.INFO`` runs the module's enabled check before reading the key,
        so one read distinguishes a missing module from a disabled one.

        Args:
            client: A freshly built glide client.

        Raises:
            RuntimeError: If the module is not loaded or is disabled.
            RequestError: Any other error from the server, unchanged.
        """
        # Third Party
        import glide_sync  # type: ignore[import-untyped]

        try:
            client.custom_command([_INFO, PROBE_KEY])
        except glide_sync.RequestError as exc:
            text = str(exc)
            if "unknown command" in text:
                raise RuntimeError(
                    "transport='blob-tcp' needs the valkey-large-object module, "
                    "but the server does not know BLOB.INFO. Load it with "
                    "--loadmodule <path>/libvalkey_large_object.so."
                ) from exc
            if text.startswith("DISABLED"):
                raise RuntimeError(
                    "transport='blob-tcp' needs the valkey-large-object module "
                    "enabled, but the server reports it disabled. Set "
                    "largeobj.enabled yes."
                ) from exc
            raise
