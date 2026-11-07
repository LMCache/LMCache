# SPDX-License-Identifier: Apache-2.0
"""
autumn-rs L2 adapter config and factory.

autumn-rs is a distributed storage engine with an ordered, replicated KV
layer. Its Python client (``pip install autumn-rs``, imported as ``autumn``)
reads and writes straight into caller buffers with the GIL released, so the
connector here is pure Python: a thread pool drives ``autumn.BatchClient`` and
reports completions through one event fd, the contract
``NativeConnectorL2Adapter`` already bridges to MP mode.

Keys are stored as ``{namespace}/{h}/lmcache/{storage_key}``, where ``h`` is
the chunk hash's first hex digit: a namespace created with
``autumn-op namespace-create --name kvc --parts 16`` is cut on that digit, so
chunks spread over every partition from the start.
"""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Optional
import asyncio
import threading

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryResidency, MemorySegment, MemoryTags
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)

_TRANSPORTS = ("tcp", "ucx")


class AutumnL2AdapterConfig(L2AdapterConfigBase):
    """Config for an L2 adapter backed by an autumn-rs cluster.

    Fields:
    - manager: comma-separated ``ip:port`` list of autumn managers.
    - namespace: autumn namespace the keys live under (default ``kvc``).
    - credential_file: autumn principal credential file; required when the
      namespace is protected by authorization.
    - num_workers: batches queued or in flight (default 4).
    - transport: ``tcp`` (default) or ``ucx``.
    - direct: read large values straight from the extent nodes.
    - ttl_secs: expiry of each stored chunk, 0 = never (default 0).
    - max_capacity_gb: L2 capacity for usage tracking / eviction.
    """

    def __init__(
        self,
        manager: str,
        namespace: str = "kvc",
        credential_file: str = "",
        num_workers: int = 4,
        transport: str = "tcp",
        direct: bool = False,
        ttl_secs: int = 0,
        max_capacity_gb: float = 0,
    ):
        super().__init__()
        self.manager = manager
        self.namespace = namespace
        self.credential_file = credential_file
        self.num_workers = num_workers
        self.transport = transport
        self.direct = direct
        self.ttl_secs = ttl_secs
        self.max_capacity_gb = max_capacity_gb

    @classmethod
    def from_dict(cls, d: dict) -> "AutumnL2AdapterConfig":
        manager = d.get("manager")
        if not isinstance(manager, str) or not manager:
            raise ValueError("manager must be a non-empty string")

        namespace = d.get("namespace", "kvc")
        if not isinstance(namespace, str) or not namespace or "/" in namespace:
            raise ValueError("namespace must be a non-empty string without '/'")

        credential_file = d.get("credential_file", "")
        if not isinstance(credential_file, str):
            raise ValueError("credential_file must be a string")

        num_workers = d.get("num_workers", 4)
        if isinstance(num_workers, bool) or not isinstance(num_workers, int):
            raise ValueError("num_workers must be a positive integer")
        if num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")

        transport = d.get("transport", "tcp")
        if transport not in _TRANSPORTS:
            raise ValueError("transport must be one of %s" % (_TRANSPORTS,))

        direct = d.get("direct", False)
        if not isinstance(direct, bool):
            raise ValueError("direct must be a boolean")

        ttl_secs = d.get("ttl_secs", 0)
        if isinstance(ttl_secs, bool) or not isinstance(ttl_secs, int) or ttl_secs < 0:
            raise ValueError("ttl_secs must be a non-negative integer")

        max_capacity_gb = d.get("max_capacity_gb", 0)
        if not isinstance(max_capacity_gb, (int, float)) or max_capacity_gb < 0:
            raise ValueError("max_capacity_gb must be a non-negative number")

        return cls(
            manager=manager,
            namespace=namespace,
            credential_file=credential_file,
            num_workers=num_workers,
            transport=transport,
            direct=direct,
            ttl_secs=ttl_secs,
            max_capacity_gb=float(max_capacity_gb),
        )

    @classmethod
    def help(cls) -> str:
        return (
            "autumn L2 adapter config fields:\n"
            "- manager (str): comma-separated ip:port list of "
            "autumn managers (required)\n"
            "- namespace (str): autumn namespace for the keys "
            "(default 'kvc')\n"
            "- credential_file (str): autumn principal credential "
            "file, needed when the namespace is protected "
            "(default empty)\n"
            "- num_workers (int): batches queued or in flight "
            "(default 4, >0)\n"
            "- transport (str): 'tcp' or 'ucx' (default 'tcp')\n"
            "- direct (bool): read large values straight from "
            "the extent nodes (default false)\n"
            "- ttl_secs (int): expiry of each stored chunk, "
            "0 = never (default 0)\n"
            "- max_capacity_gb (float): max L2 capacity in GB for "
            "usage tracking / eviction (default 0 = disabled)"
        )


def read_credential_file(path: str) -> tuple[str, bytes]:
    """Read an autumn credential file into ``(principal, raw secret)``.

    Accepts the formats ``autumn-op principal-create`` writes: a
    ``credential: <hex>`` line with an optional ``principal: <name>`` line,
    or two bare lines ``<principal>`` and ``<hex>``.
    """
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    labeled = dict(
        (k.strip(), v.strip())
        for k, sep, v in (ln.partition(":") for ln in lines)
        if sep and k.strip() in ("principal", "credential")
    )
    if "credential" in labeled:
        principal, secret = labeled.get("principal", ""), labeled["credential"]
    elif len(lines) == 2:
        principal, secret = lines
    else:
        raise ValueError("unrecognized autumn credential file: %s" % path)
    if not principal:
        raise ValueError("autumn credential file names no principal: %s" % path)
    return principal, bytes.fromhex(secret)


def _bucketed_key(namespace_rel: str) -> bytes:
    """Prefix a storage key with its chunk hash's first hex digit.

    ``namespace_rel`` is ``encode_storage_key`` output,
    ``model@rank@group@hash[@salt[@tags]]``; the model name cannot contain
    ``@``, so the hash is always the fourth field.
    """
    fields = namespace_rel.split("@")
    chunk_hex = fields[3] if len(fields) > 3 else ""
    bucket = chunk_hex[:1] or "0"
    return ("%s/lmcache/%s" % (bucket, namespace_rel)).encode("utf-8")


def _as_bytes_view(mv: memoryview) -> memoryview:
    """Return ``mv`` as a plain ``'B'`` view of the same memory, no copy.

    ``MemoryObj.byte_array`` is a ctypes view with format ``'<B'``, which the
    client's buffer check (PyO3) refuses on little-endian hosts: it accepts
    only ``@``, ``=`` or ``>`` as a byte-order prefix.
    """
    return mv if mv.format == "B" else mv.cast("B")


class AutumnConnector:
    """Connector over ``autumn.BatchClient`` with the native-connector API.

    Each ``submit_batch_*`` returns a future id at once and hands the batch to
    a pool thread, which blocks in the client with the GIL released. The
    client runs gets and puts one batch at a time and existence probes beside
    them, so ``num_workers`` bounds the batches queued or in flight, not their
    parallelism. Completions are queued and signalled through ``event_fd``.

    Thread safety: every public method may be called from any thread.
    """

    def __init__(
        self,
        autumn_module: Any,
        manager: str,
        namespace: str,
        principal: Optional[str] = None,
        credential: Optional[bytes] = None,
        num_workers: int = 4,
        direct: bool = False,
        ttl_secs: int = 0,
    ) -> None:
        """Connect the batch client and start the worker pool.

        Args:
            autumn_module: The imported ``autumn`` client module.
            manager: Comma-separated ``ip:port`` list of autumn managers.
            namespace: Namespace the client is scoped to; it prepends
                ``{namespace}/`` to every key.
            principal: Owner of ``credential``; given with it or not at all.
            credential: Raw credential bytes, for a protected namespace.
            num_workers: Pool threads, the batches queued or in flight.
            direct: Read large values straight from the storage nodes.
            ttl_secs: Expiry of each stored value, 0 = never.

        Raises:
            RuntimeError: The client could not connect.
        """
        auth: dict[str, Any] = {"scope": namespace}
        if credential is not None:
            auth["principal"] = principal
            auth["credential"] = credential
        self._autumn = autumn_module
        self._manager = manager
        self._auth = auth
        self._ttl_secs = ttl_secs
        self._batch = autumn_module.BatchClient(manager, direct=direct, **auth)
        self._pool = ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="lmcache-autumn"
        )
        self._notifier = create_event_notifier()
        self._lock = threading.Lock()
        self._next_id = 1
        self._completions: list[tuple[int, bool, str, list[bool] | None]] = []
        self._closed = False
        # Deletes go through the async client on a loop of its own. The client
        # connects on the first delete; the task is created on the loop thread,
        # so concurrent deletes share one connection.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="lmcache-autumn-delete", daemon=True
        )
        self._loop_thread.start()
        self._connect: Optional[asyncio.Future] = None

    def event_fd(self) -> int:
        """Return the fd that becomes readable when completions are queued."""
        return self._notifier.fileno()

    def submit_batch_get(self, keys: list[str], memoryviews: list) -> int:
        """Read each key's value into its buffer.

        Args:
            keys: Storage keys.
            memoryviews: Writable buffers, one per key; a value is loaded
                only when its stored size equals its buffer's.

        Returns:
            The future id; its completion carries one bool per key.
        """
        akeys = [_bucketed_key(k) for k in keys]
        bufs = [_as_bytes_view(mv) for mv in memoryviews]
        return self._submit(lambda: self._batch.get_into(akeys, bufs))

    def submit_batch_set(self, keys: list[str], memoryviews: list) -> int:
        """Write each buffer under its key.

        Args:
            keys: Storage keys.
            memoryviews: Buffers, one per key, unmodified until completion.

        Returns:
            The future id; its completion is ok only when every write was.
        """
        akeys = [_bucketed_key(k) for k in keys]
        bufs = [_as_bytes_view(mv) for mv in memoryviews]
        ttl = self._ttl_secs
        return self._submit(
            lambda: self._batch.put_from(akeys, bufs, ttl), all_required=True
        )

    def submit_batch_exists(self, keys: list[str]) -> int:
        """Probe the keys without reading their values.

        Returns:
            The future id; its completion carries one bool per key.
        """
        akeys = [_bucketed_key(k) for k in keys]
        return self._submit(lambda: self._batch.exists(akeys))

    def submit_batch_delete(self, keys: list[str]) -> int:
        """Delete the keys; a key that is already gone counts as deleted.

        Returns:
            The future id; its completion carries one bool per key.
        """
        akeys = [_bucketed_key(k) for k in keys]
        return self._submit(lambda: self._delete(akeys))

    def drain_completions(self) -> list[tuple[int, bool, str, list[bool] | None]]:
        """Return and forget the queued completions.

        Returns:
            ``(future_id, ok, error, per_key_results)`` tuples.
        """
        self._notifier.consume()
        with self._lock:
            out, self._completions = self._completions, []
        return out

    def close(self) -> None:
        """Finish the batches in flight, drop queued ones, release everything.

        Idempotent. A batch in flight is waited for: the client returns only
        once it no longer reads the caller's buffers.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._pool.shutdown(wait=True, cancel_futures=True)
            if self._connect is not None:
                asyncio.run_coroutine_threadsafe(
                    self._close_client(), self._loop
                ).result()
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join()
            self._loop.close()
            self._batch = None
            self._notifier.close()

    def _submit(self, run, all_required: bool = False) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("autumn connector is closed")
            fid = self._next_id
            self._next_id += 1
            self._pool.submit(self._complete, fid, run, all_required)
        return fid

    def _complete(self, fid: int, run, all_required: bool) -> None:
        try:
            results = [bool(r) for r in run()]
            # A set is one result for the batch; the other ops report per key.
            ok = all(results) if all_required else True
            completion = (fid, ok, "", None if all_required else results)
        except Exception as e:
            logger.warning("autumn L2 batch %d failed: %s", fid, e)
            completion = (fid, False, str(e), None)
        with self._lock:
            self._completions.append(completion)
        self._notifier.notify()

    async def _client(self) -> Any:
        if self._connect is None:
            self._connect = asyncio.ensure_future(
                self._autumn.Client.connect(self._manager, **self._auth)
            )
        return await self._connect

    async def _close_client(self) -> None:
        assert self._connect is not None
        if self._connect.done() and self._connect.exception() is None:
            await self._connect.result().close()

    def _delete(self, keys: list[bytes]) -> list[bool]:
        async def run() -> list[bool]:
            client = await self._client()
            results = await asyncio.gather(
                *(client.delete(k) for k in keys), return_exceptions=True
            )
            return [not isinstance(r, BaseException) for r in results]

        return asyncio.run_coroutine_threadsafe(run(), self._loop).result()


def _create_autumn_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create a NativeConnectorL2Adapter backed by an autumn-rs cluster."""
    try:
        # Third Party
        import autumn
    except ImportError as e:
        raise RuntimeError(
            "autumn L2 adapter requires the autumn-rs client: pip install autumn-rs"
        ) from e

    # Lazy import to avoid circular dependency
    # First Party
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (  # noqa: E501
        NativeConnectorL2Adapter,
    )

    assert isinstance(config, AutumnL2AdapterConfig)

    principal, credential = None, None
    if config.credential_file:
        principal, credential = read_credential_file(config.credential_file)
    if config.transport != "tcp":
        autumn.set_transport(config.transport)

    connector = AutumnConnector(
        autumn,
        config.manager,
        config.namespace,
        principal=principal,
        credential=credential,
        num_workers=config.num_workers,
        direct=config.direct,
        ttl_secs=config.ttl_secs,
    )
    logger.info(
        "Created autumn L2 adapter: manager=%s namespace=%s transport=%s (workers=%d)",
        config.manager,
        config.namespace,
        config.transport,
        config.num_workers,
    )
    return NativeConnectorL2Adapter(
        connector,
        MemoryTags(MemorySegment.STORAGE, MemoryResidency.REMOTE),
        max_capacity_gb=config.max_capacity_gb,
        type_name="autumn",
        extra_status={"manager": config.manager, "namespace": config.namespace},
    )


# Self-register config type and adapter factory
register_l2_adapter_type("autumn", AutumnL2AdapterConfig)
register_l2_adapter_factory("autumn", _create_autumn_l2_adapter)
