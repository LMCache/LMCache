# SPDX-License-Identifier: Apache-2.0
"""
Context for LMCache SDK operations.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Mapping, Sequence
from typing import Any
import time
import uuid

# Third Party
import requests
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.sdk.cache_kind import LMCacheSDKCacheKind
from lmcache.sdk.wrapper.ring_mapping import RingMapping, RingStateError
from lmcache.sdk.wrapper.ring_transfer import (
    RingSDKTransferWrapper,
    SDKObjectLayout,
)
from lmcache.v1.client.transport.factory import RequestClientFactory
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.transport.base import RequestClient

logger = init_logger(__name__)


class LMCacheSDKError(RuntimeError):
    """Raised when an SDK KV-cache operation fails."""


ModifyFnType = Callable[
    [Mapping[LMCacheSDKCacheKind, torch.Tensor], Sequence[int]],
    tuple[torch.Tensor, Sequence[int]],
]


class LMCacheSDKContext:
    """
    Retrieve and store KV cache tensors through an LMCache MP request client.

    The model layout must already be registered in the running LMCache server
    by an inference engine (a vLLM instance with query transfer enabled). The
    SDK allocates and registers nothing: the server copies cached chunks' L1
    objects, as raw bytes, into free blocks of the engine's Q ring over device
    IPC (the LMCache-driven transfer path), and the SDK decodes them into
    contiguous CPU tensors (see ``sdk.wrapper.ring_transfer``). vLLM must
    be idle (not generating any query tensors) when SDK transfers.
    """

    def __init__(
        self,
        url: str,
        http_url: str,
        model_name: str,
        kind: LMCacheSDKCacheKind = LMCacheSDKCacheKind.KV,
        timeout: float = 60.0,
    ) -> None:
        """
        Initialize the SDK context.

        Args:
            url: Multiprocess request endpoint. The URL scheme selects the
                transport.
            http_url: HTTP endpoint URL for fetching information.
            model_name: Model name used by the running LMCache server instance.
            kind: The type of cache.
            timeout: Timeout in seconds for blocking MQ calls. Defaults to 60.

        Returns:
            LMCacheSDKContext instance.

        Raises:
            LMCacheSDKError: If the server cannot be reached.
        """
        self._transfer_ctx: RingSDKTransferWrapper | None = None
        self._kind = kind
        self._zmq_context = zmq.Context()
        self._req_client: RequestClient = RequestClientFactory.create(
            url,
            context=self._zmq_context,
        )
        self._mq_timeout = timeout
        self._model_name = kind.server_model_name(model_name)
        self.instance_id = uuid.uuid4().int & ((1 << 63) - 1)
        self._http_url = http_url

        mp_conf = {}
        try:
            response = requests.get(f"{self._http_url}/config", timeout=timeout)
            response.raise_for_status()
            mp_conf = response.json()["mp"]
        except (requests.RequestException, KeyError, ValueError) as err:
            raise LMCacheSDKError(
                f"failed to fetch server config from {self._http_url}/config"
            ) from err
        self._chunk_size: int = int(mp_conf["chunk_size"])

        # Engine registrations of this kind's layout, keyed by instance ID, and
        # the L1 object layouts the server stores them with.
        self._engine_meta_conf: dict[str, Any] = {}
        self._object_layouts: list[dict[str, Any]] = []
        # L1 read-lock TTL in seconds, from /status (the server default if absent).
        self._read_ttl = 300
        try:
            response = requests.get(f"{self._http_url}/status", timeout=timeout)
            response.raise_for_status()
            status = response.json()
            self._engine_meta_conf = status.get(kind.status_meta_field(), {})
            self._object_layouts = status.get("object_layouts", [])
            l1s = status.get("storage_manager", {}).get("l1_managers", {})
            self._read_ttl = min(
                (
                    int(l1["read_ttl_seconds"])
                    for l1 in l1s.values()
                    if "read_ttl_seconds" in l1
                ),
                default=300,
            )
        except (requests.RequestException, KeyError, ValueError) as err:
            raise LMCacheSDKError(
                f"failed to fetch server config from {self._http_url}/status"
            ) from err

        self._pending_lookups: set[str] = set()
        self._finished_lookups: dict[str, int] = {}

        logger.info(
            "Initialized LMCacheSDKContext with instance_id=%s, model_name=%s, "
            "chunk_size=%s, kind=%s",
            self.instance_id,
            self._model_name,
            self._chunk_size,
            self._kind,
        )

    def set_max_transfers(self, max_transfers: int) -> None:
        """Set how many transfers the process runs at once (default 2).

        Shared by both Q and KV context of the process. Many concurrent
        transfers split the host's memory bandwidth and slows transfer down.
        Tune this by the total batch modify time.

        Args:
            max_transfers: The maximum number of concurrent transfers (min 1).

        Raises:
            ValueError: If max_transfers is less than 1.
        """
        RingMapping.set_max_transfers(max_transfers)

    @property
    def kind(self) -> LMCacheSDKCacheKind:
        """Return type of tensor operation the context serves."""
        return self._kind

    def register_caches(
        self,
    ) -> None:
        """Connect to the engine's Q ring for this kind's transfers.

        Reads the kind's object layout and transfers its L1 objects through
        the ring as raw bytes.

        Raises:
            LMCacheSDKError: If no engine registered this kind's layout for
                the model, the layout is unsupported, or no Q ring is
                registered.
        """
        entry = next(
            (
                e
                for e in self._engine_meta_conf.values()
                if e.get("model_name") == self._model_name
            ),
            None,
        )
        if entry is None:
            raise LMCacheSDKError(
                f"no engine registered a {self._kind.name} layout for "
                f"model_name={self._model_name!r}; start the inference engine "
                "with query transfer enabled first."
            )
        try:
            self._world_size = int(entry.get("world_size", 1))
            layout = entry.get(self._kind.status_layout_field(), {})
            if not layout:
                raise LMCacheSDKError(
                    f"no registered {self._kind.name} layout for {self._model_name!r}."
                )
            kernel_groups = layout.get("kernel_groups", [])
            if len(kernel_groups) != 1:
                raise LMCacheSDKError(
                    f"the {self._kind.name} layout of {self._model_name!r} "
                    f"reports {len(kernel_groups)} kernel groups; SDK mode "
                    "supports dense models (one kernel group)"
                )
            self._connect_sdk(kernel_groups[0])
        except LMCacheSDKError:
            raise
        except Exception as err:
            raise LMCacheSDKError(
                f"failed to decode the registered layout for "
                f"model_name={self._model_name!r}"
            ) from err

    def _connect_sdk(self, kernel_group: Mapping[str, Any]) -> None:
        """Set up the transfer context to transfer this kind through vLLM's Q
        ring as raw L1 objects.

        KV objects follow the engine's KV registration and can be stored. The
        query ring's objects follow its own registration and are read-only.
        The object layout comes from qstore's ``object_layouts`` report (the
        server's layout registry).

        Args:
            kernel_group: The registration's only kernel group, from
                ``/status``.

        Raises:
            LMCacheSDKError: If the server does not report the model's object
                layout, the layout is not a single full-attention object, is
                compressed or not chunk-aligned, KV runs with TP > 1, or no Q
                ring is registered.
        """
        is_kv = self._kind is LMCacheSDKCacheKind.KV
        object_groups = next(
            (
                entry["object_groups"]
                for entry in self._object_layouts
                if entry.get("model_name") == self._model_name
                and int(entry.get("world_size", 0)) == self._world_size
            ),
            None,
        )
        if not object_groups:
            raise LMCacheSDKError(
                "the server reports no object layout for "
                f"{self._model_name!r}; SDK mode needs the qstore module's "
                "object_layouts in /status"
            )
        tokens_per_block = int(kernel_group["tokens_per_block"])
        if int(kernel_group["slots_per_block"]) != tokens_per_block:
            raise LMCacheSDKError(
                "the kernel group is compressed (slots_per_block != "
                "tokens_per_block), which SDK mode does not decode"
            )
        if self._chunk_size % tokens_per_block:
            raise LMCacheSDKError(
                f"chunk_size {self._chunk_size} is not a multiple of "
                f"tokens_per_block {tokens_per_block}"
            )
        if self._world_size != 1:
            raise LMCacheSDKError(
                f"SDK mode supports TP 1 only, the engine runs "
                f"world_size={self._world_size}"
            )
        if len(object_groups) != 1:
            raise LMCacheSDKError(
                f"{self._model_name!r} has {len(object_groups)} object groups; "
                "SDK mode supports dense models (one object group)"
            )
        (og,) = object_groups
        window = int(og["num_chunks_in_sw"])
        if og["kind"] != "attention" or window >= 0:
            raise LMCacheSDKError(
                f"object group {og['object_group_idx']} ({og['kind']}, {window} "
                "chunks in window) cannot be served in SDK mode, which "
                "supports full attention only"
            )
        if len(og["chunk_object_shapes"]) != 1:
            raise LMCacheSDKError(
                f"object group {og['object_group_idx']} packs "
                f"{len(og['chunk_object_shapes'])} tensors; expected one"
            )
        layout = SDKObjectLayout(
            object_group_idx=int(og["object_group_idx"]),
            shape=tuple(int(d) for d in og["chunk_object_shapes"][0]),
            dtype=getattr(
                torch, str(og["chunk_object_dtypes"][0]).replace("torch.", "")
            ),
        )
        ring = RingMapping(
            self._req_client,
            LMCacheSDKCacheKind.QUERY.server_model_name(
                self._kind.base_model_name(self._model_name)
            ),
            self._world_size,
            self._mq_timeout,
        )
        try:
            ring.connect()
        except RingStateError as err:
            raise LMCacheSDKError(str(err)) from err
        self._transfer_ctx = RingSDKTransferWrapper(
            ring,
            layout,
            self._chunk_size,
            self._mq_timeout,
            writable=is_kv,
        )
        logger.info(
            "Transferring %s through vLLM's Q ring on %s: %d bytes per chunk object",
            self._model_name,
            ring.device,
            layout.num_bytes,
        )

    @property
    def chunk_size(self) -> int:
        """Return the chunk size of the context."""
        return self._chunk_size

    @property
    def mq_timeout(self) -> float:
        """Return the message queue timeout of the context."""
        return self._mq_timeout

    @property
    def transfer_ctx(self) -> RingSDKTransferWrapper:
        """Return the transfer wrapper over vLLM's Q ring.

        Raises:
            LMCacheSDKError: If ``register_caches`` has not been called.
        """
        if self._transfer_ctx is None:
            raise LMCacheSDKError("register_caches() must be called first")
        return self._transfer_ctx

    def close(self) -> None:
        """Release the opened Q ring, if any, and close the request client."""
        try:
            if self._transfer_ctx is not None:
                self._transfer_ctx.close()
                self._transfer_ctx = None
        finally:
            self._req_client.close()

    def maybe_submit_lookup_request(
        self,
        request_id: str,
        token_ids: list[int],
        cache_salt: str = "",
        request_configs: dict[str, object] | None = None,
    ) -> None:
        """Submit a LOOKUP request for the given token IDs.
        Modification from lmcache/integration/vllm/vllm_multi_process_adapter.py.
        Need duplicate since SDK has TransferContext, not Adapter, but still need
        lookup and end_session functionality.

        Args:
            request_id: Unique ID for this lookup request.
            token_ids: List of token IDs to look up.
            cache_salt: Optional cache salt string for the lookup.
            request_configs: Optional LMCache request configs to include in
                the IPC key.
        """
        if request_id in self._pending_lookups:
            # Skip if there is already a lookup request
            return

        aligned_end = (len(token_ids) // self._chunk_size) * self._chunk_size

        key = self._create_key(
            token_ids,
            start=0,
            end=aligned_end,
            request_id=request_id,
            cache_salt=cache_salt,
            request_configs=request_configs,
        ).no_worker_id_version()

        future = self._req_client.lookup(key, self._world_size)
        try:
            future.result(timeout=self._mq_timeout)
        except TimeoutError:
            logger.warning(
                "LOOKUP request timed out after %ss.",
                self._mq_timeout,
            )
            return
        self._pending_lookups.add(request_id)

    def check_lookup_result(self, request_id: str) -> int | None:
        """Check the result of a LOOKUP request.
        Modification from lmcache/integration/vllm/vllm_multi_process_adapter.py.
        Need duplicate since SDK has TransferContext, not Adapter, but still need
        lookup and end_session functionality.

        Args:
            request_id: The request ID of the LOOKUP to check.

        Returns:
            The number of prefetched tokens if the LOOKUP is finished,
            0 if not finished, or None if the request ID is not found.
        """
        if request_id not in self._pending_lookups:
            # No job — either unhealthy at submit time or already cleaned up.
            # If we have a cached result, return it to handle repeated calls.
            return self._finished_lookups.get(request_id, 0)

        if request_id in self._finished_lookups:
            # Return cached result if the job is already finished
            return self._finished_lookups[request_id]

        try:
            result = self._req_client.query_prefetch_status(request_id).result(
                timeout=self._mq_timeout
            )
        except TimeoutError:
            logger.warning(
                "QUERY_PREFETCH_STATUS timed out after %ss.",
                self._mq_timeout,
            )
            return 0

        if result is None:
            return None

        token_count = int(result)
        self._finished_lookups[request_id] = token_count
        return token_count

    def wait_for_lookup(
        self,
        request_id: str,
        tokens: Sequence[int],
        cache_salt: str = "",
        request_configs: dict[str, object] | None = None,
        total_tokens: int | None = None,
    ) -> int:
        self.maybe_submit_lookup_request(
            request_id,
            token_ids=list(tokens[:total_tokens]),
            cache_salt=cache_salt,
            request_configs=request_configs,
        )

        start_time = time.time()
        num_prefetched_tokens = self.check_lookup_result(request_id)
        while num_prefetched_tokens is None:
            if time.time() - start_time > self.mq_timeout:
                # Put end_session here because wait_for_lookup can also hit
                # mq_timeout. Too much try/finally in retrieve().
                self.end_session(request_id)
                raise LMCacheSDKError(
                    f"LOOKUP request timed out after {self.mq_timeout}s "
                    f"for request_id={request_id}"
                )
            logger.info(
                "Waiting for LOOKUP result for request_id=%s...",
                request_id,
            )
            time.sleep(0.01)
            num_prefetched_tokens = self.check_lookup_result(request_id)

        return num_prefetched_tokens

    def end_session(self, request_id: str, block_ids: list[int] | None = None) -> None:
        """End a session and clean up associated resources on the server.

        Args:
            request_id: The request ID of the session to end.
            block_ids: Optional list of block IDs to free.
        """
        self._pending_lookups.discard(request_id)
        self._finished_lookups.pop(request_id, None)
        try:
            self._req_client.end_session(request_id).result(timeout=self._mq_timeout)
        except TimeoutError:
            logger.warning(
                "END_SESSION timed out after %ss for request_id=%s.",
                self._mq_timeout,
                request_id,
            )

    def retrieve(
        self,
        tokens: Sequence[int],
        cache_salt: str = "",
        request_configs: dict[str, object] | None = None,
    ) -> torch.Tensor | None:
        """Retrieve KV cache tensors for the given token IDs.

        Args:
            tokens: The list of token IDs to retrieve KV cache for.
            cache_salt: Optional cache salt string for the lookup.
            request_configs: Optional LMCache request configs to include in
                the IPC key.

        Returns:
            A contiguous CPU tensor containing the retrieved KV cache for
            the requested tokens.
            None if retrieval fails or there are no tokens to retrieve.

        Raises:
            LMCacheSDKError: If the LOOKUP request does not complete in time,
                or the Q ring cannot serve the transfer.
        """
        if not tokens:
            logger.info("No tokens provided for retrieval; returning None.")
            return None

        # Drop tokens not fit into a whole chunk
        total_tokens = (len(tokens) // self.chunk_size) * self.chunk_size
        if total_tokens == 0:
            logger.info(
                "Number of tokens (%d) is less than chunk size (%d); returning None.",
                len(tokens),
                self.chunk_size,
            )
            return None

        # Phase 0: Trigger lookup (all requests lookup at the same time)
        warmup_id = f"retrieve-warmup-{uuid.uuid4().hex}"
        warm = self.wait_for_lookup(
            warmup_id,
            tokens,
            cache_salt,
            request_configs,
            total_tokens,
        )
        warm_at = time.monotonic()

        if warm <= 0:
            logger.info(
                "No prefetched tokens found for request_id=%s; returning None.",
                warmup_id,
            )
            self.end_session(warmup_id)
            return None

        # Phase 1: retrieve the cached prefix as one contiguous tensor.
        # Before the actual transfer happen, because this needs holding the
        # transfer lock, the initial lookup could already be expired (default
        # TTL is 300 seconds). So trigger lookup again.
        # Only run max_transfers at once (self.transfer_ctx.transfer_slot).
        request_id = f"retrieve-{uuid.uuid4().hex}"
        with self.transfer_ctx.transfer_slot:
            is_ttl_not_expired = time.monotonic() - warm_at < self._read_ttl - 30
            try:
                # Lookup again
                num_prefetched_tokens = self.wait_for_lookup(
                    request_id, tokens, cache_salt, request_configs, total_tokens
                )
                if num_prefetched_tokens <= 0:
                    logger.info(
                        "No prefetched tokens found for request_id=%s; returning None.",
                        request_id,
                    )
                    self.end_session(request_id)
                    return None

                # Now transfer
                key = self._create_key(
                    token_ids=list(tokens[:num_prefetched_tokens]),
                    start=0,
                    end=num_prefetched_tokens,
                    request_id=request_id,
                    cache_salt=cache_salt,
                    request_configs=request_configs,
                    worker_id=0,
                )
                try:
                    return self.transfer_ctx.retrieve(key, self.instance_id)
                except RingStateError as err:
                    raise LMCacheSDKError(str(err)) from err
                finally:
                    self.end_session(request_id)
            finally:
                try:
                    # Phase 2: Release the lock if not yet expired
                    if is_ttl_not_expired:
                        lock_key = self._create_key(
                            list(tokens[:warm]),
                            start=0,
                            end=warm,
                            request_id=warmup_id,
                            cache_salt=cache_salt,
                            request_configs=request_configs,
                        ).no_worker_id_version()
                        try:
                            self._req_client.free_lookup_locks(
                                lock_key, self._world_size
                            ).result(timeout=self._mq_timeout)
                        except TimeoutError:
                            logger.warning(
                                "FREE_LOOKUP_LOCKS timed out for warm-up request_id=%s",
                                warmup_id,
                            )
                finally:
                    self.end_session(warmup_id)

    def store(
        self,
        kv: torch.Tensor,
        tokens: Sequence[int],
        cache_salt: str = "",
        request_configs: dict[str, object] | None = None,
    ) -> bool:
        """Store KV cache tensors for the given token IDs.

        Args:
            kv: The KV cache tensor to store, of shape [2, L, T, D].
            tokens: The list of token IDs corresponding to the KV cache tensor.
            cache_salt: Optional cache salt string for the store.
            request_configs: Optional LMCache request configs to include in
                the IPC key.

        Returns:
            True if the store operation is successful, False otherwise.

        Raises:
            LMCacheSDKError: If the tokens do not match ``kv``, or the Q ring
                cannot serve the transfer.
        """
        if len(tokens) != kv.shape[2]:
            raise LMCacheSDKError(
                f"Number of tokens ({len(tokens)}) does not match KV tensor's "
                f"token dimension ({kv.shape[2]})."
            )
        token_ids = list(tokens)
        total_tokens = (len(token_ids) // self.chunk_size) * self.chunk_size
        token_ids = token_ids[:total_tokens]
        kv_cpu = kv[:, :, :total_tokens, :].detach().cpu().contiguous()

        # Phase 0: assign request ID to this request
        request_id = f"store-{uuid.uuid4().hex}"
        key = self._create_key(
            token_ids=token_ids,
            start=0,
            end=total_tokens,
            request_id=request_id,
            cache_salt=cache_salt,
            request_configs=request_configs,
            worker_id=0,
        )

        # Phase 1: store the KV cache tensor
        try:
            return self.transfer_ctx.store(key, self.instance_id, kv_cpu)
        except RingStateError as err:
            raise LMCacheSDKError(str(err)) from err
        finally:
            self.end_session(request_id)

    # Helper functions
    def _create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
        request_configs: dict[str, object] | None = None,
        worker_id: int | None = None,
    ) -> IPCCacheServerKey:
        """Convert token IDs to an IPC cache engine key.

        Args:
            token_ids: The token IDs.
            start: Start token index.
            end: End token index.
            request_id: The request ID.
            cache_salt: Per-user isolation salt.
            request_configs: Optional LMCache request configs to include in
                the IPC key.
            worker_id: Optional worker ID for the key.
                If None, the key will be created without a worker ID (for lookups).

        Returns:
            IPCCacheServerKey: The constructed key.
        """
        return IPCCacheServerKey.from_token_ids(
            model_name=self._model_name,
            world_size=self._world_size,
            worker_id=worker_id,
            token_ids=token_ids,
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
            request_configs=request_configs,
        )
