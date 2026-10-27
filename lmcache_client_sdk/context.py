# SPDX-License-Identifier: Apache-2.0
"""Standalone client context for inference engines.

The context connects to a running LMCache server, registers the client's
engine-driven transfer geometry, and exposes the store/retrieve operations an
inference engine needs on its runtime path. It intentionally excludes the
offline token-dropping API (``LMCacheRequestStream`` / ``LMCacheBatchedStream``)
and any cache-dropping (block release) operations: this package is the
lightweight, frozen engine-facing client contract.
"""

# Standard
from collections.abc import Mapping, Sequence
import time
import uuid

# Third Party
import torch

# First Party
from lmcache_client_sdk.key import IPCCacheServerKey
from lmcache_client_sdk.logging import init_logger
from lmcache_client_sdk.protocol import RegisterEngineDrivenContextPayload
from lmcache_client_sdk.server_config import (
    LMCacheClientError,
    fetch_cache_layout,
    fetch_server_config,
)
from lmcache_client_sdk.tensor_wire import decode_tensors, encode_tensors
from lmcache_client_sdk.transport import ZmqSyncClient

logger = init_logger(__name__)


class LMCacheClientContext:
    """Store and retrieve KV cache tensors through an LMCache server.

    The model layout must already be registered in the running LMCache server
    (e.g. by a vLLM instance that called REGISTER_KV_CACHE); the context reads
    the geometry from the server's HTTP status endpoint.

    Args:
        url: Multiprocess request endpoint. The URL scheme selects the
            transport (``tcp://`` for ZMQ).
        http_url: HTTP endpoint URL for fetching server configuration.
        model_name: Model name used by the running LMCache server instance.
        timeout: Timeout in seconds for blocking MQ calls. Defaults to 60.

    Raises:
        LMCacheClientError: If server configuration or layout discovery fails.
    """

    def __init__(
        self,
        url: str,
        http_url: str,
        model_name: str,
        timeout: float = 60.0,
    ) -> None:
        self._url = url
        self._http_url = http_url
        self._model_name = model_name
        self._mq_timeout = timeout

        server_config = fetch_server_config(http_url, timeout)
        layout = fetch_cache_layout(http_url, model_name, timeout)

        self._chunk_size = server_config.chunk_size
        self._world_size = layout.world_size
        self._num_kv_readers = layout.num_kv_readers
        self._num_layers = layout.num_layers
        self._block_size = layout.block_size
        self._hidden_dim_size = layout.hidden_dim_size
        self._use_mla = layout.use_mla
        self._chunk_dtype = getattr(torch, layout.dtype_str)
        if self._use_mla:
            self._chunk_shape = torch.Size(
                (layout.num_layers, self._chunk_size, layout.hidden_dim_size)
            )
        else:
            self._chunk_shape = torch.Size(
                (2, layout.num_layers, self._chunk_size, layout.hidden_dim_size)
            )

        self.instance_id = uuid.uuid4().int & ((1 << 63) - 1)
        self.blocks_in_chunk = self._chunk_size // self._block_size

        self._req_client = ZmqSyncClient(url, timeout)
        try:
            response = self._req_client.register_kv_cache_engine_driven_context(
                RegisterEngineDrivenContextPayload(
                    instance_id=self.instance_id,
                    model_name=model_name,
                    world_size=self._world_size,
                    block_size=self._block_size,
                    num_layers=self._num_layers,
                    hidden_dim_size=self._hidden_dim_size,
                    dtype_str=layout.dtype_str,
                    use_mla=self._use_mla,
                    num_physical_slots=None,
                    group_layouts=[],
                )
            )
        except Exception as exc:
            self._req_client.close()
            raise LMCacheClientError(
                f"failed to register engine-driven context for model "
                f"{model_name!r}"
            ) from exc
        self._null_block_id = response.null_block_id

        self._pending_lookups: set[str] = set()
        self._finished_lookups: dict[str, int] = {}

        logger.info(
            "Initialized LMCacheClientContext with instance_id=%d, "
            "model_name=%s, chunk_size=%d, world_size=%d, use_mla=%s",
            self.instance_id,
            self._model_name,
            self._chunk_size,
            self._world_size,
            self._use_mla,
        )

    @property
    def chunk_size(self) -> int:
        """Return the chunk size (tokens per LMCache chunk)."""
        return self._chunk_size

    @property
    def mq_timeout(self) -> float:
        """Return the message queue timeout of the context."""
        return self._mq_timeout

    def close(self) -> None:
        """Close the request client and release its transport resources."""
        try:
            self._req_client.unregister_kv_cache_engine_driven_context(
                self.instance_id
            )
        except Exception:
            logger.warning("failed to unregister engine-driven context")
        finally:
            self._req_client.close()

    def store(
        self,
        kv: torch.Tensor,
        tokens: Sequence[int],
        cache_salt: str = "",
        request_configs: dict[str, object] | None = None,
    ) -> bool:
        """Store KV cache tensors for the given token ids.

        Args:
            kv: The KV cache tensor to store, of shape ``[2, L, T, D]``
                (split K/V) or ``[L, T, D]`` (single-plane / MLA).
            tokens: The list of token ids corresponding to the KV tensor.
            cache_salt: Optional cache salt string for the store.
            request_configs: Optional LMCache request configs.

        Returns:
            ``True`` if the store operation is successful, ``False``
            otherwise (timeout or server rejection).

        Raises:
            LMCacheClientError: If the token count does not match the KV
                tensor's token dimension.
        """
        if len(tokens) != kv.shape[-2]:
            raise LMCacheClientError(
                f"Number of tokens ({len(tokens)}) does not match KV tensor's "
                f"token dimension ({kv.shape[-2]})."
            )
        token_ids = list(tokens)
        total_tokens = (len(token_ids) // self._chunk_size) * self._chunk_size
        token_ids = token_ids[:total_tokens]
        kv_cpu = kv[..., :total_tokens, :].detach().cpu().contiguous()

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
        try:
            self._req_client.prepare_store(key, self.instance_id)
            num_chunks = total_tokens // self._chunk_size
            chunks = [
                kv_cpu[
                    ...,
                    i * self._chunk_size : (i + 1) * self._chunk_size,
                    :,
                ].contiguous()
                for i in range(num_chunks)
            ]
            data = encode_tensors(chunks)
            return bool(self._req_client.commit_store(key, self.instance_id, data))
        except TimeoutError:
            logger.warning("store timed out for request_id=%s", request_id)
            return False
        finally:
            self._end_session(request_id)

    def retrieve(
        self,
        tokens: Sequence[int],
        cache_salt: str = "",
        request_configs: dict[str, object] | None = None,
    ) -> torch.Tensor | None:
        """Retrieve KV cache tensors for the given token ids.

        Args:
            tokens: The list of token ids to retrieve KV cache for.
            cache_salt: Optional cache salt string for the lookup.
            request_configs: Optional LMCache request configs.

        Returns:
            A contiguous CPU tensor containing the retrieved KV cache for
            the requested tokens, or ``None`` on a miss.
        """
        if not tokens:
            logger.info("no tokens provided for retrieval; returning None")
            return None

        total_tokens = (len(tokens) // self._chunk_size) * self._chunk_size
        if total_tokens == 0:
            logger.info(
                "number of tokens (%d) is less than chunk size (%d); "
                "returning None",
                len(tokens),
                self._chunk_size,
            )
            return None

        request_id = f"retrieve-{uuid.uuid4().hex}"
        self._maybe_submit_lookup_request(
            request_id,
            token_ids=list(tokens[:total_tokens]),
            cache_salt=cache_salt,
            request_configs=request_configs,
        )

        start_time = time.time()
        num_prefetched_tokens = self._check_lookup_result(request_id)
        while num_prefetched_tokens is None:
            if time.time() - start_time > self._mq_timeout:
                raise LMCacheClientError(
                    f"LOOKUP request timed out after {self._mq_timeout}s "
                    f"for request_id={request_id}"
                )
            time.sleep(0.01)
            num_prefetched_tokens = self._check_lookup_result(request_id)

        if num_prefetched_tokens <= 0:
            logger.info(
                "no prefetched tokens found for request_id=%s; returning None",
                request_id,
            )
            self._end_session(request_id)
            return None

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
            response = self._req_client.prepare_retrieve(key, self.instance_id)
            if not response.success or not response.data:
                return None
            expected_count = (key.end - key.start) // self._chunk_size
            chunks = decode_tensors(
                response.data,
                expected_shape=self._chunk_shape,
                expected_dtype=self._chunk_dtype,
                expected_count=expected_count,
            )
            token_dim = chunks[0].ndim - 2
            return torch.cat(chunks, dim=token_dim)
        finally:
            try:
                self._req_client.commit_retrieve(key, self.instance_id)
            except TimeoutError:
                logger.warning(
                    "commit_retrieve timed out for request_id=%s", request_id
                )
            self._end_session(request_id)

    # ------------------------------------------------------------------
    # Session helpers (private by design; not part of the public contract)
    # ------------------------------------------------------------------

    def _maybe_submit_lookup_request(
        self,
        request_id: str,
        token_ids: list[int],
        cache_salt: str = "",
        request_configs: dict[str, object] | None = None,
    ) -> None:
        """Submit a LOOKUP request for the given token ids."""
        if request_id in self._pending_lookups:
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
        try:
            self._req_client.lookup(key, self._world_size)
        except TimeoutError:
            logger.warning("LOOKUP request timed out after %ss", self._mq_timeout)
            return
        self._pending_lookups.add(request_id)

    def _check_lookup_result(self, request_id: str) -> int | None:
        """Check the result of a LOOKUP request.

        Returns:
            The number of prefetched tokens if the LOOKUP is finished,
            ``0`` if not finished, or ``None`` if the server has no answer yet.
        """
        if request_id not in self._pending_lookups:
            return self._finished_lookups.get(request_id, 0)
        if request_id in self._finished_lookups:
            return self._finished_lookups[request_id]
        try:
            result = self._req_client.query_prefetch_status(request_id)
        except TimeoutError:
            logger.warning(
                "QUERY_PREFETCH_STATUS timed out after %ss", self._mq_timeout
            )
            return 0
        if result is None:
            return None
        token_count = result * self._chunk_size
        self._finished_lookups[request_id] = token_count
        return token_count

    def _end_session(self, request_id: str) -> None:
        """End a session and clean up server-side session state."""
        self._pending_lookups.discard(request_id)
        self._finished_lookups.pop(request_id, None)
        try:
            self._req_client.end_session(request_id)
        except TimeoutError:
            logger.warning(
                "END_SESSION timed out after %ss for request_id=%s",
                self._mq_timeout,
                request_id,
            )

    def _create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
        request_configs: Mapping[str, object] | None = None,
        worker_id: int | None = None,
    ) -> IPCCacheServerKey:
        """Convert token ids to an IPC cache engine key.

        Args:
            token_ids: The token ids.
            start: Start token index.
            end: End token index.
            request_id: The request ID.
            cache_salt: Per-user isolation salt.
            request_configs: Optional LMCache request configs.
            worker_id: Optional worker ID. ``None`` for lookups.

        Returns:
            The constructed key.
        """
        return IPCCacheServerKey.from_token_ids(
            model_name=self._model_name,
            world_size=self._world_size,
            num_kv_readers=self._num_kv_readers,
            worker_id=worker_id,
            token_ids=token_ids,
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
            request_configs=(
                dict(request_configs) if request_configs is not None else None
            ),
        )
