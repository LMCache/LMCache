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
    RecurrentState,
    RingSDKTransferWrapper,
    SDKObjectLayout,
)
from lmcache.v1.client.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.transport.base import RequestClient

logger = init_logger(__name__)


class LMCacheSDKError(RuntimeError):
    """Raised when an SDK KV-cache operation fails."""


def _model_kv_heads(hf_model_name: str, world_size: int) -> tuple[int, int]:
    """Return one worker's ``(kv_heads, head_dim)``, from the model's config."""
    # Third Party
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(hf_model_name)
    # Multimodal configs (e.g. Qwen3.5) nest the language model's.
    text_cfg = getattr(config, "text_config", None) or config
    num_heads = int(text_cfg.num_attention_heads)
    num_kv_heads = int(getattr(text_cfg, "num_key_value_heads", num_heads))
    head_dim = int(
        getattr(text_cfg, "head_dim", None) or int(text_cfg.hidden_size) // num_heads
    )
    return max(1, num_kv_heads // world_size), head_dim


def _attention_kernel_block_size(
    group: Mapping[str, Any],
    num_kv_heads: int,
    head_dim: int,
    mp_conf: Mapping[str, Any],
) -> int:
    """Return a hybrid model's attention kernel page size.

    vLLM may register each attention block as one synthetic head but page it
    at a smaller sub-page.

    Args:
        group: The attention kernel group from ``/status``.
        num_kv_heads: KV heads per worker.
        head_dim: Head size.
        mp_conf: The server's ``/config`` ``mp`` section, which may set
            ``sdk.kernel_block_size``.

    Returns:
        The kernel page size, ``tokens_per_block`` if not sub-paged.

    Raises:
        LMCacheSDKError: If the layout or heads are unexpected, or the pages
            are sub-paged without a valid ``sdk.kernel_block_size``.
    """
    tokens_per_block = int(group["tokens_per_block"])
    labels = str(group["engine_kv_shape"]).partition("[")[2].rstrip("] ").split(",")
    if [label.strip() for label in labels] != ["NB", "2", "BS", "NH", "HS"]:
        raise LMCacheSDKError(
            f"attention group {group['kernel_group_idx']} has format "
            f"{group['engine_kv_shape']}; the hybrid path supports "
            "NL x [NB, 2, BS, NH, HS] only"
        )
    dims = str(group["engine_kv_concrete_shape"]).partition("[")[2].rstrip("] ")
    heads, width = [int(size) for size in dims.split(",")][3:5]
    if (heads, width) == (num_kv_heads, head_dim):
        return tokens_per_block
    if heads != 1 or width != num_kv_heads * head_dim:
        raise LMCacheSDKError(
            f"attention group {group['kernel_group_idx']} registers {heads} heads "
            f"x {width}, matching neither the model's {num_kv_heads} x {head_dim} "
            f"nor a sub-paged 1 x {num_kv_heads * head_dim}"
        )
    plugin_config = mp_conf.get("runtime_plugin_config")
    extra = (
        plugin_config.get("extra_config") if isinstance(plugin_config, dict) else None
    )
    override = extra.get("sdk.kernel_block_size", 0) if isinstance(extra, dict) else 0
    if isinstance(override, bool) or not isinstance(override, int):
        raise LMCacheSDKError(
            f"server sdk.kernel_block_size={override!r} must be an integer"
        )
    if not override:
        raise LMCacheSDKError(
            "the attention pages are sub-paged. Set the server's "
            "--runtime-plugin-config '{\"sdk.kernel_block_size\": 32}'"
        )
    if override >= tokens_per_block or tokens_per_block % override:
        raise LMCacheSDKError(
            f"sdk.kernel_block_size={override} must be a proper divisor of the "
            f"sub-paged block size {tokens_per_block}"
        )
    return override


class TensorDict(dict[LMCacheSDKCacheKind, torch.Tensor]):
    """The cached tensors to be passed to modify functions, keyed by cache
    kind.

    Query tensors are only available for the tokens that just got computed
    in the step, so they cover ``[segment_start_token_id, T)`` of the KV's
    ``[0, T)``. With vLLM's ``lmcache.mp.q.sw_size_tokens`` set, they cover
    only the end of that range (see `LMCacheSDKContext.windowed_range`).

    Attributes:
        segment_start_token_id: First token the latest generate() pass
            computed (chunk-aligned); tokens before it came from the cache.
    """

    def __init__(
        self,
        tensors: Mapping[LMCacheSDKCacheKind, torch.Tensor],
        segment_start_token_id: int,
    ) -> None:
        super().__init__(tensors)
        self.segment_start_token_id = segment_start_token_id


ModifyFnType = Callable[
    [TensorDict, Sequence[int]],
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
        self._mp_conf = mp_conf
        # sw_size_tokens in chunks, or -1 for all chunks
        self._window_chunks = -1

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
            # Q only supports full attention, KV supports all types
            is_kv = self._kind is LMCacheSDKCacheKind.KV
            if not kernel_groups or (len(kernel_groups) != 1 and not is_kv):
                raise LMCacheSDKError(
                    f"the {self._kind.name} layout of {self._model_name!r} "
                    f"reports {len(kernel_groups)} kernel groups; SDK mode "
                    "supports dense models (one kernel group)"
                )
            self._connect_sdk(kernel_groups)
        except LMCacheSDKError:
            raise
        except Exception as err:
            raise LMCacheSDKError(
                f"failed to decode the registered layout for "
                f"model_name={self._model_name!r}"
            ) from err

    def _connect_sdk(self, kernel_groups: Sequence[Mapping[str, Any]]) -> None:
        """Set up the transfer context to transfer this kind through vLLM's Q
        ring as raw L1 objects.

        KV objects follow the engine's KV registration and can be stored. The
        query ring's objects follow its own registration and are read-only.
        The object layout comes from qstore's ``object_layouts`` report (the
        server's layout registry). A hybrid model's KV also has one recurrent
        object group (``--separate-object-groups``), last chunk holds the state
        at the end of a range, and the attention may be sub-paged (see
        `_attention_kernel_block_size`).

        Args:
            kernel_groups: The registration's kernel groups, from ``/status``.

        Raises:
            LMCacheSDKError: If the server does not report the model's object
                layout, the layout is not a single attention object (full
                attention for KV), is compressed or not chunk-aligned, the
                engine runs with TP > 1, or no Q ring is registered.
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
        for kernel_group in kernel_groups:
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
        # A hybrid model's KV has one more object group: its recurrent state.
        recurrent = (
            [g for g in object_groups if g["kind"] == "recurrent"]
            if is_kv and len(object_groups) == 2
            else []
        )
        others = [g for g in object_groups if g not in recurrent]
        if len(others) != 1:
            raise LMCacheSDKError(
                f"{self._model_name!r} has {len(object_groups)} object groups; "
                "SDK mode supports dense models (one object group), and hybrid "
                "models' KV (full attn and one linear attn/Mamba group)."
            )
        (og,) = others
        window = int(og["num_chunks_in_sw"])
        # KV must be every chunk. Query ring may be a sliding window
        if og["kind"] != "attention" or window == 0 or (is_kv and window > 0):
            raise LMCacheSDKError(
                f"object group {og['object_group_idx']} ({og['kind']}, {window} "
                "chunks in window) cannot be served in SDK mode, which "
                "supports full attention (and a query window) only"
            )
        self._window_chunks = window
        if len(og["chunk_object_shapes"]) != 1:
            raise LMCacheSDKError(
                f"object group {og['object_group_idx']} packs "
                f"{len(og['chunk_object_shapes'])} tensors; expected one (a "
                "hybrid model needs the server's --separate-object-groups)"
            )
        tokens_per_block = kernel_block_size = 0
        state_layouts: list[SDKObjectLayout] = []
        for group in recurrent:
            if int(group["num_chunks_in_sw"]) != 1:
                raise LMCacheSDKError(
                    f"recurrent object group {group['object_group_idx']} keeps "
                    f"{group['num_chunks_in_sw']} chunks; SDK mode reads one"
                )
            state_layouts = [
                SDKObjectLayout(
                    object_group_idx=int(group["object_group_idx"]),
                    shape=tuple(int(d) for d in shape),
                    dtype=getattr(torch, str(dtype).replace("torch.", "")),
                )
                for shape, dtype in zip(
                    group["chunk_object_shapes"],
                    group["chunk_object_dtypes"],
                    strict=True,
                )
            ]
            attention_group = None
            for kernel_group in kernel_groups:
                if int(kernel_group.get("object_group_idx", 0)) == int(
                    og["object_group_idx"]
                ):
                    attention_group = kernel_group
                    break
            if attention_group is None:
                raise LMCacheSDKError(
                    f"no kernel group belongs to attention object group "
                    f"{og['object_group_idx']}"
                )
            tokens_per_block = int(attention_group["tokens_per_block"])
            kernel_block_size = _attention_kernel_block_size(
                attention_group,
                *_model_kv_heads(
                    self._kind.base_model_name(self._model_name), self._world_size
                ),
                self._mp_conf,
            )
        layout = SDKObjectLayout(
            object_group_idx=int(og["object_group_idx"]),
            shape=tuple(int(d) for d in og["chunk_object_shapes"][0]),
            dtype=getattr(
                torch, str(og["chunk_object_dtypes"][0]).replace("torch.", "")
            ),
            tokens_per_block=tokens_per_block,
            kernel_block_size=kernel_block_size,
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
            recurrent=state_layouts,
        )
        logger.info(
            "Transferring %s through vLLM's Q ring on %s: %d bytes per chunk object",
            self._model_name,
            ring.device,
            layout.num_bytes,
        )

    @property
    def is_hybrid(self) -> bool:
        """Whether the model is hybrid: a KV retrieve also returns the
        recurrent state at its end, and a KV store needs it."""
        return self._transfer_ctx is not None and self._transfer_ctx.has_recurrent_state

    def windowed_range(self, num_tokens: int) -> tuple[int, int]:
        """Return what a retrieve of ``num_tokens`` tokens reads.

        A sliding-window kind (query tensors with vLLM's
        ``lmcache.mp.q.sw_size_tokens``) keeps only the trailing chunks
        readable, and under a sub-chunk window only each chunk's last rows.

        Args:
            num_tokens: Tokens from the key chain's first token.

        Returns:
            ``(start, rows)``: the chunk-aligned first token read, and the
            rows the retrieve returns from there.
        """
        chunk_size = self._chunk_size
        end = num_tokens // chunk_size * chunk_size
        start = 0
        if self._window_chunks > 0:
            start = max(0, end - self._window_chunks * chunk_size)
        rows = chunk_size
        if self._transfer_ctx is not None:
            rows = self._transfer_ctx.rows_per_chunk
        return start, (end - start) // chunk_size * rows

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

    def _await_lookup_result(self, request_id: str) -> int:
        """Block until a submitted LOOKUP completes and return its hit length.

        Args:
            request_id: The id the LOOKUP was submitted under.

        Returns:
            The cached prefix length in tokens.

        Raises:
            LMCacheSDKError: If the LOOKUP does not complete within mq_timeout.
        """
        start_time = time.time()
        result = self.check_lookup_result(request_id)
        while result is None:
            if time.time() - start_time > self.mq_timeout:
                self.end_session(request_id)
                raise LMCacheSDKError(
                    f"LOOKUP request timed out after {self.mq_timeout}s "
                    f"for request_id={request_id}"
                )
            logger.debug("Waiting for LOOKUP result for request_id=%s...", request_id)
            time.sleep(0.01)
            result = self.check_lookup_result(request_id)
        return result

    def lookup(self, tokens: Sequence[int], cache_salt: str = "") -> int:
        """Return how many chunk-aligned tokens are currently cached.

        Args:
            tokens: The full token sequence to look up, from token 0.
            cache_salt: Optional cache salt string for the lookup.

        Returns:
            The cached prefix length in tokens; 0 when nothing is cached.

        Raises:
            LMCacheSDKError: If the LOOKUP does not complete within mq_timeout.
        """
        total_tokens = (len(tokens) // self.chunk_size) * self.chunk_size
        if total_tokens == 0:
            return 0
        request_id = f"lookup-{uuid.uuid4().hex}"
        self.maybe_submit_lookup_request(
            request_id, list(tokens[:total_tokens]), cache_salt
        )
        try:
            return self._await_lookup_result(request_id)
        finally:
            self.end_session(request_id)

    def retrieve(
        self,
        tokens: Sequence[int],
        cache_salt: str = "",
        start_token_id: int = 0,
        *,
        request_configs: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, RecurrentState | None] | None:
        """Retrieve cached KV or query tensors for the given token IDs.

        Args:
            tokens: The token IDs the cache keys are chained from, starting at
                the chain's first token (token 0 for KV, the generate() pass's
                first computed token for query tensors). Indices below are
                relative to it.
            cache_salt: Optional cache salt string for the lookup.
            start_token_id: The first token to retrieve, aligned to
                chunk_size.
            request_configs: Optional LMCache request configs to include in
                the IPC key.

        Returns:
            ``(tensor, state)``: the cached chunks from start_token_id up to
            the first uncached one, as a CPU tensor ``[2, L, T, D]`` (KV; the
            attention layers' for a hybrid model) or ``[1, L, T, D]`` (query),
            and a hybrid model's recurrent state at its end, else None.
            None if nothing is retrieved.

        Raises:
            LMCacheSDKError: If start_token_id is not a multiple of chunk_size,
                the LOOKUP request does not complete in time, or the Q ring
                cannot serve the transfer.
        """
        if not tokens:
            logger.info("No tokens provided for retrieval; returning None.")
            return None
        if start_token_id % self.chunk_size != 0:
            raise LMCacheSDKError(
                f"start_token_id ({start_token_id}) must be a multiple of "
                f"chunk_size ({self.chunk_size})"
            )

        # Drop tokens not fit into a whole chunk
        total_tokens = (len(tokens) // self.chunk_size) * self.chunk_size
        if total_tokens <= start_token_id:
            logger.info(
                "Chunk-aligned token count (%d) does not exceed start_token_id "
                "(%d); returning None.",
                total_tokens,
                start_token_id,
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
                if num_prefetched_tokens <= start_token_id:
                    logger.info(
                        "Cached kind %s for request_id=%s covers %d tokens, which does "
                        "not reach start_token_id (%d); returning None.",
                        self.kind.name,
                        request_id,
                        num_prefetched_tokens,
                        start_token_id,
                    )
                    self.end_session(request_id)
                    return None

                # Now transfer. Only trailing chunks for the windowed kind (Q)
                end = min(total_tokens, num_prefetched_tokens)
                key = self._create_key(
                    token_ids=list(tokens[:end]),
                    start=max(start_token_id, self.windowed_range(end)[0]),
                    end=end,
                    request_id=request_id,
                    cache_salt=cache_salt,
                    request_configs=request_configs,
                    worker_id=0,
                )
                try:
                    result = self.transfer_ctx.retrieve(key, self.instance_id)
                except RingStateError as err:
                    raise LMCacheSDKError(str(err)) from err
                finally:
                    self.end_session(request_id)
                return result
            finally:
                try:
                    # Phase 2: Release the lock if not yet expired
                    if is_ttl_not_expired:
                        lock_key = self._create_key(
                            token_ids=list(tokens[:warm]),
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
        recurrent_state: RecurrentState | None = None,
    ) -> bool:
        """Store KV cache tensors for the given token IDs.

        Args:
            kv: The KV cache tensor to store, of shape [2, L, T, D].
            tokens: The list of token IDs corresponding to the KV cache tensor.
            cache_salt: Optional cache salt string for the store.
            request_configs: Optional LMCache request configs to include in
                the IPC key.
            recurrent_state: the recurrent state stored at the end of the
                chunk-aligned ``tokens`` (from retrieve).

        Returns:
            True if the store operation is successful, False otherwise.

        Raises:
            LMCacheSDKError: If the tokens do not match ``kv``, a hybrid store
                has no ``recurrent_state``, or the Q ring cannot serve the
                transfer.
        """
        if self.is_hybrid != (recurrent_state is not None):
            raise LMCacheSDKError(
                "a hybrid model's store needs recurrent_state (the engine "
                "cannot hit a prefix without its state), and only a hybrid "
                "model's takes it"
            )
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
            return self.transfer_ctx.store(
                key, self.instance_id, kv_cpu, recurrent_state
            )
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
