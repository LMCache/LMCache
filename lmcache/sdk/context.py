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
    SDKKernelGroup,
    SDKObjectGroup,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

logger = init_logger(__name__)

FULL_WINDOW = -1
"""A window that keeps everything: all chunks of a range, or every row."""


class LMCacheSDKError(RuntimeError):
    """Raised when an SDK KV-cache operation fails."""


class ModifyTensors(dict[LMCacheSDKCacheKind, torch.Tensor]):
    """The cached tensors a modify function edits, keyed by cache kind.

    A plain ``dict`` of the tensors, plus where the stream's latest decoded
    segment begins. A windowed query tensor covers only the end of that
    segment, so its length does not tell where the segment starts.

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
    [ModifyTensors, Sequence[int]],
    tuple[torch.Tensor, Sequence[int]],
]


def _server_extra_config(mp_conf: Mapping[str, object], key: str, default: int) -> int:
    """Return an integer entry of the server's ``--runtime-plugin-config``.

    Args:
        mp_conf: The ``mp`` section of the server's ``/config``.
        key: The entry to read.
        default: Value when the server does not set ``key``.

    Returns:
        The entry, or ``default``.

    Raises:
        LMCacheSDKError: If the entry is not an integer.
    """
    plugin_config = mp_conf.get("runtime_plugin_config")
    extra = (
        plugin_config.get("extra_config") if isinstance(plugin_config, dict) else None
    )
    value = extra.get(key, default) if isinstance(extra, dict) else default
    if isinstance(value, bool) or not isinstance(value, int):
        raise LMCacheSDKError(f"server {key}={value!r} must be an integer")
    return value


def _model_kv_heads(hf_model_name: str, world_size: int) -> tuple[int, int]:
    """Return one worker's KV heads and head size, from the model's config.

    Args:
        hf_model_name: Hugging Face repo id of the model.
        world_size: Tensor-parallel world size the heads are split over.

    Returns:
        ``(kv_heads_per_worker, head_dim)``.
    """
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


def _registered_heads(group: Mapping[str, Any]) -> tuple[int, int]:
    """Return the registered ``(NH, HS)`` of a ``[NB, 2, BS, NH, HS]`` group.

    Raises:
        LMCacheSDKError: If the format is not that one.
    """
    labels = str(group["engine_kv_shape"]).partition("[")[2].rstrip("] ").split(",")
    labels = [label.strip() for label in labels]
    if labels != ["NB", "2", "BS", "NH", "HS"]:
        raise LMCacheSDKError(
            f"attention group {group['kernel_group_idx']} has format "
            f"{group['engine_kv_shape']}; the hybrid path supports "
            "NL x [NB, 2, BS, NH, HS] only"
        )
    dims = str(group["engine_kv_concrete_shape"]).partition("[")[2].rstrip("] ")
    sizes = [int(size) for size in dims.split(",")]
    return sizes[3], sizes[4]


def _attention_kernel_block_size(
    group: Mapping[str, Any],
    num_kv_heads: int,
    head_dim: int,
    override: int = 0,
) -> int:
    """Return the kernel page size of an attention group's registered pages.

    Args:
        group: The kernel group entry from ``/status``.
        num_kv_heads: KV heads per worker, from the model config.
        head_dim: Head size, from the model config.
        override: The kernel block size set on the server
            (``sdk.kernel_block_size``), or 0 if unset.

    Returns:
        The kernel page size: ``tokens_per_block`` when not sub-paged.

    Raises:
        LMCacheSDKError: If the registered heads match neither view, or
            the pages are sub-paged and ``override`` is unset or does not
            divide the block.
    """
    tokens_per_block = int(group["tokens_per_block"])
    heads, width = _registered_heads(group)
    if (heads, width) == (num_kv_heads, head_dim):
        return tokens_per_block
    if heads != 1 or width != num_kv_heads * head_dim:
        raise LMCacheSDKError(
            f"attention group {group['kernel_group_idx']} registers "
            f"{heads} heads x {width}, matching neither the model's "
            f"{num_kv_heads} x {head_dim} nor a sub-paged 1 x "
            f"{num_kv_heads * head_dim}"
        )
    if not override:
        raise LMCacheSDKError(
            f"attention group {group['kernel_group_idx']} is sub-paged (one "
            f"synthetic head of {width}) and the registration does not report "
            "its kernel page size. Set vLLM's attention kernel block size on "
            "the server: --runtime-plugin-config "
            "'{\"sdk.kernel_block_size\": 32}' (32 for FlashAttention or "
            f"FlashInfer with a {tokens_per_block}-token block)"
        )
    if override >= tokens_per_block or tokens_per_block % override:
        raise LMCacheSDKError(
            f"sdk.kernel_block_size={override} must be a proper divisor of the "
            f"sub-paged block size {tokens_per_block}"
        )
    logger.info(
        "Attention group %s is sub-paged: %d-token kernel pages "
        "(sdk.kernel_block_size) in %d-token blocks",
        group["kernel_group_idx"],
        override,
        tokens_per_block,
    )
    return override


class LMCacheSDKContext:
    """
    Retrieve and store KV cache tensors through an LMCache MP request client.

    The model layout must already be registered in the running LMCache server
    by an inference engine (a vLLM instance with query transfer enabled). The
    SDK allocates and registers nothing: the server copies cached chunks' L1
    objects, as raw bytes, into free blocks of the engine's Q ring over device
    IPC, and the SDK decodes them into contiguous CPU tensors (see
    `sdk.wrapper.ring_transfer`). vLLM must stop while the SDK transfers.
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
        # By default, the SDK retrieves all tokens' KV (FULL_WINDOW),
        # later for Q, only retrieve the last window (see `windowed_range`).
        self._window_chunks = FULL_WINDOW
        self._rows_per_chunk = self._chunk_size

        # Engine registrations of this kind's layout, keyed by instance ID, and
        # the L1 object layouts the server stores them with.
        self._engine_meta_conf = {}
        self._object_layouts: list[dict[str, Any]] = []
        try:
            response = requests.get(f"{self._http_url}/status", timeout=timeout)
            response.raise_for_status()
            status = response.json()
            self._engine_meta_conf = status.get(kind.status_meta_field(), {})
            self._object_layouts = status.get("object_layouts", [])
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

    @property
    def kind(self) -> LMCacheSDKCacheKind:
        """Return type of tensor operation the context serves."""
        return self._kind

    def windowed_range(self, window_tokens: int) -> tuple[int, int]:
        """Return the range of tokens this kind's addressable window covers.
        The addressable window is ``[key_origin, cached_len)`` (see
        ``LMCacheSDKCacheKind.key_origin``) or vLLM's
        ``lmcache.mp.q.sw_size_tokens``.

        Args:
            window_tokens: Tokens the addressable window covers.

        Returns:
            ``(start_offset, expected_rows)``: the chunk-aligned offset of the
            first chunk the server returns, and the rows it returns from
            there (each chunk's kept tokens).
        """
        chunk_size = self._chunk_size
        aligned = window_tokens // chunk_size * chunk_size
        start = 0
        if self._window_chunks != FULL_WINDOW:
            start = max(0, aligned - self._window_chunks * chunk_size)
        return start, (aligned - start) // chunk_size * self._rows_per_chunk

    def register_caches(
        self,
    ) -> None:
        """Connect to the engine's Q ring for this kind's transfers.

        Reads the kind's object layout (which object groups hold attention or
        recurrent state, and each one's tensors) and transfers its L1 objects
        through the ring as raw bytes.

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
            # Readers per stored object; registrations that do not publish it
            # get 1 (single reader).
            self._num_kv_readers = int(entry.get("num_kv_readers", 1))
            layout = entry.get(self._kind.status_layout_field(), {})
            if not layout:
                raise LMCacheSDKError(
                    f"no registered {self._kind.name} layout for {self._model_name!r}."
                )
            kernel_groups = layout.get("kernel_groups", [])
            if not kernel_groups:
                raise LMCacheSDKError(
                    f"the {self._kind.name} layout of {self._model_name!r} "
                    "reports no kernel groups"
                )
            if self._kind is LMCacheSDKCacheKind.QUERY and len(kernel_groups) != 1:
                raise LMCacheSDKError(
                    f"the Q ring of {self._model_name!r} reports "
                    f"{len(kernel_groups)} kernel groups; expected one"
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

        KV objects follow the engine's KV registration and can be stored.
        The query ring's objects follow its own registration, are read-only, and
        take the ring's window.

        The object layout comes from qstore's ``object_layouts`` report (the
        server's layout registry): each object group's kind, window and
        tensors. An object group's tensors are its kernel groups in ascending
        order, as the registration's ``kernel_groups`` assigns them; those also
        give each group's block size. A hybrid model's attention may be
        sub-paged (see `_attention_kernel_block_size`).

        Args:
            kernel_groups: The registration's kernel groups from ``/status``.

        Raises:
            LMCacheSDKError: If the server does not report the model's object
                layout or it does not match the kernel groups, a layout is
                compressed or not chunk-aligned, KV runs with TP > 1, an
                object group's window cannot be served, the sub-paging cannot
                be decoded, or no Q ring is registered.
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
        by_index = {int(g.get("kernel_group_idx", 0)): g for g in kernel_groups}
        members: dict[int, list[int]] = {}
        for index, kernel_group in sorted(by_index.items()):
            tokens_per_block = int(kernel_group["tokens_per_block"])
            if int(kernel_group["slots_per_block"]) != tokens_per_block:
                raise LMCacheSDKError(
                    f"kernel group {index} is compressed (slots_per_block != "
                    "tokens_per_block), which SDK mode does not decode"
                )
            if self._chunk_size % tokens_per_block:
                raise LMCacheSDKError(
                    f"chunk_size {self._chunk_size} is not a multiple of kernel "
                    f"group {index}'s tokens_per_block {tokens_per_block}"
                )
            owner = int(kernel_group.get("object_group_idx", 0))
            members.setdefault(owner, []).append(index)
        if is_kv and self._world_size != 1:
            raise LMCacheSDKError(
                f"SDK mode supports KV at TP 1 only, the engine runs "
                f"world_size={self._world_size}"
            )
        hybrid = len(kernel_groups) > 1
        if hybrid and not self._mp_conf.get("separate_object_groups", False):
            raise LMCacheSDKError(
                "a hybrid model's KV needs the server's --separate-object-groups "
                "(recurrent state and attention KV are stored per object group)"
            )
        heads: tuple[int, int] | None = None
        groups = []
        for og in object_groups:
            recurrent = og["kind"] == "recurrent"
            window = int(og["num_chunks_in_sw"])
            # KV attention must be whole (its stores write every chunk); the
            # query ring's window is read as the server reads it.
            unservable = (
                window != 1 if recurrent else (is_kv and window >= 0) or window == 0
            )
            if og["kind"] == "aux" or (recurrent and not is_kv) or unservable:
                raise LMCacheSDKError(
                    f"object group {og['object_group_idx']} ({og['kind']}, "
                    f"{window} chunks in window) cannot be served in SDK mode"
                )
            group_id = int(og["object_group_idx"])
            member_ids = members.get(group_id, [])
            if len(member_ids) != len(og["chunk_object_shapes"]):
                raise LMCacheSDKError(
                    f"object group {group_id} packs "
                    f"{len(og['chunk_object_shapes'])} tensors but has kernel "
                    f"groups {member_ids}"
                )
            tensors = []
            for kg, shape, dtype in zip(
                member_ids,
                og["chunk_object_shapes"],
                og["chunk_object_dtypes"],
                strict=True,
            ):
                tokens_per_block = int(by_index[kg]["tokens_per_block"])
                kernel_block_size = tokens_per_block
                # vLLM sub-pages attention only under its hybrid allocator.
                if hybrid and not recurrent:
                    if heads is None:
                        heads = _model_kv_heads(
                            self._kind.base_model_name(self._model_name),
                            self._world_size,
                        )
                    kernel_block_size = _attention_kernel_block_size(
                        by_index[kg],
                        *heads,
                        _server_extra_config(self._mp_conf, "sdk.kernel_block_size", 0),
                    )
                tensors.append(
                    SDKKernelGroup(
                        kernel_group_idx=kg,
                        recurrent=recurrent,
                        tokens_per_block=tokens_per_block,
                        kernel_block_size=kernel_block_size,
                        shape=tuple(int(d) for d in shape),
                        dtype=getattr(torch, str(dtype).replace("torch.", "")),
                    )
                )
            groups.append(
                SDKObjectGroup(
                    object_group_idx=group_id,
                    recurrent=recurrent,
                    window_chunks=window,
                    kernel_groups=tuple(tensors),
                )
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
        wrapper = RingSDKTransferWrapper(
            ring, groups, self._chunk_size, self._mq_timeout, writable=is_kv
        )
        self._window_chunks = wrapper.window_chunks
        self._rows_per_chunk = wrapper.rows_per_chunk
        self._transfer_ctx = wrapper
        logger.info(
            "Transferring %s through vLLM's Q ring on %s: %d object group(s), "
            "%s bytes per chunk object, window %d chunk(s) x %d rows",
            self._model_name,
            ring.device,
            len(groups),
            [group.num_bytes for group in groups],
            self._window_chunks,
            self._rows_per_chunk,
        )

    @property
    def is_hybrid(self) -> bool:
        """Whether the model is hybrid (attention + recurrent groups).

        A hybrid retrieve also returns the recurrent state at the end of the
        range (see `retrieve`), and a hybrid store needs it.
        """
        return self._transfer_ctx is not None and self._transfer_ctx.has_recurrent_state

    @property
    def chunk_size(self) -> int:
        """Return the chunk size of the context."""
        return self._chunk_size

    @property
    def mq_timeout(self) -> float:
        """Return the message queue timeout of the context."""
        return self._mq_timeout

    @property
    def transfer_ctx(
        self,
    ) -> RingSDKTransferWrapper:
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
        start_token_id: int = 0,
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
            start_token_id: The starting token ID for the lookup.
        """
        if request_id in self._pending_lookups:
            # Skip if there is already a lookup request
            return

        aligned_end = (len(token_ids) // self._chunk_size) * self._chunk_size

        key = self._create_key(
            token_ids,
            start=start_token_id,
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

        token_count = result * self._chunk_size
        self._finished_lookups[request_id] = token_count
        return token_count

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
            start_token_id: The starting token ID for the retrieval.
            request_configs: Optional LMCache request configs to include in
                the IPC key.

        Returns:
            ``(tensor, state)``. ``tensor`` is a contiguous CPU tensor holding
            the cached chunks from start_token_id onwards: ``[2, L, T, D]`` K
            and V (the attention layers' only, for a hybrid model) or
            ``[1, L, T, D]`` query rows. It stops at the first uncached chunk,
            so it may cover fewer tokens than requested. ``state`` is a hybrid
            model's recurrent state at the end of ``tensor`` (pass it to
            `store` with the edited KV), else None.
            None if retrieval fails, there are no tokens to retrieve, or
            nothing is cached at start_token_id.

        Raises:
            LMCacheSDKError: If start_token_id is not a multiple of chunk_size,
                the LOOKUP request does not complete successfully, or the Q
                ring cannot serve the transfer.
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

        # Assign request ID to this request
        request_id = f"retrieve-{uuid.uuid4().hex}"

        # Look up the whole range even when only a later part of it is wanted
        # since only locked chunks are readable.
        self.maybe_submit_lookup_request(
            request_id,
            token_ids=list(tokens[:total_tokens]),
            cache_salt=cache_salt,
            request_configs=request_configs,
            start_token_id=0,
        )

        num_prefetched_tokens = self._await_lookup_result(request_id)

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

        end = min(total_tokens, num_prefetched_tokens)
        # A windowed kind only has its trailing chunks readable.
        start = max(start_token_id, self.windowed_range(end)[0])

        # Phase 1: retrieve the cached range as one contiguous tensor.
        key = self._create_key(
            token_ids=list(tokens[:end]),
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
            request_configs=request_configs,
            worker_id=0,
        )
        try:
            return self.transfer_ctx.retrieve(key, self.instance_id)
        except RingStateError as err:
            raise LMCacheSDKError(str(err)) from err
        except LMCacheSDKError:
            logger.info(
                "Retrieve failed for kind %s request_id=%s at [%d, %d); "
                "returning None.",
                self.kind.name,
                request_id,
                start,
                end,
                exc_info=True,
            )
            return None
        finally:
            self.end_session(request_id)

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
            kv: The KV cache tensor to store, of shape [2, L, T, D]; for a
                hybrid model, the attention layers only.
            tokens: The list of token IDs corresponding to the KV cache tensor.
            cache_salt: Optional cache salt string for the store.
            request_configs: Optional LMCache request configs to include in
                the IPC key.
            recurrent_state: Hybrid models only: the recurrent state stored
                as the state at the end of the chunk-aligned ``tokens``
                (e.g. the state `retrieve` returned).

        Returns:
            True if the store operation is successful, False otherwise.

        Raises:
            LMCacheSDKError: If the tokens do not match ``kv``, a hybrid
                store has no ``recurrent_state``, or the Q ring cannot serve
                the transfer.
        """
        if self.is_hybrid and recurrent_state is None:
            raise LMCacheSDKError(
                "a hybrid model's store needs recurrent_state: without the "
                "state at the prefix end, the engine cannot hit the prefix"
            )
        if not self.is_hybrid and recurrent_state is not None:
            raise LMCacheSDKError("recurrent_state only applies to hybrid models")
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
            num_kv_readers=self._num_kv_readers,
            worker_id=worker_id,
            token_ids=token_ids,
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
            request_configs=request_configs,
        )
