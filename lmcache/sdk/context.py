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
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.sdk.cache_kind import LMCacheSDKCacheKind
from lmcache.sdk.hybrid_layout import (
    ATTENTION,
    GroupPlan,
    HybridLayoutError,
    plan_hybrid_groups,
    text_config,
)
from lmcache.sdk.wrapper.paged_pool import (
    NULL_BLOCK_ID,
    PagedPoolTransferWrapper,
    PoolCapacityError,
    PoolGroup,
    RecurrentState,
)
from lmcache.v1.gpu_connector.kv_format.types import KVLayoutName
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.transfer_context.base import compute_kv_layout
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    LMCacheDrivenTransferContext,
    MPTransferMode,
    create_transfer_context,
)
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)

FULL_WINDOW = -1
"""``sw_size_tokens`` of a kind that keeps and returns its whole window."""


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


def _layer_labels(engine_kv_shape: str) -> list[str]:
    """Return the axis labels of one layer's tensor in a shape legend.

    Args:
        engine_kv_shape: The server-reported legend of a format, e.g.
            ``"NL x [NB, BS, NH, CS]"``.

    Returns:
        The per-layer labels, e.g. ``["NB", "BS", "NH", "CS"]``.

    Raises:
        LMCacheSDKError: If the format is not one tensor per layer.
    """
    prefix, bracket, rest = engine_kv_shape.partition("[")
    if not bracket or prefix.strip() != "NL x":
        raise LMCacheSDKError(
            f"unsupported KV shape {engine_kv_shape!r}: the SDK pool needs "
            "one tensor per layer"
        )
    return [label.strip() for label in rest.rstrip("] ").split(",")]


def _hf_config(hf_model_name: str) -> Any:
    """Return a model's Hugging Face config.

    Args:
        hf_model_name: Hugging Face repo id of the model.

    Returns:
        The config; multimodal ones (e.g. Qwen3.5) nest the language model's
        under ``text_config``.
    """
    # Third Party
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(hf_model_name)


def _hf_head_sizes(hf_model_name: str, world_size: int) -> dict[str, int]:
    """Return one worker's KV head sizes from the model's Hugging Face config.

    Args:
        hf_model_name: Hugging Face repo id of the model.
        world_size: Tensor-parallel world size the heads are split over.

    Returns:
        Sizes of the ``NH``, ``HS`` and fused-K/V ``CS`` (``2 * HS``) axes.
    """
    hf_config = text_config(_hf_config(hf_model_name))
    head_dim = getattr(
        hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
    )
    num_kv_heads = getattr(
        hf_config, "num_key_value_heads", hf_config.num_attention_heads
    )
    return {"NH": num_kv_heads // world_size, "HS": head_dim, "CS": 2 * head_dim}


def _pool_layer_shape(
    kernel_group: Mapping[str, str | int],
    labels: Sequence[str],
    kind: LMCacheSDKCacheKind,
    num_blocks: int,
    world_size: int,
    hf_model_name: str,
) -> tuple[int, ...]:
    """Return the per-layer shape of a pool mirroring a registered layout.

    Args:
        kernel_group: The engine's kernel group entry from ``/status``.
        labels: Per-layer axis labels of the group's format.
        kind: The cache kind the pool serves.
        num_blocks: Blocks the pool holds.
        world_size: Tensor-parallel world size the layout is registered under.
        hf_model_name: Hugging Face repo id used when the concrete shape is
            unknown.

    Returns:
        The shape of one layer's pool tensor.

    Raises:
        LMCacheSDKError: If the format has no ``NB``/``BS`` axes, an axis
            cannot be sized, or its block size differs from the engine's.
    """
    if "NB" not in labels or "BS" not in labels:
        raise LMCacheSDKError(f"layout {labels} has no separate NB and BS axes")
    tokens_per_block = int(kernel_group["tokens_per_block"])
    concrete = str(kernel_group["engine_kv_concrete_shape"])
    if concrete.startswith("Unknown"):
        # The config only describes KV heads; a query ring must be reported.
        if kind is LMCacheSDKCacheKind.QUERY:
            raise LMCacheSDKError(f"no concrete shape for the query ring: {concrete}")
        sizes = {
            "NB": num_blocks,
            "BS": tokens_per_block,
            **_hf_head_sizes(hf_model_name, world_size),
        }
        dims = []
        for label in labels:
            if label in sizes:
                dims.append(sizes[label])
            elif label.isdigit():
                dims.append(int(label))
            else:
                raise LMCacheSDKError(f"cannot size axis {label!r} of {labels}")
        return tuple(dims)
    dims = [int(size) for size in concrete.partition("[")[2].rstrip("] ").split(",")]
    if len(dims) != len(labels):
        raise LMCacheSDKError(f"concrete shape {concrete!r} does not match {labels}")
    dims[labels.index("NB")] = num_blocks
    if dims[labels.index("BS")] != tokens_per_block:
        raise LMCacheSDKError(
            f"block size {dims[labels.index('BS')]} of {concrete!r} is not "
            f"tokens_per_block {tokens_per_block}"
        )
    return tuple(dims)


def _pool_num_blocks(data_blocks: int) -> int:
    """Return a pool's block count for ``data_blocks`` blocks of data.

    Block 0 is the null block, which the server treats as "no data", so the
    data starts at block 1.

    Args:
        data_blocks: Blocks the pool must hold data in.

    Returns:
        ``data_blocks + 1``, plus a spare block when that would be 2: a
        leading dim of 2 would detect as the K/V axis of ``[2, NB, ...]``.
    """
    num_blocks = NULL_BLOCK_ID + 1 + data_blocks
    return num_blocks + 1 if num_blocks == 2 else num_blocks


def _kv_layout(labels: Sequence[str]) -> KVLayoutName:
    """Return the vLLM layout hint for a format's per-layer labels.

    Args:
        labels: Per-layer axis labels, which include ``BS``.

    Returns:
        ``"HND"`` when the heads axis precedes the block tokens, else ``"NHD"``.
    """
    if "NH" in labels and labels.index("NH") < labels.index("BS"):
        return "HND"
    return "NHD"


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


def _server_pool_chunks(mp_conf: Mapping[str, object], default: int = 4) -> int:
    """Return the pool size the server advertises, else ``default``.

    Args:
        mp_conf: The ``mp`` section of the server's ``/config``.
        default: Pool size when the server does not set one.

    Returns:
        The ``sdk.pool_chunks`` entry of the server's
        ``--runtime-plugin-config``, or ``default``.

    Raises:
        LMCacheSDKError: If the server's value is not a positive integer.
    """
    value = _server_extra_config(mp_conf, "sdk.pool_chunks", default)
    if value < 1:
        raise LMCacheSDKError(
            f"server sdk.pool_chunks={value!r} must be a positive integer"
        )
    return value


def _resolve_window(
    kind: LMCacheSDKCacheKind,
    sw_size_tokens: int | None,
    mp_conf: Mapping[str, object],
    chunk_size: int,
) -> int:
    """Return the ``sw_size_tokens``: window of last N tokens to be retrieved
    for each chunk. Only works for QUERY.

    Args:
        kind: The cache kind the pool serves.
        sw_size_tokens: The requested window, `FULL_WINDOW`, matching
            ``lmcache.mp.q.sw_size_tokens``.
        mp_conf: The ``mp`` section of the server's ``/config``.
        chunk_size: Tokens per LMCache chunk.

    Returns:
        The window in tokens, or :data:`FULL_WINDOW`.

    Raises:
        LMCacheSDKError: If a KV window is requested, the window is invalid,
            or a window spanning whole chunks is requested from a server
            without ``--separate-object-groups`` (it would be ignored).
    """
    if sw_size_tokens is None:
        sw_size_tokens = (
            _server_extra_config(mp_conf, "sdk.q_sw_size_tokens", FULL_WINDOW)
            if kind is LMCacheSDKCacheKind.QUERY
            else FULL_WINDOW
        )
    if sw_size_tokens != FULL_WINDOW and kind is not LMCacheSDKCacheKind.QUERY:
        raise LMCacheSDKError(
            f"only the QUERY kind can be windowed, got sw_size_tokens="
            f"{sw_size_tokens} for {kind.name}"
        )
    if sw_size_tokens != FULL_WINDOW and sw_size_tokens < 1:
        raise LMCacheSDKError(
            f"sw_size_tokens must be positive or {FULL_WINDOW}, got {sw_size_tokens}"
        )
    if sw_size_tokens >= chunk_size and not mp_conf.get(
        "separate_object_groups", False
    ):
        raise LMCacheSDKError(
            f"sw_size_tokens={sw_size_tokens} spans whole chunks, which the "
            "server only honors with --separate-object-groups"
        )
    return sw_size_tokens


def _resolve_device(device: str | torch.device) -> torch.device:
    """Return the accelerator device the pool lives on.

    Args:
        device: An accelerator device; without an index, the current one.

    Returns:
        The device with an explicit index.

    Raises:
        LMCacheSDKError: If ``device`` is a CPU device.
    """
    resolved = torch.device(device)
    if resolved.type == "cpu":
        raise LMCacheSDKError(
            "the SDK pool needs an accelerator device: the server's "
            "lmcache-driven path cannot map a CPU pool on accelerator hosts"
        )
    if resolved.index is None:
        resolved = torch.device(resolved.type, torch_dev.current_device())
    return resolved


class LMCacheSDKContext:
    """
    Retrieve and store KV cache tensors through an LMCache MP request client.

    The model layout must already be registered in the running LMCache server
    by an inference engine (e.g. a vLLM instance). The SDK registers a small
    paged pool on an accelerator in the engine's own layout, in lmcache-driven
    mode: the server copies cached chunks into the pool over device IPC, and
    the SDK hands them to the caller as contiguous CPU tensors.
    """

    def __init__(
        self,
        url: str,
        http_url: str,
        model_name: str,
        kind: LMCacheSDKCacheKind = LMCacheSDKCacheKind.KV,
        timeout: float = 60.0,
        device: str | torch.device = torch_device_type,
        pool_chunks: int | None = None,
        sw_size_tokens: int | None = None,
    ) -> None:
        """
        Initialize the SDK context and register the SDK transfer strategy.

        Args:
            url: Multiprocess request endpoint. The URL scheme selects the
                transport.
            http_url: HTTP endpoint URL for fetching information.
            model_name: Model name used by the running LMCache server instance.
            kind: The type of cache.
            timeout: Timeout in seconds for blocking MQ calls. Defaults to 60.
            device: Accelerator device the transfer pool is allocated on.
                Defaults to the current device.
            pool_chunks: LMCache chunks the transfer pool holds.
            sw_size_tokens: window of last N tokens retrieved @ each chunk.

        Returns:
            LMCacheSDKContext instance.

        Raises:
            LMCacheSDKError: If ``device`` is not an accelerator,
                ``pool_chunks`` (or the server's value) is not positive, the
                window is invalid for this kind or server, or the server
                cannot be reached.
        """
        if pool_chunks is not None and pool_chunks < 1:
            raise LMCacheSDKError(f"pool_chunks must be >= 1, got {pool_chunks}")
        self._device = _resolve_device(device)
        self._transfer_ctx: PagedPoolTransferWrapper | None = None
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
        # Resolved at registration, where a hybrid layout changes the default.
        self._pool_chunks_arg = pool_chunks
        self._pool_chunks = (
            pool_chunks if pool_chunks is not None else _server_pool_chunks(mp_conf)
        )
        self._sw_size_tokens = _resolve_window(
            kind, sw_size_tokens, mp_conf, self._chunk_size
        )
        # The server reads only a window's last chunks with separation on.
        self._chunk_windowed = bool(mp_conf.get("separate_object_groups", False))

        # Engine registrations of this kind's layout, keyed by instance ID.
        self._engine_meta_conf = {}
        try:
            response = requests.get(f"{self._http_url}/status", timeout=timeout)
            response.raise_for_status()
            self._engine_meta_conf = response.json().get(kind.status_meta_field(), {})
        except (requests.RequestException, KeyError, ValueError) as err:
            raise LMCacheSDKError(
                f"failed to fetch server config from {self._http_url}/status"
            ) from err

        self._pending_lookups: set[str] = set()
        self._finished_lookups: dict[str, int] = {}

        logger.info(
            f"Initialized LMCacheSDKContext with instance_id={self.instance_id}, "
            f"model_name={self._model_name}, chunk_size={self._chunk_size}, "
            f"device={self._device}, kind={self._kind}, "
            f"sw_size_tokens={self._sw_size_tokens}"
        )

    @property
    def kind(self) -> LMCacheSDKCacheKind:
        """Return type of tensor operation the context serves."""
        return self._kind

    @property
    def sw_size_tokens(self) -> int:
        """The window this kind's pool is registered with, or ``FULL_WINDOW``."""
        return self._sw_size_tokens

    def windowed_range(self, window_tokens: int) -> tuple[int, int]:
        """Return the range of tokens this kind's addressable window covers.
        The addressable window is ``[key_origin, cached_len)`` (see
        ``LMCacheSDKCacheKind.key_origin``).

        Args:
            window_tokens: Tokens the addressable window covers.

        Returns:
            ``(start_offset, expected_rows)``: the chunk-aligned offset of the
            first chunk the server returns, and the rows it returns from
            there (each chunk's kept tokens).
        """
        chunk_size = self._chunk_size
        aligned = window_tokens // chunk_size * chunk_size
        if self._sw_size_tokens == FULL_WINDOW:
            return 0, aligned
        start = 0
        if self._chunk_windowed:
            num_chunks = -(-self._sw_size_tokens // chunk_size)
            start = max(0, aligned - num_chunks * chunk_size)
        rows_per_chunk = min(self._sw_size_tokens, chunk_size)
        return start, (aligned - start) // chunk_size * rows_per_chunk

    def register_caches(
        self,
    ) -> None:
        """Register a transfer pool mirroring the engine's layout.

        The engine's kernel groups are planned first: one attention group for
        a dense model or the query ring, or each group's role for a hybrid
        model (see :mod:`lmcache.sdk.hybrid_layout`). One pool tensor per
        layer is then allocated in its group's engine format and registered
        with the same groups, windows included, as the engine registered.

        Raises:
            LMCacheSDKError: If no engine registered this kind's layout for
                the model, the layout is unsupported, or the pool does not
                reproduce the engine's format.
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
                "(with query transfer enabled for QUERY) first."
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
            num_layers = int(layout["num_layers"])
            kernel_groups = layout.get("kernel_groups", [])
            if not kernel_groups:
                raise LMCacheSDKError(
                    f"the {self._kind.name} layout of {self._model_name!r} "
                    "reports no kernel groups"
                )
            plans, pool_chunks = self._plan_groups(num_layers, kernel_groups)
            pool, pool_groups, layout_hints, num_planes = self._allocate_pool(
                plans, kernel_groups, num_layers, pool_chunks
            )
        except LMCacheSDKError:
            raise
        except Exception as err:
            raise LMCacheSDKError(
                f"failed to decode the registered layout for "
                f"model_name={self._model_name!r}"
            ) from err

        transfer_ctx = create_transfer_context(
            pool,
            instance_id=self.instance_id,
            req_client=self._req_client,
            mode=MPTransferMode.LMCACHE_DRIVEN,
        )
        if not isinstance(transfer_ctx, LMCacheDrivenTransferContext):
            raise LMCacheSDKError(
                "SDK requires an lmcache-driven transfer context, got "
                f"{type(transfer_ctx).__name__}."
            )
        with torch_dev.device(self._device):
            transfer_ctx.register(
                pool,
                self._model_name,
                self._world_size,
                self._chunk_size // plans[0].tokens_per_block,
                self._mq_timeout,
                layout_hints=layout_hints,
                # The same groups and windows as the engine registered: the
                # server keeps one layout per model, the latest registration's.
                engine_group_infos=[plan.engine_group_info() for plan in plans],
            )
        sw_size_tokens = self._sw_size_tokens
        self._transfer_ctx = PagedPoolTransferWrapper(
            transfer_ctx,
            self.instance_id,
            pool,
            pool_groups,
            layout_hints,
            self._chunk_size,
            num_chunks=pool_chunks,
            num_planes=num_planes,
            # A sub-chunk window keeps each chunk's last tokens only.
            tokens_per_chunk=(
                self._chunk_size
                if sw_size_tokens == FULL_WINDOW
                else min(sw_size_tokens, self._chunk_size)
            ),
            req_client=self._req_client,
            timeout=self._mq_timeout,
        )
        pool_bytes = sum(t.numel() * t.element_size() for t in pool.values())
        logger.info(
            "Registered %s pool for model_name=%s on %s: %d kernel group(s), "
            "%d layers, %d tokens (%d chunks) per transfer, sw_size_tokens=%d, "
            "%.2f GiB",
            self._kind.name,
            self._model_name,
            self._device,
            len(plans),
            num_layers,
            pool_chunks * self._chunk_size,
            pool_chunks,
            sw_size_tokens,
            pool_bytes / 2**30,
        )

    def _plan_groups(
        self, num_layers: int, kernel_groups: Sequence[Mapping[str, Any]]
    ) -> tuple[list[GroupPlan], int]:
        """Plan how the pool mirrors the engine's kernel groups.

        A single group is one attention group with this kind's window. Several
        groups are a hybrid model, whose roles are inferred from its Hugging
        Face config; its recurrent state cannot be transferred in batches, so
        its pool holds a whole prefix.

        Args:
            num_layers: Registered layers of the engine's layout.
            kernel_groups: The layout's kernel groups, in kernel-group order.

        Returns:
            ``(plans, pool_chunks)``: one plan per kernel group, and the
            chunks the pool's attention groups hold.

        Raises:
            LMCacheSDKError: If the groups cannot be mirrored.
        """
        if len(kernel_groups) == 1:
            group = kernel_groups[0]
            tokens_per_block = int(group["tokens_per_block"])
            if int(group["slots_per_block"]) != tokens_per_block:
                raise LMCacheSDKError(
                    "compressed layouts (slots_per_block != tokens_per_block) "
                    "are not supported"
                )
            sw_size_tokens = self._sw_size_tokens
            if sw_size_tokens != FULL_WINDOW and sw_size_tokens % tokens_per_block:
                raise LMCacheSDKError(
                    f"sw_size_tokens {sw_size_tokens} is not a multiple of "
                    f"tokens_per_block {tokens_per_block}"
                )
            plans = [
                GroupPlan(
                    kernel_group_idx=int(group.get("kernel_group_idx", 0)),
                    engine_group_idx=int(group.get("engine_group_idx", 0)),
                    object_group_idx=int(group.get("object_group_idx", 0)),
                    layer_indices=tuple(range(num_layers)),
                    tokens_per_block=tokens_per_block,
                    role=ATTENTION,
                    sw_size_tokens=sw_size_tokens,
                    kernel_block_size=tokens_per_block,
                    source="single group",
                )
            ]
            pool_chunks = self._pool_chunks
        else:
            if self._kind is not LMCacheSDKCacheKind.KV:
                raise LMCacheSDKError(
                    f"the {self._kind.name} layout has {len(kernel_groups)} "
                    "kernel groups; only the KV kind supports hybrid layouts"
                )
            if not self._mp_conf.get("separate_object_groups", False):
                raise LMCacheSDKError(
                    "a hybrid model's KV needs the server's "
                    "--separate-object-groups (recurrent state and attention "
                    "KV are stored per object group)"
                )
            hf_config = _hf_config(self._kind.base_model_name(self._model_name))
            try:
                plans = plan_hybrid_groups(
                    kernel_groups,
                    hf_config,
                    self._world_size,
                    _server_extra_config(self._mp_conf, "sdk.kernel_block_size", 0),
                )
            except HybridLayoutError as err:
                raise LMCacheSDKError(
                    f"cannot mirror the hybrid layout of {self._model_name!r}: {err}"
                ) from err
            covered = sorted(i for plan in plans for i in plan.layer_indices)
            if covered != list(range(num_layers)):
                raise LMCacheSDKError(
                    f"the kernel groups of {self._model_name!r} do not cover its "
                    f"{num_layers} registered layers exactly once"
                )
            pool_chunks = self._hybrid_pool_chunks(hf_config)
        for plan in plans:
            if self._chunk_size % plan.tokens_per_block:
                raise LMCacheSDKError(
                    f"chunk_size {self._chunk_size} is not a multiple of kernel "
                    f"group {plan.kernel_group_idx}'s tokens_per_block "
                    f"{plan.tokens_per_block}"
                )
        return plans, pool_chunks

    def _allocate_pool(
        self,
        plans: Sequence[GroupPlan],
        kernel_groups: Sequence[Mapping[str, Any]],
        num_layers: int,
        pool_chunks: int,
    ) -> tuple[dict[str, torch.Tensor], list[PoolGroup], LayoutHints, int]:
        """Allocate one pool tensor per layer in its group's engine format.

        Block 0 of every group is the null block; attention groups hold
        ``pool_chunks`` chunks, recurrent groups one (only the state at the
        end of a prefix is ever stored or read).

        Args:
            plans: One plan per kernel group (see :meth:`_plan_groups`).
            kernel_groups: The layout's kernel groups from ``/status``.
            num_layers: Registered layers of the engine's layout.
            pool_chunks: Chunks the attention groups hold.

        Returns:
            ``(pool, groups, layout_hints, num_planes)``: the tensors keyed
            ``layer.<i>`` in registration order, the wrapper's groups, the
            layout hints they are registered with, and the planes of the
            attention groups' contiguous tensors.

        Raises:
            LMCacheSDKError: If a group's pool does not reproduce the engine's
                format.
        """
        by_index = {int(g.get("kernel_group_idx", 0)): g for g in kernel_groups}
        hf_model_name = self._kind.base_model_name(self._model_name)
        layout_hints = LayoutHints(
            kv_layout=_kv_layout(
                _layer_labels(str(kernel_groups[0]["engine_kv_shape"]))
            )
        )
        tensors: dict[int, torch.Tensor] = {}
        pool_groups: list[PoolGroup] = []
        num_planes = 0
        for plan in plans:
            group = by_index[plan.kernel_group_idx]
            blocks_per_chunk = self._chunk_size // plan.tokens_per_block
            num_blocks = _pool_num_blocks(
                blocks_per_chunk * (1 if plan.recurrent else pool_chunks)
            )
            layer_shape = _pool_layer_shape(
                group,
                _layer_labels(str(group["engine_kv_shape"])),
                self._kind,
                num_blocks,
                self._world_size,
                hf_model_name,
            )
            dtype = getattr(torch, str(group["dtype"]).replace("torch.", ""))
            group_tensors = {
                f"layer.{i}": torch.zeros(layer_shape, dtype=dtype, device=self._device)
                for i in plan.layer_indices
            }
            fmt = getattr(lmcache_native.EngineKVFormat, str(group["engine_kv_format"]))
            block_size, _, _, _, pool_fmt, planes = compute_kv_layout(
                group_tensors, layout_hints=layout_hints
            )
            if pool_fmt != fmt or block_size != plan.tokens_per_block:
                raise LMCacheSDKError(
                    f"kernel group {plan.kernel_group_idx}'s pool of per-layer "
                    f"shape {layer_shape} detects as {pool_fmt.name} with block "
                    f"size {block_size}, not the engine's {fmt.name} with block "
                    f"size {plan.tokens_per_block}"
                )
            if not plan.recurrent and not num_planes:
                num_planes = planes
            tensors.update(
                {int(name.split(".")[1]): t for name, t in group_tensors.items()}
            )
            pool_groups.append(
                PoolGroup(
                    layer_names=tuple(group_tensors),
                    tokens_per_block=plan.tokens_per_block,
                    recurrent=plan.recurrent,
                    kernel_block_size=plan.kernel_block_size,
                )
            )
        pool = {f"layer.{i}": tensors[i] for i in range(num_layers)}
        return pool, pool_groups, layout_hints, num_planes

    def _hybrid_pool_chunks(self, hf_config: Any) -> int:
        """Chunks a hybrid pool holds: one whole prefix.

        Args:
            hf_config: The model's Hugging Face config.

        Returns:
            The explicit ``pool_chunks``, else the server's
            ``sdk.pool_chunks``, else the model's context length in chunks.

        Raises:
            LMCacheSDKError: If none is available.
        """
        if self._pool_chunks_arg is not None:
            return self._pool_chunks_arg
        # 0: the server does not set sdk.pool_chunks.
        if _server_extra_config(self._mp_conf, "sdk.pool_chunks", 0):
            return _server_pool_chunks(self._mp_conf)
        max_tokens = getattr(text_config(hf_config), "max_position_embeddings", None)
        if not max_tokens:
            raise LMCacheSDKError(
                "the model config has no max_position_embeddings to size the "
                "hybrid pool; pass pool_chunks"
            )
        return -(-int(max_tokens) // self._chunk_size)

    @property
    def is_hybrid(self) -> bool:
        """Whether the pool mirrors a hybrid (attention + recurrent) layout.

        A hybrid retrieve also returns the recurrent state at the end of the
        range (see :meth:`retrieve_with_state`), and a hybrid store needs it.
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
    def transfer_ctx(self) -> PagedPoolTransferWrapper:
        """Return the pool transfer wrapper.

        Raises:
            LMCacheSDKError: If ``register_caches`` has not been called.
        """
        if self._transfer_ctx is None:
            raise LMCacheSDKError("register_caches() must be called first")
        return self._transfer_ctx

    def close(self) -> None:
        """Unregister the transfer pool, if any, and close the request client."""
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
    ) -> torch.Tensor | None:
        """Retrieve KV/Query cache tensors for the given token IDs.

        Args:
            tokens: The token IDs the cache keys are chained from, starting at
                the chain's first token (token 0 for KV, the generate() pass's
                first computed token for query tensors). Indices below are
                relative to it.
            cache_salt: Optional cache salt string for the lookup.
            request_configs: Optional LMCache request configs to include in
                the IPC key.
            start_token_id: The starting token ID for the retrieval.

        Returns:
            A contiguous CPU tensor holding the cached chunks from
            start_token_id onwards. It stops at the first uncached chunk, so it
            may cover fewer tokens than requested. A windowed kind (see
            ``window``) starts no earlier than the window, and returns only
            the tokens each chunk keeps.
            None if retrieval fails, there are no tokens to retrieve, or
            nothing is cached at start_token_id.

        Raises:
            LMCacheSDKError: If start_token_id is not a multiple of chunk_size,
            or if the LOOKUP request does not complete successfully.
        """
        result = self._retrieve_range(
            tokens, cache_salt, start_token_id, request_configs
        )
        return result[0] if result is not None else None

    def retrieve_with_state(
        self,
        tokens: Sequence[int],
        cache_salt: str = "",
        start_token_id: int = 0,
        *,
        request_configs: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, RecurrentState | None] | None:
        """Retrieve KV tensors and, for a hybrid model, the recurrent state.

        Args:
            tokens: The token IDs the cache keys are chained from.
            cache_salt: Optional cache salt string for the lookup.
            start_token_id: The starting token ID for the retrieval.
            request_configs: Optional LMCache request configs to include in
                the IPC key.

        Returns:
            ``(kv, state)`` as :meth:`retrieve` returns ``kv``; ``state`` is
            the recurrent state at the end of ``kv`` for a hybrid model, else
            None. None if nothing is retrieved.

        Raises:
            LMCacheSDKError: As :meth:`retrieve`, or if a hybrid range
                exceeds the pool.
        """
        result = self._retrieve_range(
            tokens, cache_salt, start_token_id, request_configs
        )
        return result

    def _retrieve_range(
        self,
        tokens: Sequence[int],
        cache_salt: str,
        start_token_id: int,
        request_configs: dict[str, object] | None,
    ) -> tuple[torch.Tensor, RecurrentState | None] | None:
        """Retrieve the cached range; see :meth:`retrieve`.

        Returns:
            ``(tensor, state)`` from the pool (``state`` is None unless the
            model is hybrid), or None if nothing is retrieved.
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
        except PoolCapacityError as err:
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
                (e.g. the state :meth:`retrieve_with_state` returned).

        Returns:
            True if the store operation is successful, False otherwise.

        Raises:
            LMCacheSDKError: If the tokens do not match ``kv``, or a hybrid
                store has no ``recurrent_state`` or exceeds the pool.
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
        except PoolCapacityError as err:
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
        return IPCCacheServerKey(
            model_name=self._model_name,
            world_size=self._world_size,
            num_kv_readers=self._num_kv_readers,
            worker_id=worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
            request_configs=request_configs,
        )
