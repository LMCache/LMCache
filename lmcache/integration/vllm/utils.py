# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Sequence
import hashlib
import os
import threading
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.request import Request

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig, load_ec_engine_config
from lmcache.v1.config_base import apply_remote_configs, fetch_remote_config

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.utils import LayoutHints

logger = init_logger(__name__)
ENGINE_NAME = "vllm-instance"

# Thread-safe singleton storage
_config_instance: LMCacheEngineConfig | None = None
_config_lock = threading.Lock()


def is_false(value: str) -> bool:
    """Check if the given string value is equivalent to 'false'."""
    return value.lower() in ("false", "0", "no", "n", "off")


def vllm_layout_hints() -> "LayoutHints":
    """Build layout_hints dict by querying vLLM at runtime."""
    hints: dict[str, str] = {}
    kv_layout = try_get_vllm_kv_cache_layout()
    if kv_layout is not None:
        hints["kv_layout"] = kv_layout
    return hints  # type: ignore[return-value]


def try_get_vllm_kv_cache_layout() -> Literal["NHD", "HND"] | None:
    """Try to query the KV cache layout from vLLM at runtime.

    Returns ``"NHD"`` or ``"HND"`` if vLLM is available and the layout
    has been configured, otherwise ``None``.

    Please only call this where vllm is available (i.e. not in the MP server)
    We will print an error if we try to get vllm kv layout where vllm
    is not available.
    """

    # Third Party
    try:
        # Third Party
        from vllm.v1.attention.backends.utils import (  # type: ignore[import-untyped]
            get_kv_cache_layout,
        )

        return get_kv_cache_layout()
    except Exception:
        logger.error(
            "vLLM is not available but tried to query kv cache "
            "layout information, cannot get KV cache layout"
        )
        return None


def lmcache_get_or_create_config() -> LMCacheEngineConfig:
    """Get the LMCache configuration from the environment variable
    `LMCACHE_CONFIG_FILE`. If the environment variable is not set, this
    function will return the default configuration.

    This function is thread-safe and implements singleton pattern,
    ensuring the configuration is loaded only once.

    After loading the configuration, if 'remote_config_url' is configured,
    this function will attempt to fetch additional configuration from the
    remote config service. The current config and LMCACHE environment
    variables will be sent to the service, along with 'appid' if set.
    """
    global _config_instance

    # Double-checked locking for thread-safe singleton
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:  # Check again within lock
                if "LMCACHE_CONFIG_FILE" not in os.environ:
                    logger.warning(
                        "No LMCache configuration file is set. Trying to read"
                        " configurations from the environment variables."
                    )
                    logger.warning(
                        "You can set the configuration file through "
                        "the environment variable: LMCACHE_CONFIG_FILE"
                    )
                    _config_instance = LMCacheEngineConfig.from_env()
                    # from_env() doesn't call validate(); the file path
                    # gets it via update_config_from_env() below, but the
                    # env-only path needs an explicit call.
                    _config_instance.validate()
                else:
                    config_file = os.environ["LMCACHE_CONFIG_FILE"]
                    logger.info(f"Loading LMCache config file {config_file}")
                    _config_instance = LMCacheEngineConfig.from_file(config_file)
                    # Update config from environment variables
                    _config_instance.update_config_from_env()

                # Fetch and apply remote configuration if configured
                remote_config_url = _config_instance.remote_config_url
                if remote_config_url:
                    logger.info(
                        "Fetching remote configuration from %s", remote_config_url
                    )
                    app_id = _config_instance.app_id
                    remote_response = fetch_remote_config(
                        remote_config_url, app_id, _config_instance
                    )
                    if remote_response:
                        _config_instance = apply_remote_configs(
                            _config_instance, remote_response
                        )
                    else:
                        logger.warning(
                            "Failed to fetch remote configuration from %s. "
                            "Using local configuration only.",
                            remote_config_url,
                        )
    return _config_instance


def create_lmcache_ec_config() -> LMCacheEngineConfig:
    """Create EC config from LMCache config plus EC-specific overrides."""
    return load_ec_engine_config(base_config=lmcache_get_or_create_config())


_MM_SURROGATE_NAMESPACE_BIT = 1 << 62
_MM_SURROGATE_VALUE_MASK = _MM_SURROGATE_NAMESPACE_BIT - 1
_MM_SURROGATE_DOMAIN = b"LMCache:mm-token:v1"
_MM_TEXT_SURROGATE_DOMAIN = b"LMCache:mm-conditioned-text:v1"
_MM_CONTEXT_DOMAIN = b"LMCache:mm-context:v1"


def _normalize_mm_identifier(value: object) -> str | None:
    """Normalize a multimodal identifier, returning ``None`` if unusable."""
    if value is None:
        return None
    value_str = str(value).strip()
    if not value_str:
        return None
    return value_str


def hex_hash_to_int64(value: object) -> int:
    """Convert a multimodal identifier into a namespace-safe surrogate.

    Args:
        value: Identifier supplied by vLLM.

    Returns:
        A deterministic integer in the signed-int64-safe multimodal namespace
        ``2**62 <= value < 2**63``. This is equivalent to
        ``mm_token_surrogate_id(value, 0)`` and exists for compatibility with
        older helper call sites.

    Raises:
        ValueError: If ``value`` is missing or empty.
    """
    return mm_token_surrogate_id(value, 0)


def mm_token_surrogate_id(identifier: object, local_offset: int) -> int:
    """Build a VLM cache-token surrogate for one placeholder position.

    Args:
        identifier: Stable multimodal identifier supplied by vLLM.
        local_offset: Token offset inside this placeholder span.

    Returns:
        A deterministic positive int64 value in LMCache's multimodal surrogate
        namespace. The same identifier and local offset produce the same value;
        different identifiers or local offsets produce different values with
        cryptographic-hash collision probability.

    Raises:
        ValueError: If ``identifier`` is missing/empty or ``local_offset`` is
            negative.
    """
    normalized = _normalize_mm_identifier(identifier)
    if normalized is None:
        raise ValueError("multimodal identifier is missing or empty")
    if local_offset < 0:
        raise ValueError("local_offset must be non-negative")

    identifier_bytes = normalized.encode("utf-8")
    h = hashlib.sha256()
    h.update(_MM_SURROGATE_DOMAIN)
    h.update(len(identifier_bytes).to_bytes(8, byteorder="big", signed=False))
    h.update(identifier_bytes)
    h.update(local_offset.to_bytes(8, byteorder="big", signed=False))
    raw = int.from_bytes(h.digest()[:8], byteorder="big", signed=False)
    return _MM_SURROGATE_NAMESPACE_BIT | (raw & _MM_SURROGATE_VALUE_MASK)


def _mm_context_salt(identifiers: Sequence[str]) -> bytes:
    h = hashlib.sha256()
    h.update(_MM_CONTEXT_DOMAIN)
    for identifier in identifiers:
        identifier_bytes = identifier.encode("utf-8")
        h.update(len(identifier_bytes).to_bytes(8, byteorder="big", signed=False))
        h.update(identifier_bytes)
    return h.digest()


def _mm_text_surrogate_id(context_salt: bytes, token_id: int, position: int) -> int:
    h = hashlib.sha256()
    h.update(_MM_TEXT_SURROGATE_DOMAIN)
    h.update(context_salt)
    h.update(int(token_id).to_bytes(8, byteorder="big", signed=True))
    h.update(position.to_bytes(8, byteorder="big", signed=False))
    raw = int.from_bytes(h.digest()[:8], byteorder="big", signed=False)
    return _MM_SURROGATE_NAMESPACE_BIT | (raw & _MM_SURROGATE_VALUE_MASK)


def hex_hash_to_int16(value: object) -> int:
    """Convert a multimodal identifier into a compatibility 16-bit value.

    Args:
        value: Identifier supplied by vLLM.

    Returns:
        The low 16 bits of :func:`hex_hash_to_int64`.
    """
    return hex_hash_to_int64(value) & 0xFFFF


def _build_mm_rewrite_spans(
    n_tokens: int,
    mm_hashes: Sequence[object],
    mm_positions: Sequence["PlaceholderRange"],
) -> list[tuple[int, int, list[int]]] | None:
    """Build safe multimodal rewrite spans or ``None`` to fail closed."""
    if len(mm_hashes) != len(mm_positions):
        return None

    spans: list[tuple[int, int, list[int]]] = []
    for hash_str, placeholder in zip(mm_hashes, mm_positions, strict=False):
        normalized = _normalize_mm_identifier(hash_str)
        if normalized is None:
            return None
        start, length = placeholder.offset, placeholder.length
        if start < 0 or length < 0:
            return None
        if start >= n_tokens:
            continue
        end = min(start + length, n_tokens)
        if end <= start:
            continue
        values = [mm_token_surrogate_id(normalized, i) for i in range(end - start)]
        spans.append((start, end, values))
    return spans


def _rewrite_mm_cache_token_list(
    token_ids: list[int],
    mm_hashes: Sequence[object],
    mm_positions: Sequence["PlaceholderRange"],
) -> list[int] | None:
    if len(mm_hashes) != len(mm_positions):
        return None

    entries: list[tuple[int, int, str]] = []
    for hash_str, placeholder in zip(mm_hashes, mm_positions, strict=False):
        normalized = _normalize_mm_identifier(hash_str)
        if normalized is None:
            return None
        start, length = placeholder.offset, placeholder.length
        if start < 0 or length < 0:
            return None
        if start >= len(token_ids) or length == 0:
            continue
        end = min(start + length, len(token_ids))
        if end > start:
            entries.append((start, end, normalized))

    if not entries:
        return list(token_ids)

    entries.sort(key=lambda item: item[0])
    active_identifiers: list[str] = []
    rewritten = list(token_ids)
    cursor = 0
    for start, end, identifier in entries:
        if active_identifiers:
            context_salt = _mm_context_salt(active_identifiers)
            for pos in range(cursor, start):
                rewritten[pos] = _mm_text_surrogate_id(
                    context_salt, rewritten[pos], pos
                )
        for pos in range(start, end):
            rewritten[pos] = mm_token_surrogate_id(identifier, pos - start)
        active_identifiers.append(identifier)
        cursor = end

    if active_identifiers:
        context_salt = _mm_context_salt(active_identifiers)
        for pos in range(cursor, len(rewritten)):
            rewritten[pos] = _mm_text_surrogate_id(context_salt, rewritten[pos], pos)

    return rewritten


def apply_mm_hashes_to_token_ids(
    token_ids: torch.Tensor,
    mm_hashes: Sequence[object],
    mm_positions: Sequence["PlaceholderRange"],
) -> torch.Tensor | None:
    """Rewrite multimodal cache identity tokens in place.

    Args:
        token_ids: Token tensor to rewrite. The tensor must be able to hold
            signed int64-safe surrogate ids.
        mm_hashes: Multimodal identifiers parallel to ``mm_positions``.
        mm_positions: Placeholder spans to rewrite.

    Returns:
        The same tensor object, rewritten in place, or ``None`` when the
        multimodal metadata is unsafe to use for cache identity. Placeholder
        spans use per-position media surrogate ids; following text tokens are
        salted with the preceding multimodal context so sparse CacheBlend chunks
        do not cross-match across different media.
    """
    rewritten = _rewrite_mm_cache_token_list(
        token_ids.tolist(), mm_hashes, mm_positions
    )
    if rewritten is None:
        return None
    token_ids[:] = torch.tensor(
        rewritten, dtype=token_ids.dtype, device=token_ids.device
    )
    return token_ids


def apply_mm_hashes_to_token_list(
    token_ids: list[int],
    mm_hashes: Sequence[object],
    mm_positions: Sequence["PlaceholderRange"],
) -> list[int] | None:
    """Return cache token ids with multimodal-aware identity rewriting.

    Args:
        token_ids: Raw token ids from vLLM.
        mm_hashes: Multimodal identifiers parallel to ``mm_positions``.
        mm_positions: Placeholder spans to rewrite.

    Returns:
        A new list where placeholder spans contain stable int64-safe
        multimodal surrogate ids and following text tokens are salted with
        preceding multimodal context. The input list is not modified. Returns
        ``None`` when the multimodal metadata is unsafe to use for cache
        identity.
    """
    return _rewrite_mm_cache_token_list(token_ids, mm_hashes, mm_positions)


def has_mm_metadata(request: "Request") -> bool:
    """Return whether the request carries any multimodal metadata fields."""
    if getattr(request, "mm_features", None):
        return True
    return bool(
        getattr(request, "mm_hashes", None) or getattr(request, "mm_positions", None)
    )


def try_get_mm_aware_token_ids(request: "Request") -> list[int] | None:
    """Return safe cache token ids, or ``None`` when VLM identity is unsafe.

    Args:
        request: vLLM request exposing ``all_token_ids`` and optionally
            ``mm_features`` or legacy ``mm_hashes`` / ``mm_positions``.

    Returns:
        A new token-id list suitable for LMCache cache keys and CacheBlend
        matching. Returns ``None`` if the request carries multimodal metadata
        but LMCache cannot safely rewrite the placeholder spans.
    """
    token_ids = list(request.all_token_ids)
    mm_hashes, mm_positions = extract_mm_features(request)
    if not mm_hashes or not mm_positions:
        return None if has_mm_metadata(request) else token_ids
    return apply_mm_hashes_to_token_list(token_ids, mm_hashes, mm_positions)


def has_unsafe_mm_metadata(request: "Request") -> bool:
    """Return whether multimodal metadata exists but cannot form safe cache keys."""
    return has_mm_metadata(request) and try_get_mm_aware_token_ids(request) is None


def get_mm_aware_token_ids(request: "Request") -> list[int]:
    """Return request token ids with multimodal placeholders rewritten.

    Args:
        request: vLLM request exposing ``all_token_ids`` and optionally
            ``mm_features`` or legacy ``mm_hashes`` / ``mm_positions``.

    Returns:
        A new token-id list suitable for LMCache cache keys and CacheBlend
        matching.

    Raises:
        ValueError: If the request contains multimodal metadata that cannot be
            safely rewritten.
    """
    token_ids = try_get_mm_aware_token_ids(request)
    if token_ids is None:
        raise ValueError("unsafe multimodal cache identity metadata")
    return token_ids


def mla_enabled(model_config: "ModelConfig") -> bool:
    return (
        hasattr(model_config, "use_mla")
        and isinstance(model_config.use_mla, bool)
        and model_config.use_mla
    )


def create_lmcache_metadata(
    vllm_config=None,
    model_config=None,
    parallel_config=None,
    cache_config=None,
    role=None,
):
    """
    Create LMCacheMetadata from vLLM configuration.

    This function extracts common metadata creation logic that was duplicated
    across multiple files.

    Args:
        vllm_config: vLLM configuration object containing model, parallel, and
                    cache configs (alternative to individual config parameters)
        model_config: Model configuration (alternative to vllm_config)
        parallel_config: Parallel configuration (alternative to vllm_config)
        cache_config: Cache configuration (alternative to vllm_config)

    Returns:
        tuple: (LMCacheMetadata, LMCacheEngineConfig)
    """
    # Third Party
    # Try to import from old location before merged https://github.com/vllm-project/vllm/pull/26908
    try:
        # Third Party
        from vllm.utils.torch_utils import get_kv_cache_torch_dtype
    except ImportError:
        # Third Party
        from vllm.utils import get_kv_cache_torch_dtype
    # First Party
    from lmcache.v1.metadata import LMCacheMetadata

    config = lmcache_get_or_create_config()
    # Support both vllm_config object and individual config parameters
    if vllm_config is not None:
        model_cfg = vllm_config.model_config
        parallel_cfg = vllm_config.parallel_config
        cache_cfg = vllm_config.cache_config
    else:
        model_cfg = model_config
        parallel_cfg = parallel_config
        cache_cfg = cache_config

    # Get KV cache dtype
    kv_dtype = get_kv_cache_torch_dtype(cache_cfg.cache_dtype, model_cfg.dtype)

    # Check if MLA is enabled
    use_mla = mla_enabled(model_cfg)

    # Construct KV shape (for memory pool)
    num_layer = model_cfg.get_num_layers(parallel_cfg)
    chunk_size = config.chunk_size
    num_kv_head = model_cfg.get_num_kv_heads(parallel_cfg)
    head_size = model_cfg.get_head_size()
    kv_shape = (num_layer, 1 if use_mla else 2, chunk_size, num_kv_head, head_size)

    # Extract engine_id and kv_connector_extra_config from vllm_config if available
    engine_id = None
    kv_connector_extra_config = None
    if vllm_config is not None and hasattr(vllm_config, "kv_transfer_config"):
        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            engine_id = getattr(kv_transfer_config, "engine_id", None)
            kv_connector_extra_config = getattr(
                kv_transfer_config, "kv_connector_extra_config", None
            )

    # Create metadata
    metadata = LMCacheMetadata(
        model_name=model_cfg.model,
        world_size=parallel_cfg.world_size,
        local_world_size=parallel_cfg.world_size,
        worker_id=parallel_cfg.rank,
        local_worker_id=parallel_cfg.rank,
        kv_dtype=kv_dtype,
        kv_shape=kv_shape,
        use_mla=use_mla,
        role=role,
        served_model_name=model_cfg.served_model_name,
        engine_id=engine_id,
        kv_connector_extra_config=kv_connector_extra_config,
    )

    return metadata, config


def extract_mm_features(
    request: "Request", modify: bool = False
) -> tuple[list[str], list["PlaceholderRange"]]:
    """
    Normalize multimodal information from a Request into parallel lists.

    This helper reads either:
      1) `request.mm_features` (objects each exposing `.identifier` and
      `.mm_position`), or
      2) legacy fields `request.mm_hashes` and `request.mm_positions`.

    It returns two equally sized lists: the multimodal hash identifiers and their
    corresponding positions. If the request contains no multimodal info, it returns
    `([], [])`.

    Args:
        request (Request): The source object.
        modify (bool):
            Controls copy semantics for the legacy-path return values.
            - If True and legacy fields are used, shallow-copies are returned so
              the caller can mutate the lists without affecting `request`.
            - If False, the original legacy sequences are returned as-is
              (zero-copy); treat them as read-only.

    Returns:
        Tuple[list[str], list[PlaceholderRange]]: (`mm_hashes`, `mm_positions`).
        May be `([], [])` when no multimodal data is present.
    """
    if getattr(request, "mm_features", None):
        mm_hashes, mm_positions = zip(
            *((f.identifier, f.mm_position) for f in request.mm_features), strict=False
        )
        return (list(mm_hashes), list(mm_positions))
    elif hasattr(request, "mm_hashes") or hasattr(request, "mm_positions"):
        mm_hashes = getattr(request, "mm_hashes", None)
        mm_positions = getattr(request, "mm_positions", None)
        if modify:
            return (
                mm_hashes.copy() if mm_hashes is not None else [],
                mm_positions.copy() if mm_positions is not None else [],
            )
        else:
            return (
                mm_hashes if mm_hashes is not None else [],
                mm_positions if mm_positions is not None else [],
            )
    else:
        return ([], [])


def get_size_bytes(shapes: list[torch.Size], kv_dtypes: list[torch.dtype]):
    """
    Calculate the size in bytes with the given shapes and dtypes.
    """
    assert len(shapes) == len(kv_dtypes), (
        f"shapes and dtypes must have the same length, "
        f"but got {len(shapes)} and {len(kv_dtypes)}"
    )
    return sum(
        shape.numel() * kv_dtype.itemsize
        for shape, kv_dtype in zip(shapes, kv_dtypes, strict=True)
    )


def calculate_local_rank_and_world_size(vllm_config: "VllmConfig") -> tuple[int, int]:
    """
    Calculate the local worker id and local world size.

    Current assumption (TODO: add custom logic in the future):
    - Tensor Parallel is intra-node
    - Pipeline Parallel is inter-node

    Returns:
        Tuple[int, int]: (local_worker_id, local_world_size)
    """
    # First Party
    from lmcache import torch_dev

    parallel_config = vllm_config.parallel_config
    global_rank = parallel_config.rank
    global_world_size = parallel_config.world_size
    num_gpus = torch_dev.device_count()
    if global_world_size <= num_gpus:
        # single node case
        return parallel_config.rank, parallel_config.world_size
    else:
        tp_size = parallel_config.tensor_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        local_world_size = global_world_size // pp_size
        assert local_world_size == tp_size, (
            "LMCache is operating under the assumption that the "
            "local world size is equal to the tensor parallel size "
            "in multi-node deployment."
        )
        local_worker_id = global_rank % local_world_size
        return local_worker_id, local_world_size


def validate_mla_config(config: LMCacheEngineConfig, use_mla: bool) -> None:
    """Validate MLA-related configuration."""
    if use_mla and (config.remote_serde != "naive" and config.remote_serde is not None):
        raise ValueError("MLA only works with naive serde mode..")

    if use_mla and config.use_layerwise and config.enable_blending:
        raise ValueError(
            "We haven't supported MLA with Cacheblend yet. Please disable blending."
        )


def calculate_draft_layers(vllm_config: "VllmConfig") -> int:
    """Calculate the number of draft layers for speculative decoding."""
    assert vllm_config is not None, "vllm_config required for vLLM mode"

    num_draft_layers = 0
    model_config = vllm_config.model_config

    if vllm_config.speculative_config is not None:
        logger.info(
            "vllm_config.speculative_config: %s", vllm_config.speculative_config
        )
        if vllm_config.speculative_config.method == "deepseek_mtp":
            num_draft_layers = getattr(
                model_config.hf_config, "num_nextn_predict_layers", 0
            )
        elif vllm_config.speculative_config.use_eagle():
            try:
                draft_model_config = vllm_config.speculative_config.draft_model_config
                num_draft_layers = draft_model_config.get_num_layers(
                    vllm_config.parallel_config
                )
                logger.info("EAGLE detected %d extra layer(s)", num_draft_layers)
            except Exception:
                logger.info(
                    "EAGLE detected, but failed to get the number of extra layers"
                    "falling back to 1"
                )
                num_draft_layers = 1
    return num_draft_layers


def is_dp_rank0(vllm_config: "VllmConfig") -> bool:
    return vllm_config.parallel_config.data_parallel_rank_local == 0
