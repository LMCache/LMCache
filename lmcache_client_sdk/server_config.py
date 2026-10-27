# SPDX-License-Identifier: Apache-2.0
"""Server configuration discovery over HTTP.

The client reads the server's ``/config`` and ``/status`` endpoints to learn
the chunk size and the registered KV cache geometry for a model. Geometry is
decoded from the server-reported ``engine_kv_concrete_shape`` string using
the same format-name template the server renders it from, so no native
extension or format registry is required.
"""

# Standard
from dataclasses import dataclass

# Third Party
import requests


class LMCacheClientError(RuntimeError):
    """Raised when a client-side operation fails."""


# Format tokens that render as literal axis values rather than geometry sizes.
_LITERAL_LABELS = frozenset({"ONE", "TWO", "NP"})
# Format tokens that appear in the bracketed inner shape.
_INNER_LABELS = frozenset(
    {"NB", "NL", "TWO", "BS", "NH", "HS", "CS", "NBBS", "BSV", "BSS", "ONE", "NP"}
)


@dataclass(frozen=True)
class ServerConfig:
    """Server-side multiprocess configuration.

    Attributes:
        chunk_size: Tokens per LMCache chunk.
        shm_name: Name of the server's SHM-backed L1 pool, or ``""``.
    """

    chunk_size: int
    shm_name: str = ""


@dataclass(frozen=True)
class CacheLayout:
    """Registered KV cache geometry for one model.

    Attributes:
        world_size: Number of TP workers the layout was registered for.
        num_kv_readers: Readers per stored object.
        num_layers: Number of model layers.
        block_size: Tokens per paged block.
        num_kv_heads: Number of KV heads (``1`` for single-plane formats).
        head_dim: Per-head dimension.
        hidden_dim_size: Flattened hidden dimension per token.
        dtype_str: Torch dtype name (e.g. ``"float16"``).
        use_mla: Whether the layout is single-plane (MLA / fused KV).
    """

    world_size: int
    num_kv_readers: int
    num_layers: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    hidden_dim_size: int
    dtype_str: str
    use_mla: bool


def _parse_concrete_shape(shape_str: str, format_name: str) -> tuple[int, int, int]:
    """Decode ``(block_size, num_kv_heads, head_dim)`` from a shape report.

    The server renders a format's bracketed shape from its enum name template
    (e.g. ``NL_X_TWO_NB_BS_NH_HS`` -> ``80 x [2, 2048, 16, 8, 128]``). This
    function walks the same template in reverse, mapping each dimension back
    to its label.

    Args:
        shape_str: The ``engine_kv_concrete_shape`` string from ``/status``.
        format_name: The ``engine_kv_format`` enum name for the kernel group.

    Returns:
        ``(block_size, num_kv_heads, head_dim)``. ``num_kv_heads`` is ``1``
        when the format has no ``NH`` axis (single-plane formats).

    Raises:
        LMCacheClientError: If the shape cannot be parsed or lacks a block
            axis.
    """
    inner = shape_str.split("[", 1)
    if len(inner) != 2:
        raise LMCacheClientError(f"malformed engine_kv_concrete_shape {shape_str!r}")
    inner_str = inner[1].rsplit("]", 1)[0]
    try:
        dims = [int(token.strip()) for token in inner_str.split(",")]
    except ValueError as exc:
        raise LMCacheClientError(
            f"non-numeric engine_kv_concrete_shape {shape_str!r}"
        ) from exc

    labels = format_name.split("_X_")[-1].split("_")
    if len(labels) != len(dims):
        raise LMCacheClientError(
            f"engine_kv_concrete_shape {shape_str!r} does not match "
            f"format {format_name!r}"
        )
    label_to_dim = {
        label: dim
        for label, dim in zip(labels, dims, strict=True)
        if label not in _LITERAL_LABELS and label in _INNER_LABELS
    }
    block_size = label_to_dim.get("BS")
    if block_size is None:
        raise LMCacheClientError(
            f"format {format_name!r} has no BS axis in {shape_str!r}"
        )
    num_kv_heads = label_to_dim.get("NH", 1)
    head_dim = label_to_dim.get("HS") or label_to_dim.get("CS")
    if head_dim is None:
        raise LMCacheClientError(
            f"format {format_name!r} has no HS/CS axis in {shape_str!r}"
        )
    return block_size, num_kv_heads, head_dim


def fetch_server_config(http_url: str, timeout: float) -> ServerConfig:
    """Fetch the server's multiprocess configuration.

    Args:
        http_url: HTTP endpoint URL of the LMCache server.
        timeout: Timeout in seconds for the HTTP requests.

    Returns:
        The server's chunk size and SHM pool name.

    Raises:
        LMCacheClientError: If the server does not expose a valid ``/config``.
    """
    try:
        response = requests.get(f"{http_url}/config", timeout=timeout)
        response.raise_for_status()
        mp_conf = response.json()["mp"]
    except (requests.RequestException, KeyError, ValueError) as exc:
        raise LMCacheClientError(
            f"failed to fetch server config from {http_url}/config"
        ) from exc
    chunk_size = int(mp_conf["chunk_size"])
    shm_name = str(mp_conf.get("shm_name", "")).lstrip("/")
    if shm_name and not shm_name.startswith("lmcache_l1_pool_"):
        shm_name = f"lmcache_l1_pool_{shm_name}"
    return ServerConfig(chunk_size=chunk_size, shm_name=shm_name)


def fetch_cache_layout(http_url: str, model_name: str, timeout: float) -> CacheLayout:
    """Fetch the registered KV cache layout for a model.

    Args:
        http_url: HTTP endpoint URL of the LMCache server.
        model_name: The model name the layout was registered under.
        timeout: Timeout in seconds for the HTTP requests.

    Returns:
        The decoded cache geometry.

    Raises:
        LMCacheClientError: If the server does not expose a valid ``/status``
            or no layout is registered for the model.
    """
    try:
        response = requests.get(f"{http_url}/status", timeout=timeout)
        response.raise_for_status()
        meta_conf = response.json().get("cache_context_meta", {})
    except (requests.RequestException, KeyError, ValueError) as exc:
        raise LMCacheClientError(
            f"failed to fetch server status from {http_url}/status"
        ) from exc

    entry = next(
        (e for e in meta_conf.values() if e.get("model_name") == model_name),
        None,
    )
    if entry is None:
        raise LMCacheClientError(
            f"no registered GPU layout for model_name={model_name!r}; "
            "register from a vLLM instance first"
        )

    try:
        world_size = int(entry.get("world_size", 1))
        num_kv_readers = int(entry.get("num_kv_readers", 1))
        kv_cache_layout = entry["kv_cache_layout"]
        num_layers = int(kv_cache_layout["num_layers"])
        kernel_groups = kv_cache_layout["kernel_groups"]
        if len(kernel_groups) != 1:
            raise LMCacheClientError(
                "currently not supporting hybrid models with multiple "
                f"kernel groups; found {len(kernel_groups)}"
            )
        kernel_group = kernel_groups[0]
        dtype_str = str(kernel_group["dtype"]).removeprefix("torch.")
        tokens_per_block = int(kernel_group["tokens_per_block"])
        use_mla = bool(kernel_group.get("is_mla", False))
        format_name = kernel_group["engine_kv_format"]
        shape_str = kernel_group["engine_kv_concrete_shape"]
    except (KeyError, TypeError, ValueError) as exc:
        raise LMCacheClientError(
            f"failed to decode KV cache layout for model_name={model_name!r}"
        ) from exc

    block_size, num_kv_heads, head_dim = _parse_concrete_shape(
        shape_str, format_name
    )
    if block_size != tokens_per_block:
        raise LMCacheClientError(
            f"decoded block_size {block_size} != tokens_per_block "
            f"{tokens_per_block} for model_name={model_name!r}"
        )
    hidden_dim_size = head_dim if use_mla else num_kv_heads * head_dim
    return CacheLayout(
        world_size=world_size,
        num_kv_readers=num_kv_readers,
        num_layers=num_layers,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_dim_size=hidden_dim_size,
        dtype_str=dtype_str,
        use_mla=use_mla,
    )
