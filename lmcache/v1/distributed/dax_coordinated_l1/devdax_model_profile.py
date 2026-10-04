# SPDX-License-Identifier: Apache-2.0
"""PR1 model admission and fixed-slot sizing for DAX-Coordinated L1."""

# Standard
from dataclasses import dataclass
from hashlib import sha256
import json
import re

# First Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.v1.distributed.api import MemoryLayoutDesc

_QWEN3_MODEL_PATTERN = re.compile(
    r"^qwen[-_]?3(?:\.0)?(?=$|[-_])",
    re.IGNORECASE,
)
_LLAMA_MODEL_PATTERN = re.compile(
    r"^(?:meta[-_])?llama(?=$|[-_.])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DevDaxModelProfile:
    """One admitted homogeneous model layout for the PR1 fixed-slot arena."""

    model_name: str
    model_family: str
    kv_world_size: int
    chunk_size: int
    payload_bytes: int
    layout_digest: bytes
    dtype_names: tuple[str, ...]


def classify_devdax_model_family(model_name: str) -> str:
    """Return the supported PR1 family for an exact serving model name.

    Args:
        model_name: Exact model ID or local model path supplied by the serving
            engine.

    Returns:
        ``"llama"`` or ``"qwen3"``.

    Raises:
        ValueError: If the name is empty or is outside the PR1 allowlist.
    """
    normalized = model_name.strip()
    if not normalized:
        raise ValueError("DAX-Coordinated L1 requires a non-empty model name")
    lowered = normalized.lower()
    segments = [segment for segment in re.split(r"/|--", lowered) if segment]
    if any(_QWEN3_MODEL_PATTERN.search(segment) for segment in segments):
        return "qwen3"
    if any(_LLAMA_MODEL_PATTERN.search(segment) for segment in segments):
        return "llama"
    raise ValueError(
        "DAX-Coordinated L1 PR1 supports only Llama and Qwen3 model names; "
        f"got {model_name!r}"
    )


def resolve_devdax_model_profile(
    model_name: str,
    kv_world_size: int,
    chunk_size: int,
    layout_descs: list[MemoryLayoutDesc],
    *,
    expected_kv_world_size: int = 1,
) -> DevDaxModelProfile:
    """Validate and size one runtime-registered PR1 model layout.

    The serving engine and LMCache remain the sources of truth for chunk size
    and dtype. Device-DAX does not load model configuration or maintain a table
    of model-size constants: it sizes the fixed slot from the complete runtime
    ``MemoryLayoutDesc`` registered for the GPU cache.

    Args:
        model_name: Exact serving model name used in LMCache object keys.
        kv_world_size: Number of KV-cache rank slices registered with one
            LMCache server.
        chunk_size: LMCache tokens per object.
        layout_descs: Runtime layouts, one per LMCache object group.
        expected_kv_world_size: Explicit local TP size from the placement
            configuration. Defaults to the existing TP=1 contract.

    Returns:
        An immutable model profile containing the complete object byte size
        and a stable cross-host layout digest.

    Raises:
        ValueError: If the model family, rank count, chunk size, object-group
            count, or runtime layout is outside the PR1 contract.
    """
    family = classify_devdax_model_family(model_name)
    if not 1 <= expected_kv_world_size <= 8:
        raise ValueError("DAX-Coordinated L1 supports local TP=1..8/PP=1")
    if kv_world_size != expected_kv_world_size:
        rank_description = (
            "one KV rank"
            if expected_kv_world_size == 1
            else f"{expected_kv_world_size} KV ranks"
        )
        raise ValueError(
            "DAX-Coordinated L1 requires "
            f"{rank_description} "
            f"(TP={expected_kv_world_size}/PP=1 with one LMCache server); "
            f"got {kv_world_size}"
        )
    if chunk_size <= 0:
        raise ValueError("DAX-Coordinated L1 chunk_size must be positive")
    if len(layout_descs) != 1:
        raise ValueError(
            "DAX-Coordinated L1 PR1 requires one homogeneous object group; "
            f"got {len(layout_descs)}"
        )

    layout = layout_descs[0]
    if not layout.shapes or len(layout.shapes) != len(layout.dtypes):
        raise ValueError("DAX-Coordinated L1 received an invalid model layout")
    if len(layout.shapes) != 1 or len(layout.shapes[0]) != 4:
        raise ValueError(
            "DAX-Coordinated L1 PR1 requires one standard "
            "[K/V, layers, tokens, hidden] KV layout"
        )
    shape = layout.shapes[0]
    # vLLM can expose the same complete KV payload in either of two runtime
    # representations. Older layouts use kv_size=2 with the per-K/V hidden
    # dimension, while current NHD layouts use kv_size=1 and fold K/V into a
    # doubled hidden dimension. The runtime MemoryLayoutDesc remains the byte
    # sizing authority, so admit both representations and bind their exact
    # shape into the persistent layout digest.
    if shape[0] not in (1, 2):
        raise ValueError(
            "DAX-Coordinated L1 PR1 requires a complete packed or separate "
            f"K/V payload (layout dimension 0 must be 1 or 2); got {shape[0]}"
        )
    if shape[2] != chunk_size:
        raise ValueError(
            "DAX-Coordinated L1 runtime layout token dimension must match "
            f"LMCache chunk_size={chunk_size}; got {shape[2]}"
        )
    if any(dimension <= 0 for dimension in shape):
        raise ValueError("DAX-Coordinated L1 model layout dimensions must be positive")
    payload_bytes = get_size_bytes(layout.shapes, layout.dtypes)
    if payload_bytes <= 0:
        raise ValueError("DAX-Coordinated L1 model payload must be positive")
    # Native reservation arguments and on-media payload lengths are uint32_t.
    # Reject before mapping or formatting an arena that cannot serve writes.
    if payload_bytes > (1 << 32) - 1:
        raise ValueError(
            "DAX-Coordinated L1 model payload must fit the native "
            f"uint32 byte length (maximum {(1 << 32) - 1}); got {payload_bytes}"
        )

    dtype_names = tuple(str(dtype) for dtype in layout.dtypes)
    canonical = {
        "schema": 1,
        "model_name": model_name,
        "model_family": family,
        "kv_world_size": kv_world_size,
        "chunk_size": chunk_size,
        "layouts": [
            {
                "shapes": [list(shape) for shape in layout.shapes],
                "dtypes": list(dtype_names),
            }
        ],
    }
    layout_digest = sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return DevDaxModelProfile(
        model_name=model_name,
        model_family=family,
        kv_world_size=kv_world_size,
        chunk_size=chunk_size,
        payload_bytes=payload_bytes,
        layout_digest=layout_digest,
        dtype_names=dtype_names,
    )
