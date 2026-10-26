# SPDX-License-Identifier: Apache-2.0
"""Runtime model layouts and fixed-slot sizing for DAX-Coordinated L1."""

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
    """Registered layouts with one fixed-stride pool per object group."""

    model_name: str
    model_family: str
    kv_world_size: int
    chunk_size: int
    payload_bytes: int
    layout_digest: bytes
    dtype_names: tuple[str, ...]
    group_payload_bytes: tuple[int, ...]


def classify_devdax_model_family(model_name: str) -> str:
    """Return a diagnostic family label for an exact serving model name.

    Args:
        model_name: Exact model ID or local model path supplied by the serving
            engine.

    Returns:
        ``"llama"``, ``"qwen3"``, or ``"runtime"`` for other names.

    Raises:
        ValueError: If the name is empty. Layout admission does not depend
            on this label.
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
    return "runtime"


def resolve_devdax_model_profile(
    model_name: str,
    kv_world_size: int,
    chunk_size: int,
    layout_descs: list[MemoryLayoutDesc],
) -> DevDaxModelProfile:
    """Validate runtime object groups and size their payload pools.

    The serving engine and LMCache remain the sources of truth for chunk size
    and dtype. Device-DAX does not load model configuration or maintain a table
    of model-size constants: it sizes each group's fixed slot from its runtime
    ``MemoryLayoutDesc`` registered for the GPU cache.

    Args:
        model_name: Exact serving model name used in LMCache object keys.
        kv_world_size: Number of KV-cache rank slices registered with one
            LMCache server.
        chunk_size: LMCache tokens per object.
        layout_descs: Runtime layouts, one per LMCache object group.

    Returns:
        An immutable model profile containing the largest object byte size
        and a stable cross-host layout digest.

    Raises:
        ValueError: If the rank count, chunk size, object groups, or runtime layout
            cannot be represented safely by the fixed-slot arena.
    """
    family = classify_devdax_model_family(model_name)
    if (
        isinstance(kv_world_size, bool)
        or not isinstance(kv_world_size, int)
        or not 1 <= kv_world_size <= 8
    ):
        raise ValueError("DAX-Coordinated L1 supports kv_world_size=1..8")
    if chunk_size <= 0:
        raise ValueError("DAX-Coordinated L1 chunk_size must be positive")
    if not layout_descs:
        raise ValueError("DAX-Coordinated L1 requires at least one object group")

    group_payload_bytes = []
    for layout in layout_descs:
        if not layout.shapes or len(layout.shapes) != len(layout.dtypes):
            raise ValueError("DAX-Coordinated L1 received an invalid model layout")
        offset = 0
        for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True):
            if len(shape) != 4 or shape[0] not in (1, 2):
                raise ValueError(
                    "DAX-Coordinated L1 requires [K/V, layers, slots, hidden] "
                    "layouts with dimension 0 equal to 1 or 2"
                )
            if any(dimension <= 0 for dimension in shape):
                raise ValueError(
                    "DAX-Coordinated L1 model layout dimensions must be positive"
                )
            # Slots can be compressed or windowed; they need not equal tokens.
            # TensorMemoryObj concatenates bytes without inter-tensor padding.
            if offset % dtype.itemsize:
                raise ValueError(
                    "DAX-Coordinated L1 tensor offset is not dtype aligned"
                )
            offset += get_size_bytes([shape], [dtype])
        # Native reservation arguments and on-media lengths are uint32_t.
        if offset > (1 << 32) - 1:
            raise ValueError(
                "DAX-Coordinated L1 model payload must fit the native "
                f"uint32 byte length (maximum {(1 << 32) - 1}); got {offset}"
            )
        group_payload_bytes.append(offset)

    # Keep the maximum for diagnostics; geometry uses every group's size.
    payload_bytes = max(group_payload_bytes)
    dtype_names = tuple(
        str(dtype) for layout in layout_descs for dtype in layout.dtypes
    )
    canonical = {
        "schema": 2,
        "allocation": "group_fixed_slot_pools_v1",
        "model_name": model_name,
        "model_family": family,
        "kv_world_size": kv_world_size,
        "chunk_size": chunk_size,
        "layouts": [
            {
                "shapes": [list(shape) for shape in layout.shapes],
                "dtypes": [str(dtype) for dtype in layout.dtypes],
            }
            for layout in layout_descs
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
        group_payload_bytes=tuple(group_payload_bytes),
    )
