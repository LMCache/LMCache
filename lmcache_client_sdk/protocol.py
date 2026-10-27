# SPDX-License-Identifier: Apache-2.0
"""Wire protocol types for the engine-driven transfer path.

These types mirror the frozen contracts in
``lmcache.v1.multiprocess.custom_types``. They are serialized with msgspec
over the ZMQ transport and must stay byte-compatible with the server.
"""

# Standard
from dataclasses import dataclass, field

# Third Party
import msgspec


class GroupLayout(msgspec.Struct):
    """Per-LMCache-group layout for multi-group engine-driven registration.

    Attributes:
        num_layers: Number of KV layers in this group.
        hidden_dim_size: Flattened hidden dimension per token for this group.
        dtype_str: Torch dtype name for this group's KV tensors.
        tokens_per_block: Tokens covered by one paged block of this group.
        window_tokens: Tokens stored per LMCache chunk for this group.
        kv_size: Object planes for this group (``2`` for separate K/V,
            ``1`` for single-plane formats).
        window_chunks: Trailing LMCache chunks this group needs for a hit.
        recurrent: Whether the group's pages hold recurrent state snapshots.
    """

    num_layers: int
    hidden_dim_size: int
    dtype_str: str
    tokens_per_block: int
    window_tokens: int = 0
    kv_size: int = 0
    window_chunks: int = 0
    recurrent: bool = False


class RegisterEngineDrivenContextPayload(msgspec.Struct):
    """Payload for the REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT message.

    Attributes:
        instance_id: Worker instance identifier (typically PID).
        model_name: Model name associated with this worker.
        world_size: Worker world size used in cache keys.
        block_size: Tokens per paged block.
        num_layers: Number of model layers.
        hidden_dim_size: Flattened hidden dimension per token.
        dtype_str: Torch dtype name (e.g. ``"float16"``).
        use_mla: Whether the worker KV format is single-plane (MLA / fused).
        num_physical_slots: Number of physical KV slots gathered into one
            LMCache chunk. ``None`` accepts the legacy protocol.
        group_layouts: Per-group layouts for multi-group workers. Empty
            means legacy single-group.
    """

    instance_id: int
    model_name: str
    world_size: int
    block_size: int
    num_layers: int
    hidden_dim_size: int
    dtype_str: str
    use_mla: bool
    num_physical_slots: int | None = None
    group_layouts: list[GroupLayout] = []


@dataclass
class RegisterEngineDrivenContextResponse:
    """Shared response for engine-driven context registration.

    Attributes:
        shm_name: Name of the server's SHM-backed L1 pool, or ``""``.
        pool_size: Size of that pool in bytes.
        null_block_id: Engine block ID the server treats as absent KV data.
    """

    shm_name: str = ""
    pool_size: int = 0
    null_block_id: int = 0


@dataclass
class PrepareStoreResponse:
    """Shared response for an engine-driven store preparation."""

    context: dict = field(default_factory=dict)


@dataclass
class PrepareRetrieveResponse:
    """Shared response for an engine-driven retrieve preparation.

    Attributes:
        success: Whether the server found the object.
        data: Inline serialized tensor envelope bytes on a hit.
        context: Opaque server context (unused on the inline path).
    """

    success: bool
    data: bytes = b""
    context: dict = field(default_factory=dict)
