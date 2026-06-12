# SPDX-License-Identifier: Apache-2.0
"""Shared request/response schemas for the mp coordinator REST API.

These Pydantic models are the wire contract between the coordinator and mp
servers. The coordinator uses them to validate requests and shape responses; an
mp server (when it registers) imports the same models to build its request
bodies and parse replies, so both sides agree on the schema in one place.
"""

# Standard
from dataclasses import dataclass
from enum import Enum
from typing import Annotated

# Third Party
from pydantic import BaseModel, Field, StringConstraints


class RegisterRequest(BaseModel):
    """Body of a ``POST /instances`` registration request.

    Attributes:
        instance_id: Identifier of the mp server. Optional -- if empty (or
            whitespace-only), the coordinator generates one and returns it.
        ip: IP/host of the mp server's HTTP server. Whitespace is stripped and a
            blank value is rejected, since the coordinator calls this address.
        http_port: Port of the mp server's HTTP server, which the coordinator
            calls to push work to this instance.
        metadata: Free-form registration hints.
    """

    instance_id: Annotated[str, StringConstraints(strip_whitespace=True)] = ""
    ip: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    http_port: int = Field(ge=1, le=65535)
    metadata: dict[str, str] = Field(default_factory=dict)


class RegisterResponse(BaseModel):
    """Reply to a successful ``POST /instances``.

    Attributes:
        instance_id: The registered instance's id.
        re_registered: ``True`` if this replaced an existing registration.
    """

    instance_id: str
    re_registered: bool


class HeartbeatResponse(BaseModel):
    """Reply to a successful ``PUT /instances/{id}/heartbeat``.

    Attributes:
        instance_id: The instance whose heartbeat was recorded.
    """

    instance_id: str


# -- Quota management --------------------------------------------------------


class SetQuotaRequest(BaseModel):
    """Body of ``PUT /quota/{cache_salt}``.

    Attributes:
        limit_gb: Non-negative byte budget in GiB.
    """

    limit_gb: float = Field(ge=0.0)


class QuotaResponse(BaseModel):
    """Reply to ``PUT`` or ``DELETE /quota/{cache_salt}``.

    Attributes:
        cache_salt: The tenant identifier (``_default`` for empty salt).
        limit_gb: The current limit in GiB (0.0 after deletion).
        status: ``"ok"`` or ``"removed"`` or ``"not_found"``.
    """

    cache_salt: str
    limit_gb: float
    status: str


# -- L2 usage tracking -------------------------------------------------------


@dataclass(frozen=True)
class CacheKey:
    """Lightweight, torch-free equivalent of ``ObjectKey``.

    Used both as the wire format in usage events and as the key type
    in the eviction controller's per-salt LRU. Frozen so it can be
    used as a dict key / in ``OrderedDict``.

    Attributes:
        chunk_hash_hex: Hex-encoded content hash of the chunk.
        model_name: Name of the model this chunk belongs to.
        kv_rank: Packed rank bitmap (world_size, global_rank, etc.).
        cache_salt: The tenant identifier.
    """

    chunk_hash_hex: str
    model_name: str
    kv_rank: int
    cache_salt: str


class EventType(str, Enum):
    """Type of L2 cache event."""

    STORE = "store"
    LOOKUP = "lookup"


class UsageEvent(BaseModel):
    """A single L2 store or lookup event reported by an MP server.

    Attributes:
        type: The event type.
        key: The cache key this event applies to. The tenant
            identifier (``cache_salt``) is carried inside the key.
        bytes: Number of bytes stored (for ``"store"`` events).
    """

    type: EventType
    key: CacheKey
    bytes: int = Field(ge=0)


class ReportUsageRequest(BaseModel):
    """Body of ``POST /l2/events``.

    Attributes:
        instance_id: Identifier of the MP server that produced this batch.
        seq: Monotonically increasing sequence number scoped to this
            ``instance_id``. Starts at 1 for the first flush after the
            server starts.
        events: Batch of store/lookup events to record.
    """

    instance_id: str
    seq: int = Field(ge=1)
    events: list[UsageEvent]


class ReportUsageResponse(BaseModel):
    """Reply to ``POST /l2/events``.

    Attributes:
        recorded: Number of events processed.
    """

    recorded: int


class L2StatusResponse(BaseModel):
    """Combined quota and usage for a single ``cache_salt``.

    Attributes:
        cache_salt: The tenant identifier.
        quota_limit_gb: The byte budget in GiB (0.0 if no quota set).
        quota_exists: Whether an explicit quota is registered.
        usage_gb: Current L2 usage in GiB.
    """

    cache_salt: str
    quota_limit_gb: float
    quota_exists: bool
    usage_gb: float


class L2StatusListResponse(BaseModel):
    """Reply to ``GET /l2/status``.

    Attributes:
        total_gb: Aggregate L2 usage in GiB.
        by_cache_salt: Per-tenant breakdown with quota and usage.
    """

    total_gb: float
    by_cache_salt: list[L2StatusResponse]
