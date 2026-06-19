# SPDX-License-Identifier: Apache-2.0
"""Shared request/response schemas for the mp coordinator REST API.

These Pydantic models are the wire contract between the coordinator and mp
servers. The coordinator uses them to validate requests and shape responses; an
mp server (when it registers) imports the same models to build its request
bodies and parse replies, so both sides agree on the schema in one place.
"""

# Standard
from enum import Enum
from typing import Annotated
import base64

# Third Party
from pydantic import BaseModel, Field, StringConstraints, field_validator
import numpy as np

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey  # noqa: F401  re-exported


def encode_tokens(tokens: "list[int] | np.ndarray") -> str:
    """Encode token ids into a compact base64 wire string.

    Token ids fit in ``uint32``, so a little-endian ``uint32`` buffer is far
    smaller than a JSON integer list and decodes in one ``np.frombuffer`` call.

    Args:
        tokens: Token ids (a ``list[int]`` or any array castable to ``uint32``).

    Returns:
        Base64 of the little-endian ``uint32`` token buffer.
    """
    arr = np.ascontiguousarray(np.asarray(tokens, dtype="<u4"))
    return base64.b64encode(arr.tobytes()).decode("ascii")


def decode_tokens(tokens_b64: str) -> np.ndarray:
    """Decode a base64 token string produced by :func:`encode_tokens`.

    Args:
        tokens_b64: Base64 of a little-endian ``uint32`` token buffer.

    Returns:
        A ``uint64`` array of token ids (widened so it feeds the hashers
        directly).

    Raises:
        ValueError: If ``tokens_b64`` is not valid base64 or not a multiple of
            4 bytes.
    """
    try:
        raw = base64.b64decode(tokens_b64, validate=True)
    except Exception as exc:
        raise ValueError(f"tokens_b64 is not valid base64: {exc}") from exc
    if len(raw) % 4 != 0:
        raise ValueError(
            f"tokens_b64 byte length {len(raw)} is not a multiple of 4 "
            "(malformed uint32 token buffer)"
        )
    return np.frombuffer(raw, dtype="<u4").astype(np.uint64)


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
        p2p_advertised_url: URL the instance advertises for peer-to-peer
            transfers. Optional -- empty when the instance does not participate
            in P2P.
        mq_port: Port of the instance's ZMQ message-queue server that P2P peers
            send lookup/unlock RPCs to, reachable at the instance's ``ip``.
            Optional -- 0 when P2P is disabled.
    """

    instance_id: Annotated[str, StringConstraints(strip_whitespace=True)] = ""
    ip: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    http_port: int = Field(ge=1, le=65535)
    metadata: dict[str, str] = Field(default_factory=dict)
    p2p_advertised_url: Annotated[str, StringConstraints(strip_whitespace=True)] = ""
    mq_port: int = Field(default=0, ge=0, le=65535)


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


class EventType(str, Enum):
    """L2 cache events reported by an MP server."""

    STORE = "store"
    LOOKUP = "lookup"
    DELETE = "delete"


class UsageEvent(BaseModel):
    """A single L2 event reported by an MP server.

    Attributes:
        type: The event type.
        key: The cache key this event applies to.
        bytes: Bytes stored (``store`` only; ``0`` for other types).
    """

    type: EventType
    key: EncodedObjectKey
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


# -- Global CacheBlend fingerprint directory ------------------------------


class StoreRangeModel(BaseModel):
    """Wire form of one published stored token range (see ``StoreRange``).

    The coordinator chunks ``tokens`` at its chunk size and hashes each chunk;
    chunk ``i`` maps to ``object_keys[i]`` at ``old_st_base + i * chunk_size``.

    Attributes:
        model_scope: Reuse scope (the model name).
        tokens: The stored tokens (``token_ids[start:end]``).
        object_keys: Shared-L2 storage key (hex) per chunk, in order.
        old_st_base: Token position of the range's first token.
    """

    model_scope: str
    tokens: list[int] = Field(default_factory=list)
    object_keys: list[str] = Field(default_factory=list)
    old_st_base: int = Field(ge=0)


class BlendFingerprintRequest(BaseModel):
    """Body of ``POST /blend/fingerprints``: register stored ranges.

    Attributes:
        ranges: Stored token ranges to register (idempotent).
    """

    ranges: list[StoreRangeModel] = Field(default_factory=list)


class BlendFingerprintResponse(BaseModel):
    """Reply to ``POST /blend/fingerprints``.

    Attributes:
        inserted: Number of fingerprints newly registered.
    """

    inserted: int


class BlendEvictRequest(BaseModel):
    """Body of ``DELETE /blend/fingerprints``: evict by storage key.

    Attributes:
        object_keys: ``object_key`` values to evict.
    """

    object_keys: list[str] = Field(default_factory=list)


class BlendEvictResponse(BaseModel):
    """Reply to ``DELETE /blend/fingerprints``.

    Attributes:
        removed: Number of fingerprint entries evicted.
    """

    removed: int


class BlendMatchRequest(BaseModel):
    """Body of ``POST /blend/match``.

    Attributes:
        model_scope: Scope to match within.
        tokens_b64: The request tokens, packed via :func:`encode_tokens`
            (base64 little-endian ``uint32``).
    """

    model_scope: str
    tokens_b64: str = ""

    @field_validator("tokens_b64")
    @classmethod
    def _validate_tokens_b64(cls, value: str) -> str:
        """Reject a malformed token buffer at request validation.

        Without this, ``decode_tokens`` would raise ``ValueError`` inside the
        route handler, which FastAPI surfaces as a 500 (server error) for what
        is really bad client input. Validating here returns a 422 instead.

        Args:
            value: The base64 ``tokens_b64`` field.

        Returns:
            The unchanged value once it is confirmed decodable.

        Raises:
            ValueError: If ``value`` is not valid base64 or not a whole number
                of ``uint32`` tokens (surfaced by FastAPI as 422).
        """
        decode_tokens(value)
        return value


class GlobalMatchModel(BaseModel):
    """Wire form of one matched chunk (see ``GlobalMatch``).

    Attributes:
        object_key: Shared-L2 storage key of the matched chunk.
        old_st: Token position in the stored sequence (re-RoPE source).
        cur_st: Token position in the request (re-RoPE target).
    """

    object_key: str
    old_st: int
    cur_st: int


class BlendMatchResponse(BaseModel):
    """Reply to ``POST /blend/match``.

    Attributes:
        matches: Matched chunks, ascending by ``cur_st``.
    """

    matches: list[GlobalMatchModel]
