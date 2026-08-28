# SPDX-License-Identifier: Apache-2.0
"""Shared request/response schemas for the mp coordinator REST API.

These Pydantic models are the wire contract between the coordinator and mp
servers. The coordinator uses them to validate requests and shape responses; an
mp server (when it registers) imports the same models to build its request
bodies and parse replies, so both sides agree on the schema in one place.

This module holds HTTP models only.
"""

# Standard
from typing import Annotated

# Third Party
from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey  # noqa: F401  re-exported
from lmcache.v1.distributed.api import Tier
from lmcache.v1.mp_coordinator.api import CacheEventBatch
from lmcache.v1.mp_coordinator.views.key_directory import Placement


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
        tier: Cache tier the quota applies to (only ``l2`` is supported today).
    """

    limit_gb: float = Field(ge=0.0)
    tier: Tier = Tier.L2


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


class QuotaConfigRequest(BaseModel):
    """Body of ``PUT /quota/config``.

    Attributes:
        default_limit_gb: Byte budget in GiB applied to salts with no
            explicit quota entry. ``None`` (default) leaves unquota'd
            salts exempt from eviction.
        tier: Cache tier the config applies to (only ``l2`` today).
    """

    default_limit_gb: float | None = Field(default=None, ge=0.0)
    tier: Tier = Tier.L2


class QuotaConfigResponse(BaseModel):
    """Reply to ``GET`` / ``PUT /quota/config``.

    Attributes:
        default_limit_gb: Current default limit in GiB, or ``None``
            when unquota'd salts are exempt from eviction.
    """

    default_limit_gb: float | None


# -- Usage / quota status ----------------------------------------------------


class StatusResponse(BaseModel):
    """Combined quota and usage for a single ``cache_salt``, on one tier.

    Every field describes the tier the request asked for. Quotas are
    enforced on L2 only, so an ``l1`` request reports L1 usage with
    ``quota_exists=False`` — never the L2 quota, which governs different
    bytes.

    Attributes:
        cache_salt: The tenant identifier.
        quota_limit_gb: The byte budget in GiB (0.0 if no quota applies
            to the requested tier).
        quota_exists: Whether an explicit quota is registered for the
            requested tier.
        usage_gb: Current usage in GiB on the requested tier.
    """

    cache_salt: str
    quota_limit_gb: float
    quota_exists: bool
    usage_gb: float


class StatusListResponse(BaseModel):
    """Reply to ``GET /quota``, scoped to the requested tier.

    Attributes:
        total_gb: Aggregate usage in GiB on the requested tier.
        by_cache_salt: Per-tenant breakdown with quota and usage. Rows
            come from the tier's usage plus the quotas that apply to it,
            so an ``l1`` listing holds only salts with L1 usage.
    """

    total_gb: float
    by_cache_salt: list[StatusResponse]


# -- Memory pressure ---------------------------------------------------------


class ModuleMemoryStatus(BaseModel):
    """Usage joined to declared capacity for one memory compartment.

    Attributes:
        tier: ``l1`` or ``l2``.
        backend: Storage backend within the tier.
        shared: Set for a fleet-shared pool, whose bytes are counted once
            for the fleet, not once per mounting instance.
        used_bytes: Bytes held, from the admitted cache-event stream.
        capacity_bytes: Declared capacity, or ``0`` if none was declared.
        usage_ratio: ``used_bytes / capacity_bytes``, or ``None`` when no
            capacity was declared -- ``None`` rather than a sentinel, which
            would read as real occupancy. Values above ``1.0`` are not
            clamped: they mean the declared cap disagrees with what the
            tier admitted.
    """

    tier: Tier
    backend: str
    shared: bool
    used_bytes: int
    capacity_bytes: int
    usage_ratio: float | None = None


class InstanceMemoryStatus(BaseModel):
    """One MP server's memory compartments.

    Attributes:
        instance_id: The server this describes.
        registered: Whether it is currently in the instance registry. A
            deregistered server can still hold L2 bytes, so ``False`` is
            valid.
        declared_capacity: Whether any capacity was declared. When
            ``False``, every module's ``usage_ratio`` is ``None``.
        modules: Privately-owned compartments, sorted by tier then backend.
            Shared pools are reported at the fleet level instead.
    """

    instance_id: str
    registered: bool
    declared_capacity: bool
    modules: list[ModuleMemoryStatus] = Field(default_factory=list)


class FleetMemoryResponse(BaseModel):
    """Fleet-wide memory view: every server plus the shared pools.

    Attributes:
        instances: Per-server status, sorted by ``instance_id``.
        shared_modules: Fleet-shared compartments, counted once. Capacity is
            reported only when every declaring server agrees; a disagreement
            reads as undeclared.
    """

    instances: list[InstanceMemoryStatus] = Field(default_factory=list)
    shared_modules: list[ModuleMemoryStatus] = Field(default_factory=list)


# -- Key directory -----------------------------------------------------------


class CacheEventsRequest(BaseModel):
    """Body of ``POST /events``.

    Attributes:
        batches: Event batches to apply, in emission order per instance.
            Includes ``config`` batches, which declare capacity rather than
            report placements.
    """

    batches: list[CacheEventBatch] = Field(default_factory=list)

    @field_validator("batches")
    @classmethod
    def _validate_batches(cls, value: list[CacheEventBatch]) -> list[CacheEventBatch]:
        """Enforce encoding-level constraints on every entry.

        Args:
            value: The hydrated batches.

        Returns:
            The unchanged batches once every constraint holds.

        Raises:
            ValueError: If an entry's key cannot convert into an
                ``ObjectKey`` (surfaced as 422).
        """
        for batch in value:
            for entry in batch.entries:
                entry.key.to_object_key()
        return value


class CacheEventsResponse(BaseModel):
    """Reply to ``POST /events``.

    Attributes:
        applied: Batches applied to the directory.
        duplicates: Batches dropped as already-applied replays.
        stale: Batches dropped for carrying an outdated incarnation.
    """

    applied: int = 0
    duplicates: int = 0
    stale: int = 0


class TokenSequenceForm(BaseModel):
    """A token sequence plus the identity that resolves it to keys.

    The shared shape of every directory query that speaks tokens rather
    than keys: a `chunk_hash` names content only, so the model, salt, and
    rank fan-out that turn it into an :class:`EncodedObjectKey` have to
    travel with it. ``/directory/lookup`` uses them to *build* keys;
    ``/directory/blend-lookup`` uses them to keep matches inside the
    namespace the caller can actually retrieve from.

    Chunk hashes are prefix-chained, so ``token_ids`` must be the
    request's **full** sequence from position 0; a mid-request slice
    resolves to different keys. Trailing incomplete chunks are ignored.

    Attributes:
        token_ids: The request's full token sequence.
        model_name: Model whose rank fan-out to use.
        world_size: World size selecting the per-rank fan-out.
        cache_salt: Per-tenant isolation salt applied to produced keys.
    """

    token_ids: list[int] = Field(default_factory=list)
    model_name: str = ""
    world_size: int = Field(default=1, ge=1)
    cache_salt: str = ""

    @model_validator(mode="after")
    def _validate_token_identity(self) -> "TokenSequenceForm":
        """Reject a token sequence with no model to resolve it against.

        Returns:
            The unchanged request once any token sequence names a model.

        Raises:
            ValueError: If ``token_ids`` is supplied without
                ``model_name``.
        """
        if self.token_ids and not self.model_name:
            raise ValueError("'model_name' is required with 'token_ids'")
        return self


class DirectoryLookupRequest(TokenSequenceForm):
    """Body of ``POST /directory/lookup``.

    Supply exactly one of the two lookup forms:

    * ``keys`` — resolve these keys directly.
    * the inherited tokens form — resolve a request's token sequence to
      the object keys of its complete chunks, the same fan-out the pin
      APIs use.

    Attributes:
        keys: The keys to resolve (keys form).
    """

    keys: list[EncodedObjectKey] = Field(default_factory=list)

    @field_validator("keys")
    @classmethod
    def _validate_keys(cls, value: list[EncodedObjectKey]) -> list[EncodedObjectKey]:
        """Reject undecodable keys at request validation (surfaced as 422).

        Args:
            value: The hydrated keys.

        Returns:
            The unchanged keys once each converts to an ``ObjectKey``.

        Raises:
            ValueError: If a key cannot convert into an ``ObjectKey``.
        """
        for encoded in value:
            encoded.to_object_key()
        return value

    @model_validator(mode="after")
    def _validate_one_form(self) -> "DirectoryLookupRequest":
        """Enforce that exactly one lookup form is supplied.

        Returns:
            The unchanged request once exactly one form is present.

        Raises:
            ValueError: If both or neither of ``keys`` / ``token_ids``
                is supplied.
        """
        if bool(self.keys) == bool(self.token_ids):
            raise ValueError("supply exactly one of 'keys' or 'token_ids'")
        return self


class DirectoryKeyPlacements(BaseModel):
    """Placements and token ids for one resolved key.

    Attributes:
        key: The resolved key, echoed back.
        placements: Known placements; empty when the directory knows
            nothing about the key.
        token_ids: The chunk's token ids; empty when unknown.
    """

    key: EncodedObjectKey
    placements: list[Placement] = Field(default_factory=list)
    token_ids: list[int] = Field(default_factory=list)


class DirectoryLookupResponse(BaseModel):
    """Reply to ``POST /directory/lookup``.

    Attributes:
        chunks: Complete chunks the request resolved to — the token
            sequence's chunk count (tokens form) or the number of
            requested keys (keys form).
        results: One entry per resolved key, in request order (tokens
            form: ``chunks`` x per-rank fan-out).
    """

    chunks: int = 0
    results: list[DirectoryKeyPlacements] = Field(default_factory=list)


class DirectoryKeyInfo(BaseModel):
    """One listed directory key.

    Attributes:
        key: The listed key.
        placements: The key's placements that matched the listing filters.
        num_tokens: Token ids known for the key's chunk (``0`` = unknown).
    """

    key: EncodedObjectKey
    placements: list[Placement] = Field(default_factory=list)
    num_tokens: int = 0


class DirectoryListResponse(BaseModel):
    """Reply to ``GET /directory/keys``.

    Attributes:
        total: Keys with at least one placement matching the filters.
        keys: The requested page of them, in directory iteration order.
    """

    total: int = 0
    keys: list[DirectoryKeyInfo] = Field(default_factory=list)


class BlendLookupRequest(TokenSequenceForm):
    """Body of ``POST /directory/blend-lookup``.

    The same tokens form ``/directory/lookup`` takes -- the two questions
    differ in where the content may sit, not in how it is described. Only
    the tokens form exists here: a fragment query is a search over
    content, so there is nothing to name by key.

    Unlike ``/directory/lookup`` the query need not be a prefix, but the
    identity fields matter just as much: they scope matches to chunks the
    caller's own key expansion can retrieve.
    """

    @model_validator(mode="after")
    def _validate_tokens_present(self) -> "BlendLookupRequest":
        """Enforce that a query sequence was supplied.

        Returns:
            The unchanged request once it carries tokens.

        Raises:
            ValueError: If ``token_ids`` is empty.
        """
        if not self.token_ids:
            raise ValueError("'token_ids' is required")
        return self


class BlendMatchModel(BaseModel):
    """Wire form of one blend match (see ``BlendMatch``).

    Attributes:
        chunk_hash: Hex of the matched chunk's ``ObjectKey.chunk_hash``.
        old_st: Its position in the sequence it was stored under
            (re-RoPE source).
        cur_st: Its position in the query (re-RoPE target).
    """

    chunk_hash: str
    old_st: int
    cur_st: int


class BlendLookupResponse(BaseModel):
    """Reply to ``POST /directory/blend-lookup``.

    Attributes:
        matches: Matched chunks, ascending by ``cur_st``. They may
            overlap in the query; the caller resolves overlaps.
    """

    matches: list[BlendMatchModel] = Field(default_factory=list)


class PrefetchRequest(BaseModel):
    """Body of ``POST /cache/prefetches`` on the coordinator.

    Asks the coordinator to warm one MP server's L1 with the chunks of a token
    sequence. The caller describes content by ``token_ids`` -- the unit the
    cache speaks -- not by internal cache keys, which it cannot construct. The
    coordinator forwards the request verbatim to that server's own
    ``POST /cache/prefetches``, which hashes the tokens and expands them into the
    per-rank keys.

    Attributes:
        instance_id: Identifier of the target MP server (must be registered).
        model_name: Model whose layout the target uses to allocate L1 buffers.
        world_size: World size selecting the layout and the per-rank fan-out.
        token_ids: Prompt tokens whose complete chunks should be warmed.
        cache_salt: Per-tenant isolation salt applied to the produced keys.
    """

    instance_id: str
    model_name: str
    world_size: int = Field(ge=1)
    token_ids: list[int] = Field(default_factory=list)
    cache_salt: str = ""


class PrefetchResponse(BaseModel):
    """Reply to ``POST /cache/prefetches`` on the coordinator.

    Attributes:
        instance_id: The target MP server the prefetch was submitted to.
        request_id: The server's job id to poll via
            ``GET /cache/prefetches/{instance_id}/{request_id}``. Empty when
            ``status`` is ``"noop"`` (nothing to warm).
        chunks: Number of whole chunks submitted to warm.
        status: ``"submitted"`` (a job is in flight) or ``"noop"`` (the
            sequence was shorter than one chunk).
    """

    instance_id: str
    request_id: str = ""
    chunks: int = 0
    status: str


class PinRequest(BaseModel):
    """Body of ``POST`` / ``DELETE /cache/pins`` on the coordinator.

    Pinning protects the resolved keys from L2 eviction until unpinned. The
    coordinator resolves ``token_ids`` to keys locally; L2 pins are fleet-wide
    (per ``cache_salt``), so no target instance is needed.

    Attributes:
        model_name: Model whose rank fan-out to use when resolving keys.
        world_size: World size selecting the per-rank fan-out.
        token_ids: Prompt tokens whose complete chunks should be (un)pinned.
        cache_salt: Per-tenant isolation salt applied to the produced keys.
    """

    model_name: str
    world_size: int = Field(ge=1)
    token_ids: list[int] = Field(default_factory=list)
    cache_salt: str = ""


class PinResponse(BaseModel):
    """Reply to ``POST`` / ``DELETE /cache/pins`` on the coordinator.

    Attributes:
        requested: Number of whole chunks the token sequence resolved to.
        affected: Number of L2 keys pinned (on pin) or unpinned (on unpin);
            disambiguated by ``status``.
        status: ``"pinned"`` / ``"unpinned"``.
    """

    requested: int = 0
    affected: int = 0
    status: str


class DeleteRequest(BaseModel):
    """Body of ``POST /cache/delete`` on the coordinator.

    Attributes:
        instance_id: Identifier of the target MP server (must be registered).
        model_name: Model whose layout the target uses to resolve keys.
        world_size: World size selecting the layout and the per-rank fan-out.
        token_ids: Prompt tokens whose complete chunks should be deleted.
        cache_salt: Per-tenant isolation salt applied to the produced keys.
        tier: Which tier(s) to delete: ``l1`` (L1 only), ``l2`` (L2 only), or
            ``all`` (both). ``l1`` never touches L2 and vice versa.
        force: When True, delete even locked/pinned keys -- bypasses L1
            locks/pins on the node and the coordinator's L2 pin filter.
    """

    instance_id: str
    model_name: str
    world_size: int = Field(ge=1)
    token_ids: list[int] = Field(default_factory=list)
    cache_salt: str = ""
    tier: Tier = Tier.ALL
    force: bool = False


class DeleteResponse(BaseModel):
    """Reply to ``POST /cache/delete`` on the coordinator.

    Attributes:
        instance_id: The target MP server the request was dispatched to.
        requested: Number of whole chunks the token sequence resolved to.
        affected: Total keys removed across the tiers acted on -- L1 keys deleted
            by the node plus L2 keys deleted by the coordinator. A chunk resident
            in both tiers (``tier=all``) contributes to both, so ``affected`` may
            exceed ``requested`` (which counts chunks, not per-tier keys).
        skipped: Total keys refused because they were locked/pinned (non-force
            only) -- L1 keys the node refused plus L2 keys held back for an L2 pin.
        status: ``"deleted"`` / ``"noop"``.
    """

    instance_id: str
    requested: int = 0
    affected: int = 0
    skipped: int = 0
    status: str
