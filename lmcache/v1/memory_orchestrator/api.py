# SPDX-License-Identifier: Apache-2.0
"""Types shared by the memory orchestrator state machine, servicer and client.

These mirror the messages of ``protos/memory_orchestrator.proto`` as plain
Python values so the state machine and the MP-side callers never handle
protobuf objects. The module imports neither torch nor gRPC.
"""

# Standard
from dataclasses import dataclass
import enum

DEFAULT_LAYOUT_FINGERPRINT: bytes = b"lmcache/kv_2ltd/v1"
"""Layout fingerprint of the region when the operator does not set one."""

VISIBILITY_MODES: tuple[str, ...] = ("software_fenced", "coherent")
"""Accepted ways a region makes writes visible to other hosts."""

MAX_UINT64: int = 2**64 - 1
"""Largest epoch, incarnation or request id the wire carries."""


class WriteStatus(enum.Enum):
    """Per-entry outcome of a write reservation."""

    WRITE_GRANTED = enum.auto()
    """A new extent was allocated; the caller owns it until finish or abort."""

    EXISTS_VALID = enum.auto()
    """The object is already committed; nothing was allocated."""

    BUSY_WRITING = enum.auto()
    """Another writer holds the object; nothing was allocated."""

    OUT_OF_SPACE = enum.auto()
    """The batch's new objects did not fit; none of them was allocated."""


class ReadStatus(enum.Enum):
    """Per-entry outcome of a read reservation."""

    READ_GRANTED = enum.auto()
    """The object is committed; leases were issued."""

    MISS = enum.auto()
    """The object is unknown to the orchestrator."""

    BUSY_WRITING = enum.auto()
    """The object exists but is not committed yet."""


class TokenStatus(enum.Enum):
    """Per-token outcome of a finish or abort call."""

    OK = enum.auto()
    """The transition happened now or on an earlier call with the same token."""

    STALE_TOKEN = enum.auto()
    """The token is unknown, foreign, retired, or used with the wrong call."""


@dataclass(frozen=True)
class WireObjectKey:
    """Identity of one stored KV object; hashable so it can index state."""

    chunk_hash: bytes
    model_name: str
    kv_rank: int
    object_group_id: int
    cache_salt: str


@dataclass(frozen=True)
class WireLayout:
    """Tensor layout stored with an object and returned to its readers.

    ``dtypes[i]`` is the torch dtype name, without the ``torch.`` prefix, of
    the tensor with shape ``shapes[i]``.
    """

    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]


@dataclass(frozen=True)
class Handle:
    """Location of an object's extent inside the shared region."""

    offset: int
    """Byte offset of the extent from the start of the region."""

    length: int
    """Extent length in bytes, a multiple of the region alignment."""

    generation: int
    """Extent generation; always 1 because extents are never reused."""


@dataclass(frozen=True)
class Envelope:
    """Identity and fencing fields carried by every request."""

    region_id: str
    expected_region_epoch: int
    """Epoch the client registered under; 0 only for DescribeRegion."""

    client_id: str
    """Stable across restarts of one MP server."""

    client_incarnation: int
    """Unique per MP server start."""

    request_id: int
    """Idempotency key, unique per (client_id, client_incarnation)."""


@dataclass(frozen=True)
class RegionContract:
    """What the orchestrator promises about its region."""

    region_id: str
    region_epoch: int
    capacity_bytes: int
    alignment_bytes: int
    layout_fingerprint: bytes
    visibility_mode: str
    max_batch_entries: int
    reset_required: bool
    """The orchestrator restarted uncleanly and refuses every client call."""


@dataclass(frozen=True)
class RegisterResult:
    """Outcome of registering a client incarnation."""

    region_epoch: int
    retired_writes: int
    """WRITING objects of an older incarnation that were moved to CONSUMED."""

    released_leases: int
    """Read leases of an older incarnation that were dropped."""


@dataclass(frozen=True)
class WriteRequest:
    """One object to reserve for writing."""

    key: WireObjectKey
    payload_bytes: int
    """Unaligned payload size; must be > 0."""

    layout: WireLayout


@dataclass(frozen=True)
class WriteGrantResult:
    """Outcome of one write reservation; ``handle`` and ``token`` are set
    only for ``WRITE_GRANTED``."""

    status: WriteStatus
    handle: Handle | None = None
    token: bytes | None = None


@dataclass(frozen=True)
class ReadRequest:
    """One object to reserve for reading."""

    key: WireObjectKey
    reader_count: int
    """Number of leases to issue, one per reader; 1 to 1024."""


@dataclass(frozen=True)
class ReadGrantResult:
    """Outcome of one read reservation; every field but ``status`` is set
    only for ``READ_GRANTED``."""

    status: ReadStatus
    handle: Handle | None = None
    leases: tuple[bytes, ...] = ()
    """``reader_count`` lease tokens, each released by one finish_read."""

    layout: WireLayout | None = None
    payload_bytes: int = 0


@dataclass(frozen=True)
class RegionUsage:
    """Snapshot of the region's allocation and object counters."""

    capacity_bytes: int
    allocated_bytes: int
    """Monotonic bump pointer; includes the extents of CONSUMED objects."""

    valid_bytes: int
    """Sum of the aligned extent lengths of VALID objects."""

    writing: int
    valid: int
    consumed: int
    read_leases: int
    clients: int


@dataclass(frozen=True)
class CloseResult:
    """Outcome of closing a client incarnation."""

    aborted_writes: int
    released_leases: int


class OrchestratorError(Exception):
    """Base class of every error raised by the orchestrator client."""


class OrchestratorUnavailableError(OrchestratorError):
    """The orchestrator could not be reached within the retry budget."""


class RegionFencedError(OrchestratorError):
    """The orchestrator refused the call (gRPC ``FAILED_PRECONDITION``).

    Raised for a wrong region, a stale epoch, an unregistered client, a
    registration mismatch, or a region that needs an offline reset.

    Attributes:
        reset_required: True when the orchestrator restarted uncleanly and
            refuses every call until an operator resets the region.
    """

    def __init__(self, message: str, *, reset_required: bool = False) -> None:
        super().__init__(message)
        self.reset_required = reset_required


class RequestRejectedError(OrchestratorError):
    """The orchestrator rejected the request itself.

    Raised for gRPC ``INVALID_ARGUMENT``, ``RESOURCE_EXHAUSTED`` and
    ``ALREADY_EXISTS``; the request changed no state.

    Attributes:
        code: Name of the gRPC status code, e.g. ``"INVALID_ARGUMENT"``.
    """

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
