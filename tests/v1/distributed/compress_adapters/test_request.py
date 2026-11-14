# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for backend-independent decompression requests."""

# Standard
from dataclasses import replace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.compress_adapters.device import DeviceExecutionContext
from lmcache.v1.distributed.compress_adapters.format import (
    CompressionCodec,
    CompressionFraming,
    PostDecompressTransform,
    StoredCompressionFormat,
    ValidatedCompressedRecord,
    validate_complete_record,
)
from lmcache.v1.distributed.compress_adapters.lease import DeviceOutputLeaseManager
from lmcache.v1.distributed.compress_adapters.reference import (
    encode_reference_record,
)
from lmcache.v1.distributed.compress_adapters.request import (
    GpuDecompressItem,
    GpuDecompressRequest,
    GpuDecompressRequestError,
    GpuDecompressRequestLimits,
)
from lmcache.v1.platform.base.device_spec import DeviceSpec
import lmcache.v1.distributed.compress_adapters.device as device_module


class _TestDeviceSpec(DeviceSpec):
    """Select a CPU device without requiring an accelerator."""

    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def backend_name(self) -> str:
        return "cpu"

    @property
    def torch_module_name(self) -> str:
        return "cpu"


class _TestCacheContext:
    """Expose distinct fixed staging slots through the public context API."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.stream = object()
        backing = torch.empty(512, dtype=torch.uint8)
        self._buffers = {
            (0, 0): backing[0:128],
            (1, 0): backing[128:256],
            (2, 0): backing[256:384],
            (3, 0): backing[384:512],
            (4, 0): backing[0:0],
        }

    def get_temp_object_group_buffer(
        self,
        batch_idx: int,
        object_group_idx: int,
    ) -> torch.Tensor:
        """Return the selected staging slot."""
        try:
            return self._buffers[(batch_idx, object_group_idx)]
        except KeyError as exc:
            raise ValueError("invalid staging slot or object group") from exc


def _stored_format(
    framing: CompressionFraming = CompressionFraming.RAW,
) -> StoredCompressionFormat:
    """Select Deflate framing with no post-decompression transform.

    Args:
        framing: Raw or Gzip framing to use for the test record.

    Returns:
        The complete stored-format identity for the selected framing.
    """
    return StoredCompressionFormat(
        codec=CompressionCodec.DEFLATE,
        framing=framing,
        post_decompress_transform=PostDecompressTransform.NONE,
    )


def _validated_record(
    data: bytes,
    *,
    framing: CompressionFraming = CompressionFraming.RAW,
) -> ValidatedCompressedRecord:
    """Encode and validate test input using independent 16-byte chunks.

    Args:
        data: Immutable uncompressed bytes to encode.
        framing: Raw or Gzip framing for each compressed stream.

    Returns:
        An immutable record whose stored-payload integrity is verified.

    Raises:
        ValueError: If the input requires more chunks than the format permits.
    """
    return validate_complete_record(
        encode_reference_record(
            data,
            stored_format=_stored_format(framing),
            chunk_size=16,
        )
    )


def _manager(monkeypatch: pytest.MonkeyPatch) -> DeviceOutputLeaseManager:
    """Create a CPU manager with distinct staging slots for request tests.

    Args:
        monkeypatch: Fixture used to select the test CPU backend.

    Returns:
        A manager retaining a fresh test context. Tests must release each
        acquired output lease after checking the request contract.
    """
    monkeypatch.setattr(device_module, "current_device_spec", _TestDeviceSpec())
    context = _TestCacheContext()
    execution_context = DeviceExecutionContext.from_cache_context(
        context  # type: ignore[arg-type]
    )
    return DeviceOutputLeaseManager.for_execution_context(execution_context)


def _item(
    manager: DeviceOutputLeaseManager,
    data: bytes,
    *,
    batch_idx: int = 0,
    framing: CompressionFraming = CompressionFraming.RAW,
) -> GpuDecompressItem:
    """Pair a test record with an exactly sized reservation in object group zero.

    Args:
        manager: Reservation authority created for the test context.
        data: Uncompressed bytes determining both record and output sizes.
        batch_idx: Staging slot to reserve within the manager's context.
        framing: Raw or Gzip framing for the stored record.

    Returns:
        An item whose active output lease must be released by the test.

    Raises:
        ValueError: If the record exceeds format bounds or the staging range
            cannot hold ``data``.
        DeviceOutputLeaseError: If the manager cannot reserve the range.
    """
    return GpuDecompressItem(
        validated_record=_validated_record(data, framing=framing),
        output_lease=manager.acquire(
            batch_idx=batch_idx,
            object_group_idx=0,
            byte_length=len(data),
        ),
        expected_uncompressed_size=len(data),
    )


def _exact_limits(
    items: tuple[GpuDecompressItem, ...],
) -> GpuDecompressRequestLimits:
    """Set every request ceiling to the supplied items' exact aggregate.

    Args:
        items: Non-empty tuple of constructed decompression items.

    Returns:
        Limits that accept equality at every record, chunk, and byte boundary.

    Raises:
        ValueError: If ``items`` is empty and would produce a zero record cap.
    """
    return GpuDecompressRequestLimits(
        max_records=len(items),
        max_compression_chunks=sum(
            len(item.validated_record.header.chunks) for item in items
        ),
        max_total_record_bytes=sum(
            item.validated_record.header.record_size for item in items
        ),
        max_total_compressed_payload_bytes=sum(
            chunk.compressed_size
            for item in items
            for chunk in item.validated_record.header.chunks
        ),
        max_total_uncompressed_bytes=sum(
            item.expected_uncompressed_size for item in items
        ),
    )


def test_item_joins_validated_record_and_exact_output_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One item retains matching input proof, logical size, and destination."""
    manager = _manager(monkeypatch)
    data = b"0123456789abcdef"
    item = _item(manager, data)

    assert item.validated_record.record_bytes.startswith(b"LMCR")
    assert item.validated_record.header.uncompressed_size == len(data)
    assert item.output_lease.buffer_range.byte_length == len(data)
    assert item.expected_uncompressed_size == len(data)

    item.output_lease.release()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("validated_record", object(), "validated_record"),
        ("output_lease", object(), "output_lease"),
        ("expected_uncompressed_size", True, "expected_uncompressed_size"),
        ("expected_uncompressed_size", -1, "expected_uncompressed_size"),
    ],
)
def test_item_validates_argument_contracts(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    """Items reject invalid public arguments with domain-specific messages."""
    manager = _manager(monkeypatch)
    record = _validated_record(b"0123456789abcdef")
    lease = manager.acquire(batch_idx=0, object_group_idx=0, byte_length=16)
    arguments: dict[str, object] = {
        "validated_record": record,
        "output_lease": lease,
        "expected_uncompressed_size": 16,
    }
    arguments[field] = value

    exception = ValueError if value == -1 else TypeError
    with pytest.raises(exception, match=message):
        GpuDecompressItem(**arguments)  # type: ignore[arg-type]

    lease.release()


def test_item_requires_active_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    """A released staging reservation cannot enter a new item."""
    manager = _manager(monkeypatch)
    record = _validated_record(b"0123456789abcdef")
    lease = manager.acquire(batch_idx=0, object_group_idx=0, byte_length=16)
    lease.release()

    with pytest.raises(GpuDecompressRequestError, match="not active"):
        GpuDecompressItem(
            validated_record=record,
            output_lease=lease,
            expected_uncompressed_size=16,
        )


def test_item_requires_record_to_match_logical_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stored metadata cannot select output bytes different from KV layout."""
    manager = _manager(monkeypatch)
    record = _validated_record(b"0123456789abcdef")
    lease = manager.acquire(batch_idx=0, object_group_idx=0, byte_length=17)

    with pytest.raises(GpuDecompressRequestError, match="record uncompressed_size"):
        GpuDecompressItem(
            validated_record=record,
            output_lease=lease,
            expected_uncompressed_size=17,
        )

    lease.release()


def test_item_requires_lease_to_match_logical_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record cannot be submitted into a smaller or larger destination."""
    manager = _manager(monkeypatch)
    record = _validated_record(b"0123456789abcdef")
    lease = manager.acquire(batch_idx=0, object_group_idx=0, byte_length=15)

    with pytest.raises(GpuDecompressRequestError, match="output lease byte_length"):
        GpuDecompressItem(
            validated_record=record,
            output_lease=lease,
            expected_uncompressed_size=16,
        )

    lease.release()


def test_request_exposes_homogeneous_aggregate_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid batch exposes the exact work and allocation quantities."""
    manager = _manager(monkeypatch)
    items = (
        _item(manager, b"a" * 32, batch_idx=0),
        _item(manager, b"b" * 16, batch_idx=1),
    )
    limits = _exact_limits(items)
    request = GpuDecompressRequest(items=items, limits=limits)

    assert request.lease_manager is manager
    assert request.version == items[0].validated_record.header.version
    assert request.stored_format == _stored_format()
    assert request.record_count == 2
    assert request.compression_chunk_count == 3
    assert request.total_record_bytes == limits.max_total_record_bytes
    assert (
        request.total_compressed_payload_bytes
        == limits.max_total_compressed_payload_bytes
    )
    assert request.total_uncompressed_bytes == 48
    request.validate_active_leases()

    for item in items:
        item.output_lease.release()


def test_request_requires_nonempty_tuple_and_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch shape and operational ceilings are always explicit."""
    manager = _manager(monkeypatch)
    item = _item(manager, b"a" * 16)
    limits = _exact_limits((item,))

    with pytest.raises(TypeError, match="items must be a tuple"):
        GpuDecompressRequest(items=[item], limits=limits)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        GpuDecompressRequest(items=(), limits=limits)
    with pytest.raises(TypeError, match=r"items\[0\]"):
        GpuDecompressRequest(items=(object(),), limits=limits)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="GpuDecompressRequestLimits"):
        GpuDecompressRequest(items=(item,), limits=object())  # type: ignore[arg-type]

    item.output_lease.release()


def test_request_rejects_duplicate_output_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One destination capability cannot represent two outputs in a batch."""
    manager = _manager(monkeypatch)
    item = _item(manager, b"a" * 16)
    items = (item, item)

    with pytest.raises(GpuDecompressRequestError, match="reuses output lease"):
        GpuDecompressRequest(items=items, limits=_exact_limits(items))

    item.output_lease.release()


def test_request_requires_one_lease_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One request cannot cross cache-context reservation authorities."""
    first = _item(_manager(monkeypatch), b"a" * 16)
    second = _item(_manager(monkeypatch), b"b" * 16)
    items = (first, second)

    with pytest.raises(GpuDecompressRequestError, match="different.*manager"):
        GpuDecompressRequest(items=items, limits=_exact_limits(items))

    first.output_lease.release()
    second.output_lease.release()


def test_request_requires_homogeneous_stored_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native batch receives one record-version and stored-format identity."""
    manager = _manager(monkeypatch)
    first = _item(manager, b"a" * 16, batch_idx=0)
    second = _item(
        manager,
        b"b" * 16,
        batch_idx=1,
        framing=CompressionFraming.GZIP,
    )
    items = (first, second)

    with pytest.raises(GpuDecompressRequestError, match="not homogeneous"):
        GpuDecompressRequest(items=items, limits=_exact_limits(items))

    first.output_lease.release()
    second.output_lease.release()


def test_request_revalidates_lease_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release after item construction is detected before native submission."""
    manager = _manager(monkeypatch)
    item = _item(manager, b"a" * 16)
    limits = _exact_limits((item,))
    request = GpuDecompressRequest(items=(item,), limits=limits)
    item.output_lease.release()

    with pytest.raises(GpuDecompressRequestError, match="not active"):
        request.validate_active_leases()
    with pytest.raises(GpuDecompressRequestError, match="not active"):
        GpuDecompressRequest(items=(item,), limits=limits)


def test_request_accepts_exact_operational_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every ceiling is inclusive rather than rejecting its documented max."""
    manager = _manager(monkeypatch)
    items = (
        _item(manager, b"a" * 32, batch_idx=0),
        _item(manager, b"b" * 16, batch_idx=1),
    )

    request = GpuDecompressRequest(items=items, limits=_exact_limits(items))

    assert request.record_count == 2
    for item in items:
        item.output_lease.release()


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("max_records", "record_count"),
        ("max_compression_chunks", "compression_chunk_count"),
        ("max_total_record_bytes", "total_record_bytes"),
        (
            "max_total_compressed_payload_bytes",
            "total_compressed_payload_bytes",
        ),
        ("max_total_uncompressed_bytes", "total_uncompressed_bytes"),
    ],
)
def test_request_rejects_each_exceeded_operational_limit(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    message: str,
) -> None:
    """Each bounded resource is enforced and named independently."""
    manager = _manager(monkeypatch)
    items = (
        _item(manager, b"a" * 32, batch_idx=0),
        _item(manager, b"b" * 16, batch_idx=1),
    )
    limits = _exact_limits(items)
    limited = replace(limits, **{field: getattr(limits, field) - 1})

    with pytest.raises(GpuDecompressRequestError, match=message):
        GpuDecompressRequest(items=items, limits=limited)

    for item in items:
        item.output_lease.release()


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("max_records", True, TypeError),
        ("max_records", 0, ValueError),
        ("max_compression_chunks", -1, ValueError),
        ("max_total_record_bytes", True, TypeError),
        ("max_total_compressed_payload_bytes", -1, ValueError),
        ("max_total_uncompressed_bytes", -1, ValueError),
    ],
)
def test_limits_validate_integer_bounds(
    field: str,
    value: object,
    exception: type[Exception],
) -> None:
    """Operational limits reject booleans, negatives, and zero record caps."""
    arguments: dict[str, object] = {
        "max_records": 1,
        "max_compression_chunks": 1,
        "max_total_record_bytes": 1,
        "max_total_compressed_payload_bytes": 1,
        "max_total_uncompressed_bytes": 1,
    }
    arguments[field] = value

    with pytest.raises(exception, match=field):
        GpuDecompressRequestLimits(**arguments)  # type: ignore[arg-type]


def test_all_empty_request_needs_no_chunks_or_output_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty logical records remain valid without creating native chunk work."""
    manager = _manager(monkeypatch)
    item = _item(manager, b"", batch_idx=4)
    limits = _exact_limits((item,))

    request = GpuDecompressRequest(items=(item,), limits=limits)

    assert request.record_count == 1
    assert request.compression_chunk_count == 0
    assert request.total_compressed_payload_bytes == 0
    assert request.total_uncompressed_bytes == 0
    assert request.total_record_bytes > 0

    item.output_lease.release()
