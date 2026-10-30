# SPDX-License-Identifier: Apache-2.0
"""Backend-independent GPU decompression request contracts."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field

# First Party
from lmcache.v1.distributed.compress_adapters.format import (
    StoredCompressionFormat,
    ValidatedCompressedRecord,
)
from lmcache.v1.distributed.compress_adapters.lease import (
    DeviceOutputLease,
    DeviceOutputLeaseManager,
)


def _require_nonnegative_int(name: str, value: int) -> None:
    """Validate a request size or limit without accepting booleans as integers.

    Args:
        name: Field name used in validation errors.
        value: Integer that must be zero or greater.

    Returns:
        None when the value is valid.

    Raises:
        TypeError: If ``value`` is not an exact ``int``, including ``bool``.
        ValueError: If ``value`` is negative.
    """
    if type(value) is not int:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}")


def _raise_if_limit_exceeded(name: str, value: int, maximum: int) -> None:
    """Enforce one inclusive operational ceiling during request construction.

    Args:
        name: Aggregate quantity named in the failure message.
        value: Nonnegative aggregate computed from validated items.
        maximum: Validated nonnegative ceiling from the request limits.

    Returns:
        None when ``value`` is at or below ``maximum``.

    Raises:
        GpuDecompressRequestError: If ``value`` exceeds ``maximum``.
    """
    if value > maximum:
        raise GpuDecompressRequestError(
            f"{name} {value} exceeds request limit {maximum}"
        )


class GpuDecompressRequestError(ValueError):
    """Raised when a decompression item or request violates its contract."""


@dataclass(frozen=True, kw_only=True, slots=True)
class GpuDecompressRequestLimits:
    """Backend-independent operational limits for one request.

    Args:
        max_records: Positive maximum number of portable records.
        max_compression_chunks: Maximum flattened record chunk count.
        max_total_record_bytes: Maximum aggregate complete record bytes,
            including headers, alignment gaps, and compressed payloads.
        max_total_compressed_payload_bytes: Maximum aggregate sum of
            descriptor-declared compressed stream bytes, excluding headers and
            alignment gaps.
        max_total_uncompressed_bytes: Maximum aggregate expected output bytes.

    Raises:
        TypeError: If a limit is not an integer.
        ValueError: If ``max_records`` is not positive or another limit is
            negative.

    Notes:
        These caller-selected limits bound request construction before a
        backend is chosen. A backend may advertise and enforce stricter limits
        for its device, library, and execution engine.
    """

    max_records: int
    max_compression_chunks: int
    max_total_record_bytes: int
    max_total_compressed_payload_bytes: int
    max_total_uncompressed_bytes: int

    def __post_init__(self) -> None:
        _require_nonnegative_int("max_records", self.max_records)
        _require_nonnegative_int(
            "max_compression_chunks",
            self.max_compression_chunks,
        )
        _require_nonnegative_int(
            "max_total_record_bytes",
            self.max_total_record_bytes,
        )
        _require_nonnegative_int(
            "max_total_compressed_payload_bytes",
            self.max_total_compressed_payload_bytes,
        )
        _require_nonnegative_int(
            "max_total_uncompressed_bytes",
            self.max_total_uncompressed_bytes,
        )
        if self.max_records == 0:
            raise ValueError("max_records must be greater than zero")


@dataclass(frozen=True, kw_only=True, slots=True)
class GpuDecompressItem:
    """One validated record paired with an exclusive staging destination.

    Args:
        validated_record: Immutable complete record whose header and stored
            payload checksum have been validated.
        output_lease: Active reservation of the context-owned staging bytes
            that will receive this record's output.
        expected_uncompressed_size: Exact output bytes required by the
            caller's logical KV layout.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``expected_uncompressed_size`` is negative.
        GpuDecompressRequestError: If the lease is inactive or the record,
            logical expectation, and leased output size do not agree exactly.

    Notes:
        Construction performs backend-independent validation only. The native
        backend must revalidate the lease, live pointer, alignment, and its own
        capabilities immediately before submission.
    """

    validated_record: ValidatedCompressedRecord
    output_lease: DeviceOutputLease
    expected_uncompressed_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.validated_record, ValidatedCompressedRecord):
            raise TypeError(
                "validated_record must be a ValidatedCompressedRecord, got "
                f"{type(self.validated_record).__name__}"
            )
        if not isinstance(self.output_lease, DeviceOutputLease):
            raise TypeError(
                "output_lease must be a DeviceOutputLease, got "
                f"{type(self.output_lease).__name__}"
            )
        _require_nonnegative_int(
            "expected_uncompressed_size",
            self.expected_uncompressed_size,
        )
        if not self.output_lease.is_active:
            raise GpuDecompressRequestError("output_lease is not active")

        record_size = self.validated_record.header.uncompressed_size
        if record_size != self.expected_uncompressed_size:
            raise GpuDecompressRequestError(
                f"record uncompressed_size {record_size} does not match "
                f"expected_uncompressed_size {self.expected_uncompressed_size}"
            )

        leased_size = self.output_lease.buffer_range.byte_length
        if leased_size != self.expected_uncompressed_size:
            raise GpuDecompressRequestError(
                f"output lease byte_length {leased_size} does not match "
                f"expected_uncompressed_size {self.expected_uncompressed_size}"
            )


@dataclass(frozen=True, kw_only=True, slots=True)
class GpuDecompressRequest:
    """Homogeneous bounded batch of portable-record decompression items.

    Args:
        items: Non-empty tuple of items using unique active leases from one
            cache-context lease manager.
        limits: Backend-independent operational bounds for this batch.

    Raises:
        TypeError: If an argument or item has the wrong type.
        ValueError: If ``items`` is empty.
        GpuDecompressRequestError: If a lease is inactive or duplicated, items
            use different managers or stored formats, or an aggregate exceeds
            ``limits``.

    Notes:
        Stored-format homogeneity includes record version, codec, framing, and
        post-decompression transform. Backend, library, pointer-alignment, and
        engine-specific validation occur later.
    """

    items: tuple[GpuDecompressItem, ...]
    limits: GpuDecompressRequestLimits
    _record_count: int = field(init=False, repr=False, compare=False)
    _compression_chunk_count: int = field(init=False, repr=False, compare=False)
    _total_record_bytes: int = field(init=False, repr=False, compare=False)
    _total_compressed_payload_bytes: int = field(
        init=False,
        repr=False,
        compare=False,
    )
    _total_uncompressed_bytes: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise TypeError(f"items must be a tuple, got {type(self.items).__name__}")
        if not isinstance(self.limits, GpuDecompressRequestLimits):
            raise TypeError(
                "limits must be a GpuDecompressRequestLimits, got "
                f"{type(self.limits).__name__}"
            )
        for index, item in enumerate(self.items):
            if not isinstance(item, GpuDecompressItem):
                raise TypeError(
                    f"items[{index}] must be a GpuDecompressItem, got "
                    f"{type(item).__name__}"
                )
        if not self.items:
            raise ValueError("items must not be empty")

        _raise_if_limit_exceeded(
            "record_count",
            len(self.items),
            self.limits.max_records,
        )

        first_item = self.items[0]
        lease_manager = first_item.output_lease.owner
        version = first_item.validated_record.header.version
        stored_format = first_item.validated_record.header.stored_format
        seen_leases: set[DeviceOutputLease] = set()
        compression_chunk_count = 0
        total_record_bytes = 0
        total_compressed_payload_bytes = 0
        total_uncompressed_bytes = 0

        for index, item in enumerate(self.items):
            lease = item.output_lease
            if lease.owner is not lease_manager:
                raise GpuDecompressRequestError(
                    f"items[{index}] uses a different output lease manager"
                )
            if lease in seen_leases:
                raise GpuDecompressRequestError(
                    f"items[{index}] reuses output lease {lease.lease_id}"
                )
            seen_leases.add(lease)
            if not lease.is_active:
                raise GpuDecompressRequestError(
                    f"items[{index}] output lease is not active"
                )

            header = item.validated_record.header
            if header.version != version or header.stored_format != stored_format:
                raise GpuDecompressRequestError(
                    f"items[{index}] stored format is not homogeneous with items[0]"
                )

            compression_chunk_count += len(header.chunks)
            total_record_bytes += header.record_size
            total_compressed_payload_bytes += sum(
                chunk.compressed_size for chunk in header.chunks
            )
            total_uncompressed_bytes += item.expected_uncompressed_size

            _raise_if_limit_exceeded(
                "compression_chunk_count",
                compression_chunk_count,
                self.limits.max_compression_chunks,
            )
            _raise_if_limit_exceeded(
                "total_record_bytes",
                total_record_bytes,
                self.limits.max_total_record_bytes,
            )
            _raise_if_limit_exceeded(
                "total_compressed_payload_bytes",
                total_compressed_payload_bytes,
                self.limits.max_total_compressed_payload_bytes,
            )
            _raise_if_limit_exceeded(
                "total_uncompressed_bytes",
                total_uncompressed_bytes,
                self.limits.max_total_uncompressed_bytes,
            )

        object.__setattr__(self, "_record_count", len(self.items))
        object.__setattr__(
            self,
            "_compression_chunk_count",
            compression_chunk_count,
        )
        object.__setattr__(self, "_total_record_bytes", total_record_bytes)
        object.__setattr__(
            self,
            "_total_compressed_payload_bytes",
            total_compressed_payload_bytes,
        )
        object.__setattr__(
            self,
            "_total_uncompressed_bytes",
            total_uncompressed_bytes,
        )

    @property
    def lease_manager(self) -> DeviceOutputLeaseManager:
        """Return the single output reservation authority for this batch."""
        return self.items[0].output_lease.owner

    @property
    def version(self) -> int:
        """Return the homogeneous portable-record version."""
        return self.items[0].validated_record.header.version

    @property
    def stored_format(self) -> StoredCompressionFormat:
        """Return the homogeneous codec, framing, and transform identity."""
        return self.items[0].validated_record.header.stored_format

    @property
    def record_count(self) -> int:
        """Return the number of portable records in this request."""
        return self._record_count

    @property
    def compression_chunk_count(self) -> int:
        """Return the flattened number of independently compressed chunks."""
        return self._compression_chunk_count

    @property
    def total_record_bytes(self) -> int:
        """Return complete record bytes copied to backend-owned device input."""
        return self._total_record_bytes

    @property
    def total_compressed_payload_bytes(self) -> int:
        """Return descriptor-declared compressed bytes excluding metadata."""
        return self._total_compressed_payload_bytes

    @property
    def total_uncompressed_bytes(self) -> int:
        """Return the exact aggregate bytes expected in leased outputs."""
        return self._total_uncompressed_bytes

    def validate_active_leases(self) -> None:
        """Fail if any output lease was released after construction.

        Raises:
            GpuDecompressRequestError: If an item's lease is no longer active.

        Notes:
            A backend calls this immediately before taking submission
            ownership. The later submission contract must sequence that
            ownership transfer with release. This check does not replace live
            pointer, alignment, or overlap revalidation.
        """
        for index, item in enumerate(self.items):
            if not item.output_lease.is_active:
                raise GpuDecompressRequestError(
                    f"items[{index}] output lease is not active"
                )
