# SPDX-License-Identifier: Apache-2.0
"""Vendor-neutral GPU decompression policies and capability profiles."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
import enum

# First Party
from lmcache.v1.distributed.compress_adapters.device import DeviceIdentity
from lmcache.v1.distributed.compress_adapters.format import StoredCompressionFormat

_UINT8_MAX = (1 << 8) - 1


def _require_nonempty_string(name: str, value: str) -> None:
    """Validate a capability identifier without silently trimming its value.

    Args:
        name: Field name used in validation errors.
        value: Identifier that must be non-empty and have no outer whitespace.

    Returns:
        None when the value is valid.

    Raises:
        TypeError: If ``value`` is not a string.
        ValueError: If ``value`` is empty or contains surrounding whitespace.
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{name} must not be empty")
    if value != value.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace")


def _require_positive_int(name: str, value: int) -> None:
    """Validate a positive capacity or alignment, excluding boolean values.

    Args:
        name: Field name used in validation errors.
        value: Integer that must be greater than zero.

    Returns:
        None when the value is valid.

    Raises:
        TypeError: If ``value`` is not an exact ``int``, including ``bool``.
        ValueError: If ``value`` is zero or negative.
    """
    if type(value) is not int:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}")


class GpuDecompressExecutionPolicy(str, enum.Enum):
    """Required class of GPU decompression execution.

    ``ANY_GPU_ENGINE`` allows the backend to choose any supported GPU engine,
    ``SOFTWARE_GPU_REQUIRED`` requires ordinary GPU compute execution, and
    ``FIXED_FUNCTION_REQUIRED`` requires a dedicated hardware decompressor.
    No policy permits CPU fallback.
    """

    ANY_GPU_ENGINE = "any_gpu_engine"
    SOFTWARE_GPU_REQUIRED = "software_gpu_required"
    FIXED_FUNCTION_REQUIRED = "fixed_function_required"


class GpuDecompressBackendStability(str, enum.Enum):
    """Conservative stability level of one complete capability profile."""

    STABLE = "stable"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True, kw_only=True, slots=True)
class SupportedCompressedRecordFormat:
    """Portable record identity accepted by a decompression profile.

    Args:
        record_version: Positive unsigned 8-bit portable-record version.
        stored_format: Codec, framing, and post-decompression transform.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``record_version`` is outside ``[1, 255]``.

    Notes:
        This value may describe a future version even when the current parser
        cannot yet construct that record. Backend validation still rejects
        any request that the active parser cannot produce.
    """

    record_version: int
    stored_format: StoredCompressionFormat

    def __post_init__(self) -> None:
        if type(self.record_version) is not int:
            raise TypeError(
                "record_version must be an int, got "
                f"{type(self.record_version).__name__}"
            )
        if self.record_version <= 0 or self.record_version > _UINT8_MAX:
            raise ValueError(
                f"record_version must be in [1, {_UINT8_MAX}], "
                f"got {self.record_version}"
            )
        if not isinstance(self.stored_format, StoredCompressionFormat):
            raise TypeError(
                "stored_format must be a StoredCompressionFormat, got "
                f"{type(self.stored_format).__name__}"
            )


@dataclass(frozen=True, kw_only=True, slots=True)
class GpuDecompressCapabilities:
    """Immutable decompression capabilities for one concrete device.

    Args:
        decompress_backend_name: LMCache decompression implementation name, such as
            ``"nvcomp"`` or ``"hipcomp"``.
        library_name: Installed native dependency name.
        library_version: Opaque installed native dependency version.
        stability: Conservative stability of the complete profile.
        device: Exact device for which all limits and policies apply.
        supported_record_formats: Non-empty immutable set of portable formats.
        supported_execution_policies: Non-empty immutable policy set including
            :attr:`GpuDecompressExecutionPolicy.ANY_GPU_ENGINE`.
        max_compressed_chunk_bytes: Maximum bytes in one compressed stream.
        max_uncompressed_chunk_bytes: Maximum output bytes from one stream.
        max_records: Maximum records in one submission.
        max_compression_chunks: Maximum flattened streams in one submission.
        max_total_record_bytes: Maximum complete input-record bytes per
            submission, including headers and alignment gaps.
        max_total_compressed_payload_bytes: Maximum descriptor-declared stream
            bytes per submission.
        max_total_uncompressed_bytes: Maximum output bytes per submission.
        max_active_submissions: Maximum concurrently owned submissions.
        input_alignment_bytes: Required native device-input alignment.
        output_alignment_bytes: Required native output alignment.
        workspace_alignment_bytes: Required native workspace alignment.
        is_asynchronous: Whether submission may complete after returning.

    Raises:
        TypeError: If a field or collection member has the wrong type.
        ValueError: If a name is empty or padded, a collection is empty, a
            required policy is absent, a numeric bound is nonpositive, or an
            aggregate limit contradicts a per-chunk maximum.

    Notes:
        This value reports facts; it does not validate a request or reserve
        resources. Concrete backends derive profiles from their installed
        library, selected device, and execution engine without importing
        vendor types into this module. A synchronous profile still returns a
        completion that callers must validate or discard so resources follow
        one finalization contract.
    """

    decompress_backend_name: str
    library_name: str
    library_version: str
    stability: GpuDecompressBackendStability
    device: DeviceIdentity
    supported_record_formats: frozenset[SupportedCompressedRecordFormat]
    supported_execution_policies: frozenset[GpuDecompressExecutionPolicy]
    max_compressed_chunk_bytes: int
    max_uncompressed_chunk_bytes: int
    max_records: int
    max_compression_chunks: int
    max_total_record_bytes: int
    max_total_compressed_payload_bytes: int
    max_total_uncompressed_bytes: int
    max_active_submissions: int
    input_alignment_bytes: int
    output_alignment_bytes: int
    workspace_alignment_bytes: int
    is_asynchronous: bool

    def __post_init__(self) -> None:
        _require_nonempty_string(
            "decompress_backend_name",
            self.decompress_backend_name,
        )
        _require_nonempty_string("library_name", self.library_name)
        _require_nonempty_string("library_version", self.library_version)

        if not isinstance(self.stability, GpuDecompressBackendStability):
            raise TypeError(
                "stability must be a GpuDecompressBackendStability, got "
                f"{type(self.stability).__name__}"
            )
        if not isinstance(self.device, DeviceIdentity):
            raise TypeError(
                f"device must be a DeviceIdentity, got {type(self.device).__name__}"
            )

        self._validate_supported_record_formats()
        self._validate_supported_execution_policies()

        for name, value in (
            ("max_compressed_chunk_bytes", self.max_compressed_chunk_bytes),
            ("max_uncompressed_chunk_bytes", self.max_uncompressed_chunk_bytes),
            ("max_records", self.max_records),
            ("max_compression_chunks", self.max_compression_chunks),
            ("max_total_record_bytes", self.max_total_record_bytes),
            (
                "max_total_compressed_payload_bytes",
                self.max_total_compressed_payload_bytes,
            ),
            ("max_total_uncompressed_bytes", self.max_total_uncompressed_bytes),
            ("max_active_submissions", self.max_active_submissions),
            ("input_alignment_bytes", self.input_alignment_bytes),
            ("output_alignment_bytes", self.output_alignment_bytes),
            ("workspace_alignment_bytes", self.workspace_alignment_bytes),
        ):
            _require_positive_int(name, value)

        if type(self.is_asynchronous) is not bool:
            raise TypeError(
                "is_asynchronous must be a bool, got "
                f"{type(self.is_asynchronous).__name__}"
            )

        if self.max_total_compressed_payload_bytes < self.max_compressed_chunk_bytes:
            raise ValueError(
                "max_total_compressed_payload_bytes must be greater than or "
                "equal to max_compressed_chunk_bytes"
            )
        if self.max_total_uncompressed_bytes < self.max_uncompressed_chunk_bytes:
            raise ValueError(
                "max_total_uncompressed_bytes must be greater than or equal "
                "to max_uncompressed_chunk_bytes"
            )
        if self.max_total_record_bytes < self.max_compressed_chunk_bytes:
            raise ValueError(
                "max_total_record_bytes must be greater than or equal to "
                "max_compressed_chunk_bytes"
            )

    def supports_record_format(
        self,
        *,
        record_version: int,
        stored_format: StoredCompressionFormat,
    ) -> bool:
        """Return whether this profile accepts one portable format.

        Args:
            record_version: Positive unsigned 8-bit record version.
            stored_format: Codec, framing, and transform identity.

        Returns:
            ``True`` when the complete format identity is advertised. Integer
            versions outside ``[1, 255]`` return ``False``.

        Raises:
            TypeError: If an argument has the wrong type.
        """
        if type(record_version) is not int:
            raise TypeError(
                f"record_version must be an int, got {type(record_version).__name__}"
            )
        if not isinstance(stored_format, StoredCompressionFormat):
            raise TypeError(
                "stored_format must be a StoredCompressionFormat, got "
                f"{type(stored_format).__name__}"
            )
        if record_version <= 0 or record_version > _UINT8_MAX:
            return False
        record_format = SupportedCompressedRecordFormat(
            record_version=record_version,
            stored_format=stored_format,
        )
        return record_format in self.supported_record_formats

    def supports_execution_policy(
        self,
        policy: GpuDecompressExecutionPolicy,
    ) -> bool:
        """Return whether this profile can enforce ``policy``.

        Args:
            policy: Required GPU execution-engine behavior.

        Returns:
            ``True`` when ``policy`` is advertised.

        Raises:
            TypeError: If ``policy`` has the wrong type.
        """
        if not isinstance(policy, GpuDecompressExecutionPolicy):
            raise TypeError(
                "policy must be a GpuDecompressExecutionPolicy, got "
                f"{type(policy).__name__}"
            )
        return policy in self.supported_execution_policies

    def supports_device(self, device: DeviceIdentity) -> bool:
        """Return whether this profile belongs to ``device``.

        Args:
            device: Exact device identity to compare.

        Returns:
            ``True`` only for the profile's concrete device.

        Raises:
            TypeError: If ``device`` has the wrong type.
        """
        if not isinstance(device, DeviceIdentity):
            raise TypeError(
                f"device must be a DeviceIdentity, got {type(device).__name__}"
            )
        return device == self.device

    def _validate_supported_record_formats(self) -> None:
        """Require a non-empty frozenset of typed portable-record identities."""
        if not isinstance(self.supported_record_formats, frozenset):
            raise TypeError(
                "supported_record_formats must be a frozenset, got "
                f"{type(self.supported_record_formats).__name__}"
            )
        if not self.supported_record_formats:
            raise ValueError("supported_record_formats must not be empty")
        for record_format in self.supported_record_formats:
            if not isinstance(record_format, SupportedCompressedRecordFormat):
                raise TypeError(
                    "supported_record_formats members must be "
                    "SupportedCompressedRecordFormat values, got "
                    f"{type(record_format).__name__}"
                )

    def _validate_supported_execution_policies(self) -> None:
        """Require typed policies including the permissive GPU-engine policy."""
        if not isinstance(self.supported_execution_policies, frozenset):
            raise TypeError(
                "supported_execution_policies must be a frozenset, got "
                f"{type(self.supported_execution_policies).__name__}"
            )
        if not self.supported_execution_policies:
            raise ValueError("supported_execution_policies must not be empty")
        for policy in self.supported_execution_policies:
            if not isinstance(policy, GpuDecompressExecutionPolicy):
                raise TypeError(
                    "supported_execution_policies members must be "
                    "GpuDecompressExecutionPolicy values, got "
                    f"{type(policy).__name__}"
                )
        if (
            GpuDecompressExecutionPolicy.ANY_GPU_ENGINE
            not in self.supported_execution_policies
        ):
            raise ValueError("supported_execution_policies must include any_gpu_engine")
