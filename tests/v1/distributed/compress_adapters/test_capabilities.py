# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for GPU decompression capability profiles."""

# Standard
from dataclasses import FrozenInstanceError

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.compress_adapters.capabilities import (
    GpuDecompressBackendStability,
    GpuDecompressCapabilities,
    GpuDecompressExecutionPolicy,
    SupportedCompressedRecordFormat,
)
from lmcache.v1.distributed.compress_adapters.device import DeviceIdentity
from lmcache.v1.distributed.compress_adapters.format import (
    CompressionCodec,
    CompressionFraming,
    PostDecompressTransform,
    StoredCompressionFormat,
)

_NVIDIA_DEVICE = DeviceIdentity(
    device_type="cuda",
    backend_name="cuda",
    device_index=0,
)
_AMD_DEVICE = DeviceIdentity(
    device_type="cuda",
    backend_name="rocm",
    device_index=0,
)


def _stored_format(
    framing: CompressionFraming = CompressionFraming.RAW,
) -> StoredCompressionFormat:
    """Select Deflate framing with no post-decompression transform.

    Args:
        framing: Raw or Gzip framing for the advertised format.

    Returns:
        The complete stored-format identity for the selected framing.
    """
    return StoredCompressionFormat(
        codec=CompressionCodec.DEFLATE,
        framing=framing,
        post_decompress_transform=PostDecompressTransform.NONE,
    )


def _format_support(
    framing: CompressionFraming = CompressionFraming.RAW,
    *,
    record_version: int = 1,
) -> SupportedCompressedRecordFormat:
    """Associate a test Deflate format with a portable-record version.

    Args:
        framing: Raw or Gzip framing to advertise.
        record_version: Positive unsigned-byte version, including future ones.

    Returns:
        A hashable format identity suitable for a capability profile.

    Raises:
        TypeError: If ``record_version`` is not an exact int.
        ValueError: If ``record_version`` is outside ``[1, 255]``.
    """
    return SupportedCompressedRecordFormat(
        record_version=record_version,
        stored_format=_stored_format(framing),
    )


def _profile(**overrides: object) -> GpuDecompressCapabilities:
    """Construct a concrete-device profile with targeted test overrides.

    Args:
        overrides: Capability constructor field names mapped to replacement
            values. Tests may supply invalid values to exercise validation.

    Returns:
        A profile with small explicit limits for CUDA device zero, unless
        overridden by the test.

    Raises:
        TypeError: If an override has an unknown field name or wrong type.
        ValueError: If the resulting profile violates capability constraints.
    """
    arguments: dict[str, object] = {
        "decompress_backend_name": "nvcomp",
        "library_name": "libnvcomp",
        "library_version": "5.3.0.16",
        "stability": GpuDecompressBackendStability.STABLE,
        "device": _NVIDIA_DEVICE,
        "supported_record_formats": frozenset({_format_support()}),
        "supported_execution_policies": frozenset(
            {
                GpuDecompressExecutionPolicy.ANY_GPU_ENGINE,
                GpuDecompressExecutionPolicy.SOFTWARE_GPU_REQUIRED,
            }
        ),
        "max_compressed_chunk_bytes": 64,
        "max_uncompressed_chunk_bytes": 128,
        "max_records": 4,
        "max_compression_chunks": 8,
        "max_total_record_bytes": 512,
        "max_total_compressed_payload_bytes": 256,
        "max_total_uncompressed_bytes": 512,
        "max_active_submissions": 2,
        "input_alignment_bytes": 16,
        "output_alignment_bytes": 16,
        "workspace_alignment_bytes": 64,
        "is_asynchronous": True,
    }
    arguments.update(overrides)
    return GpuDecompressCapabilities(**arguments)  # type: ignore[arg-type]


def test_execution_policy_values_are_stable() -> None:
    """Caller-facing policy strings retain their documented meanings."""
    assert GpuDecompressExecutionPolicy.ANY_GPU_ENGINE.value == "any_gpu_engine"
    assert (
        GpuDecompressExecutionPolicy.SOFTWARE_GPU_REQUIRED.value
        == "software_gpu_required"
    )
    assert (
        GpuDecompressExecutionPolicy.FIXED_FUNCTION_REQUIRED.value
        == "fixed_function_required"
    )


def test_backend_stability_values_are_stable() -> None:
    """Capability diagnostics use a small explicit stability vocabulary."""
    assert GpuDecompressBackendStability.STABLE.value == "stable"
    assert GpuDecompressBackendStability.EXPERIMENTAL.value == "experimental"


def test_unknown_enum_values_are_rejected() -> None:
    """Unknown policies and stability values fail closed."""
    with pytest.raises(ValueError):
        GpuDecompressExecutionPolicy("cpu_fallback")
    with pytest.raises(ValueError):
        GpuDecompressBackendStability("unsupported")


def test_supported_record_format_is_hashable() -> None:
    """Complete format identities work as immutable capability members."""
    first = _format_support()
    second = _format_support()

    assert first == second
    assert first in frozenset({second})


@pytest.mark.parametrize(
    ("record_version", "exception"),
    [
        (True, TypeError),
        (0, ValueError),
        (-1, ValueError),
        (256, ValueError),
    ],
)
def test_supported_record_format_validates_version(
    record_version: int,
    exception: type[Exception],
) -> None:
    """Record versions retain the positive unsigned-byte contract."""
    with pytest.raises(exception, match="record_version"):
        SupportedCompressedRecordFormat(
            record_version=record_version,
            stored_format=_stored_format(),
        )


def test_supported_record_format_requires_stored_format() -> None:
    """A version cannot be advertised without a typed stored format."""
    with pytest.raises(TypeError, match="StoredCompressionFormat"):
        SupportedCompressedRecordFormat(
            record_version=1,
            stored_format="deflate-raw",  # type: ignore[arg-type]
        )


def test_valid_capability_profile_retains_exact_values() -> None:
    """One immutable profile reports its complete concrete-device contract."""
    formats = frozenset(
        {
            _format_support(),
            _format_support(CompressionFraming.GZIP),
        }
    )
    profile = _profile(supported_record_formats=formats)

    assert profile.decompress_backend_name == "nvcomp"
    assert profile.library_name == "libnvcomp"
    assert profile.library_version == "5.3.0.16"
    assert profile.stability is GpuDecompressBackendStability.STABLE
    assert profile.device == _NVIDIA_DEVICE
    assert profile.supported_record_formats == formats
    assert profile.max_compressed_chunk_bytes == 64
    assert profile.max_uncompressed_chunk_bytes == 128
    assert profile.max_records == 4
    assert profile.max_compression_chunks == 8
    assert profile.max_total_record_bytes == 512
    assert profile.max_total_compressed_payload_bytes == 256
    assert profile.max_total_uncompressed_bytes == 512
    assert profile.max_active_submissions == 2
    assert profile.input_alignment_bytes == 16
    assert profile.output_alignment_bytes == 16
    assert profile.workspace_alignment_bytes == 64
    assert profile.is_asynchronous

    with pytest.raises(FrozenInstanceError):
        profile.decompress_backend_name = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("decompress_backend_name", 1, TypeError),
        ("decompress_backend_name", "", ValueError),
        ("decompress_backend_name", " nvcomp", ValueError),
        ("library_name", 1, TypeError),
        ("library_name", "", ValueError),
        ("library_name", "libnvcomp ", ValueError),
        ("library_version", 1, TypeError),
        ("library_version", "", ValueError),
        ("library_version", " 5.3.0.16", ValueError),
    ],
)
def test_capability_profile_validates_names_and_version(
    field: str,
    value: object,
    exception: type[Exception],
) -> None:
    """Diagnostic identifiers are non-empty exact strings."""
    with pytest.raises(exception, match=field):
        _profile(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "exception", "message"),
    [
        ("stability", "stable", TypeError, "stability"),
        ("device", "cuda:0", TypeError, "DeviceIdentity"),
        (
            "supported_record_formats",
            (_format_support(),),
            TypeError,
            "frozenset",
        ),
        ("supported_record_formats", frozenset(), ValueError, "must not be empty"),
        (
            "supported_record_formats",
            frozenset({"deflate"}),
            TypeError,
            "members",
        ),
        (
            "supported_execution_policies",
            (GpuDecompressExecutionPolicy.ANY_GPU_ENGINE,),
            TypeError,
            "frozenset",
        ),
        (
            "supported_execution_policies",
            frozenset(),
            ValueError,
            "must not be empty",
        ),
        (
            "supported_execution_policies",
            frozenset({"any_gpu_engine"}),
            TypeError,
            "members",
        ),
        (
            "supported_execution_policies",
            frozenset({GpuDecompressExecutionPolicy.SOFTWARE_GPU_REQUIRED}),
            ValueError,
            "include any_gpu_engine",
        ),
        ("is_asynchronous", 1, TypeError, "bool"),
        ("is_asynchronous", "yes", TypeError, "bool"),
    ],
)
def test_capability_profile_validates_structural_fields(
    field: str,
    value: object,
    exception: type[Exception],
    message: str,
) -> None:
    """Profiles reject mutable collections and untyped public values."""
    with pytest.raises(exception, match=message):
        _profile(**{field: value})


_POSITIVE_INTEGER_FIELDS = (
    "max_compressed_chunk_bytes",
    "max_uncompressed_chunk_bytes",
    "max_records",
    "max_compression_chunks",
    "max_total_record_bytes",
    "max_total_compressed_payload_bytes",
    "max_total_uncompressed_bytes",
    "max_active_submissions",
    "input_alignment_bytes",
    "output_alignment_bytes",
    "workspace_alignment_bytes",
)


@pytest.mark.parametrize("field", _POSITIVE_INTEGER_FIELDS)
@pytest.mark.parametrize(
    ("value", "exception"),
    [
        (True, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_capability_profile_requires_positive_integer_bounds(
    field: str,
    value: object,
    exception: type[Exception],
) -> None:
    """Every capacity and alignment rejects booleans and nonpositive values."""
    with pytest.raises(exception, match=field):
        _profile(**{field: value})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"max_total_compressed_payload_bytes": 63},
            "max_total_compressed_payload_bytes",
        ),
        (
            {"max_total_uncompressed_bytes": 127},
            "max_total_uncompressed_bytes",
        ),
        ({"max_total_record_bytes": 63}, "max_total_record_bytes"),
    ],
)
def test_capability_profile_rejects_contradictory_aggregate_limits(
    overrides: dict[str, object],
    message: str,
) -> None:
    """Related aggregates cannot be below one chunk's payload or output."""
    with pytest.raises(ValueError, match=message):
        _profile(**overrides)


def test_record_and_chunk_limits_remain_independent() -> None:
    """Empty records permit a record cap larger than the stream cap."""
    profile = _profile(max_records=8, max_compression_chunks=1)

    assert profile.max_records == 8
    assert profile.max_compression_chunks == 1


def test_minimum_positive_bounds_and_synchronous_profile_are_valid() -> None:
    """The generic contract does not impose vendor-specific minimums."""
    profile = _profile(
        max_compressed_chunk_bytes=1,
        max_uncompressed_chunk_bytes=1,
        max_records=1,
        max_compression_chunks=1,
        max_total_record_bytes=1,
        max_total_compressed_payload_bytes=1,
        max_total_uncompressed_bytes=1,
        max_active_submissions=1,
        input_alignment_bytes=1,
        output_alignment_bytes=1,
        workspace_alignment_bytes=1,
        is_asynchronous=False,
    )

    assert profile.max_total_record_bytes == 1
    assert not profile.is_asynchronous


def test_alignment_need_not_be_a_power_of_two() -> None:
    """Generic profiles support modulo-based positive alignment contracts."""
    profile = _profile(
        input_alignment_bytes=1,
        output_alignment_bytes=3,
        workspace_alignment_bytes=16,
    )

    assert profile.input_alignment_bytes == 1
    assert profile.output_alignment_bytes == 3
    assert profile.workspace_alignment_bytes == 16


def test_supports_record_format_uses_version_and_stored_identity() -> None:
    """Format support includes version, codec, framing, and transform."""
    profile = _profile()

    assert profile.supports_record_format(
        record_version=1,
        stored_format=_stored_format(),
    )
    assert not profile.supports_record_format(
        record_version=1,
        stored_format=_stored_format(CompressionFraming.GZIP),
    )
    assert not profile.supports_record_format(
        record_version=2,
        stored_format=_stored_format(),
    )


def test_supports_record_format_validates_arguments() -> None:
    """Format queries preserve the supported-format constructor contract."""
    profile = _profile()

    with pytest.raises(TypeError, match="record_version"):
        profile.supports_record_format(
            record_version=True,
            stored_format=_stored_format(),
        )
    assert not profile.supports_record_format(
        record_version=0,
        stored_format=_stored_format(),
    )
    assert not profile.supports_record_format(
        record_version=256,
        stored_format=_stored_format(),
    )


@pytest.mark.parametrize("record_version", [-1, 0, 1, 256])
def test_supports_record_format_rejects_wrong_format_type_for_any_version(
    record_version: int,
) -> None:
    """An unsupported version cannot hide a malformed format argument."""
    profile = _profile()

    with pytest.raises(TypeError, match="StoredCompressionFormat"):
        profile.supports_record_format(
            record_version=record_version,
            stored_format="raw",  # type: ignore[arg-type]
        )


def test_future_record_version_can_be_advertised_and_queried() -> None:
    """Capability vocabulary can describe versions beyond the active parser."""
    future_format = _format_support(record_version=255)
    profile = _profile(supported_record_formats=frozenset({future_format}))

    assert profile.supports_record_format(
        record_version=255,
        stored_format=_stored_format(),
    )


def test_supports_execution_policy() -> None:
    """Strict policies are advertised independently of permissive execution."""
    profile = _profile()

    assert profile.supports_execution_policy(
        GpuDecompressExecutionPolicy.ANY_GPU_ENGINE
    )
    assert profile.supports_execution_policy(
        GpuDecompressExecutionPolicy.SOFTWARE_GPU_REQUIRED
    )
    assert not profile.supports_execution_policy(
        GpuDecompressExecutionPolicy.FIXED_FUNCTION_REQUIRED
    )
    with pytest.raises(TypeError, match="GpuDecompressExecutionPolicy"):
        profile.supports_execution_policy("any_gpu_engine")  # type: ignore[arg-type]


def test_supports_exact_device_identity() -> None:
    """CUDA, ROCm, and concrete device indices cannot be conflated."""
    profile = _profile()

    assert profile.supports_device(_NVIDIA_DEVICE)
    assert not profile.supports_device(
        DeviceIdentity(
            device_type="cuda",
            backend_name="cuda",
            device_index=1,
        )
    )
    assert not profile.supports_device(_AMD_DEVICE)
    with pytest.raises(TypeError, match="DeviceIdentity"):
        profile.supports_device("cuda:0")  # type: ignore[arg-type]
