# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from examples.serde.turboquant.bench_serde_baselines import (
    build_service_profiles,
    percentile,
    transfer_time_ms,
)


def test_percentile_interpolates_samples() -> None:
    """Percentiles use linear interpolation between adjacent samples."""
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


@pytest.mark.parametrize("quantile", [-0.1, 1.1])
def test_percentile_rejects_invalid_quantile(quantile: float) -> None:
    """Percentiles reject values outside the closed unit interval."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        percentile([1.0], quantile)


def test_transfer_time_uses_gigabits_per_second() -> None:
    """A one-gigabit payload takes one second over a one-Gbps link."""
    assert transfer_time_ms(125_000_000, 1.0) == pytest.approx(1000.0)


def test_service_profile_reports_break_even_bandwidth() -> None:
    """The model identifies links where compression saves end-to-end time."""
    profiles = build_service_profiles(
        raw_bytes=1_000_000_000,
        serialized_bytes=500_000_000,
        encode_ms=10.0,
        decode_ms=10.0,
        bandwidths_gbps=[100.0, 400.0],
    )

    assert profiles[0]["break_even_bandwidth_gbps"] == pytest.approx(200.0)
    assert profiles[0]["beneficial"] is True
    assert profiles[0]["speedup_vs_raw_transfer"] == pytest.approx(4 / 3)
    assert profiles[1]["beneficial"] is False
    assert profiles[1]["speedup_vs_raw_transfer"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"raw_bytes": 0}, "raw_bytes must be positive"),
        ({"serialized_bytes": -1}, "serialized_bytes must be non-negative"),
        ({"encode_ms": -1.0}, "codec latencies must be non-negative"),
        ({"bandwidths_gbps": [0.0]}, "bandwidth_gbps must be positive"),
    ],
)
def test_service_profile_rejects_invalid_inputs(
    kwargs: dict[str, object], message: str
) -> None:
    """Invalid size, latency, and bandwidth inputs fail clearly."""
    inputs: dict[str, object] = {
        "raw_bytes": 100,
        "serialized_bytes": 50,
        "encode_ms": 1.0,
        "decode_ms": 1.0,
        "bandwidths_gbps": [10.0],
    }
    inputs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        build_service_profiles(**inputs)  # type: ignore[arg-type]
