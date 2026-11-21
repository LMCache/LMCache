# SPDX-License-Identifier: Apache-2.0
"""Tests for tensor metrics used by the TurboQuant benchmark."""

# Standard
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import math

# Third Party
import pytest

torch = pytest.importorskip("torch")

_UTILS_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "serde"
    / "turboquant"
    / "benchmark_tensor_utils.py"
)
_SPEC = spec_from_file_location("turboquant_benchmark_tensor_utils", _UTILS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_UTILS = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_UTILS)


def test_tensor_error_metrics_streams_across_chunks() -> None:
    original = torch.tensor([1.0, 2.0, 3.0, 4.0])
    recovered = torch.tensor([2.0, 4.0, 6.0, 8.0])

    metrics = _UTILS.tensor_error_metrics(original, recovered, chunk_elements=3)

    assert metrics.corr == pytest.approx(1.0)
    assert metrics.mean_abs_err == pytest.approx(2.5)
    assert metrics.max_abs_err == pytest.approx(4.0)


def test_tensor_error_metrics_reports_negative_correlation() -> None:
    original = torch.tensor([1.0, 2.0, 3.0])
    recovered = torch.tensor([3.0, 2.0, 1.0])

    metrics = _UTILS.tensor_error_metrics(original, recovered, chunk_elements=2)

    assert metrics.corr == pytest.approx(-1.0)


def test_tensor_error_metrics_returns_nan_for_constant_input() -> None:
    original = torch.ones(4)
    recovered = torch.arange(4, dtype=torch.float32)

    metrics = _UTILS.tensor_error_metrics(original, recovered, chunk_elements=2)

    assert math.isnan(metrics.corr)


@pytest.mark.parametrize(
    ("original", "recovered", "chunk_elements", "match"),
    [
        (torch.ones(2), torch.ones(3), 2, "same shape"),
        (torch.empty(0), torch.empty(0), 2, "non-empty"),
        (torch.ones(2), torch.ones(2), 0, "chunk_elements"),
    ],
)
def test_tensor_error_metrics_rejects_invalid_input(
    original,
    recovered,
    chunk_elements: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _UTILS.tensor_error_metrics(original, recovered, chunk_elements)
