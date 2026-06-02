# SPDX-License-Identifier: Apache-2.0

"""Tests for the L0 E2E Prometheus scrape assertion tool."""

# Standard
import importlib.util
import sys
from pathlib import Path

# Third Party
import pytest

_SCRIPT = (
    Path(__file__).parents[2]
    / "tools"
    / "mp_observability"
    / "assert_l0_e2e_metrics.py"
)
_SPEC = importlib.util.spec_from_file_location("assert_l0_e2e_metrics", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
assert_l0_e2e_metrics = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = assert_l0_e2e_metrics
_SPEC.loader.exec_module(assert_l0_e2e_metrics)


def test_parse_positive_samples_ignores_zero_and_comments() -> None:
    scrape = """
# HELP lmcache_mp_l0_l1_load_requests_total example
lmcache_mp_l0_l1_load_requests_total{worker_id="0"} 1
lmcache_mp_l0_l1_load_bytes_total{worker_id="0"} 0
lmcache_mp_num_chunks_loaded_total{worker_id="0"} 2.0
"""

    positive = assert_l0_e2e_metrics.parse_positive_samples(scrape)

    assert "lmcache_mp_l0_l1_load_requests_total" in positive
    assert "lmcache_mp_num_chunks_loaded_total" in positive
    assert "lmcache_mp_l0_l1_load_bytes_total" not in positive


def test_l0_cpu_gpu_scope_accepts_positive_required_metrics(tmp_path: Path) -> None:
    scrape = tmp_path / "scrape.prom"
    scrape.write_text(
        "\n".join(
            [
                "lmcache_mp_l0_l1_load_requests_total 1",
                "lmcache_mp_l0_l1_load_bytes_total 4096",
                "lmcache_mp_num_chunks_loaded_total 2",
                "lmcache_mp_l0_l1_load_throughput_GB_per_second_count 1",
            ]
        ),
        encoding="utf-8",
    )

    assert assert_l0_e2e_metrics.main([str(scrape)]) == 0


def test_full_e2e_scope_requires_l2_and_l0_allocation_metrics(
    tmp_path: Path,
) -> None:
    scrape = tmp_path / "scrape.prom"
    scrape.write_text(
        "\n".join(
            [
                "lmcache_mp_l0_l1_load_requests_total 1",
                "lmcache_mp_l0_l1_load_bytes_total 4096",
                "lmcache_mp_num_chunks_loaded_total 2",
                "lmcache_mp_l0_l1_load_throughput_GB_per_second_count 1",
            ]
        ),
        encoding="utf-8",
    )

    assert assert_l0_e2e_metrics.main([str(scrape), "--scope", "full-e2e"]) == 1


@pytest.mark.parametrize(
    "sample_name",
    [
        "lmcache_mp_l0_l1_load_requests_total",
        "lmcache_mp_l0_l1_load_bytes_total",
        "lmcache_mp_num_chunks_loaded_total",
        "lmcache_mp_l0_l1_load_throughput_GB_per_second_count",
    ],
)
def test_l0_cpu_gpu_scope_reports_each_missing_metric(
    sample_name: str,
) -> None:
    positive_samples = {
        "lmcache_mp_l0_l1_load_requests_total",
        "lmcache_mp_l0_l1_load_bytes_total",
        "lmcache_mp_num_chunks_loaded_total",
        "lmcache_mp_l0_l1_load_throughput_GB_per_second_count",
    }
    positive_samples.remove(sample_name)

    missing = assert_l0_e2e_metrics.missing_required_metrics(
        positive_samples,
        assert_l0_e2e_metrics.L0_CPU_GPU_REQUIRED,
    )

    assert [metric.sample_name for metric in missing] == [sample_name]
