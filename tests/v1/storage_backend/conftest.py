# SPDX-License-Identifier: Apache-2.0
"""conftest for storage_backend tests — blkio throughput summary hook."""


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print a throughput summary table with GB/s after benchmarks."""
    try:
        from tests.v1.storage_backend.test_blkio_block_device import (
            _throughput_results,
        )
    except ImportError:
        return
    if not _throughput_results:
        return
    terminalreporter.section("blkio throughput summary")
    hdr = (
        f"{'Test':<40} {'Size':>6} {'Mean':>10} "
        f"{'Min GB/s':>10} {'Mean GB/s':>10} {'Max GB/s':>10}"
    )
    terminalreporter.line(hdr)
    terminalreporter.line("-" * len(hdr))
    for r in _throughput_results:
        terminalreporter.line(
            f"{r['label']:<40} {r['size_mb']:>5.0f}M "
            f"{r['mean_us']:>9.0f}us "
            f"{r['min_gbps']:>9.2f} "
            f"{r['mean_gbps']:>9.2f} "
            f"{r['max_gbps']:>9.2f}"
        )
