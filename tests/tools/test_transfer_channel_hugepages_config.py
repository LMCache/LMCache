# SPDX-License-Identifier: Apache-2.0
"""Configuration checks for a comparable HugeTLB transfer benchmark."""

# Standard
import argparse

# Third Party
import pytest

# First Party
from lmcache.tools.transfer_channel_benchmark.config import (
    add_benchmark_arguments,
    build_config,
)


def test_benchmark_hugepages_and_anonymous_baseline_options() -> None:
    """The CLI selects either HugeTLB or eager anonymous DRAM with no SHM."""
    parser = argparse.ArgumentParser()
    add_benchmark_arguments(parser)
    hugepages = build_config(parser.parse_args(["--role", "server", "--use-hugepages"]))
    baseline = build_config(parser.parse_args(["--role", "server", "--disable-shm"]))
    assert hugepages.use_hugepages is True
    assert hugepages.use_lazy is False
    assert baseline.disable_shm is True
    assert baseline.use_hugepages is False


def test_benchmark_rejects_lazy_hugepages() -> None:
    """Benchmark settings reject a combination the L1 cannot allocate."""
    parser = argparse.ArgumentParser()
    add_benchmark_arguments(parser)
    with pytest.raises(ValueError, match="--use-hugepages conflicts with --use-lazy"):
        build_config(
            parser.parse_args(["--role", "server", "--use-hugepages", "--use-lazy"])
        )
