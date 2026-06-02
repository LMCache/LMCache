# SPDX-License-Identifier: Apache-2.0

"""Assert that a Prometheus scrape proves L0 CPU-to-GPU reuse.

The script is intentionally launch-environment agnostic: it reads a text-format
Prometheus scrape from a file or stdin and checks for positive samples in the
metric families that prove LMCache loaded chunks and bytes across the L1 CPU to
L0 GPU boundary.
"""

# Future
from __future__ import annotations

# Standard
import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


@dataclass(frozen=True)
class RequiredMetric:
    """A required positive Prometheus metric family.

    Args:
        name: Prometheus metric family name. For histograms, this should be the
            base family name, not the generated ``_bucket`` series.
        description: Human-readable reason this metric is required.
        histogram: Whether to validate the generated ``_count`` series instead
            of the base name.
    """

    name: str
    description: str
    histogram: bool = False

    @property
    def sample_name(self) -> str:
        """Return the concrete sample name to check in a scrape."""
        if self.histogram:
            return f"{self.name}_count"
        return self.name


L0_CPU_GPU_REQUIRED = (
    RequiredMetric(
        name="lmcache_mp_l0_l1_load_requests_total",
        description="completed L1 CPU to L0 GPU retrieve operations",
    ),
    RequiredMetric(
        name="lmcache_mp_l0_l1_load_bytes_total",
        description="bytes copied from L1 CPU memory into L0 GPU KV blocks",
    ),
    RequiredMetric(
        name="lmcache_mp_num_chunks_loaded_total",
        description="chunks loaded from LMCache into a vLLM worker",
    ),
    RequiredMetric(
        name="lmcache_mp_l0_l1_load_throughput_GB_per_second",
        description="CPU-to-GPU throughput samples",
        histogram=True,
    ),
)

FULL_E2E_REQUIRED = L0_CPU_GPU_REQUIRED + (
    RequiredMetric(
        name="lmcache_mp_l2_load_completed_requests_total",
        description="completed L2 to L1 load tasks",
    ),
    RequiredMetric(
        name="lmcache_mp_l2_prefetch_hit_chunks_total",
        description="L2 prefetch hit chunks",
    ),
    RequiredMetric(
        name="lmcache_mp_l0_block_allocation_records_total",
        description="vLLM L0 block allocation records observed by LMCache",
    ),
    RequiredMetric(
        name="lmcache_mp_l0_block_allocated_blocks_total",
        description="vLLM L0 block IDs observed by LMCache",
    ),
)


def parse_positive_samples(scrape: str) -> set[str]:
    """Parse positive Prometheus samples from text-format scrape data.

    Args:
        scrape: Prometheus text exposition data.

    Returns:
        Names of metric samples whose parsed value is greater than zero.
    """
    positive: set[str] = set()
    for line in scrape.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _SAMPLE_RE.match(stripped)
        if match is None:
            continue
        if float(match.group("value")) > 0:
            positive.add(match.group("name"))
    return positive


def missing_required_metrics(
    positive_samples: set[str], required: tuple[RequiredMetric, ...]
) -> list[RequiredMetric]:
    """Find required metric families with no positive sample.

    Args:
        positive_samples: Sample names with values greater than zero.
        required: Metric families to require.

    Returns:
        Required metrics whose concrete sample name was not positive.
    """
    return [metric for metric in required if metric.sample_name not in positive_samples]


def read_scrape(path: str, stdin: TextIO) -> str:
    """Read a Prometheus scrape from a path or stdin.

    Args:
        path: File path, or ``-`` to read stdin.
        stdin: Stream used when ``path`` is ``-``.

    Returns:
        Scrape text.
    """
    if path == "-":
        return stdin.read()
    return Path(path).read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the command-line assertion tool.

    Args:
        argv: Optional argument vector. Defaults to ``sys.argv``.

    Returns:
        Process exit code: ``0`` when all required metrics have positive
        samples, otherwise ``1``.
    """
    parser = argparse.ArgumentParser(
        description="Assert that a Prometheus scrape proves LMCache L0 reuse."
    )
    parser.add_argument(
        "scrape",
        help="Path to a Prometheus text scrape, or '-' to read stdin.",
    )
    parser.add_argument(
        "--scope",
        choices=("l0-cpu-gpu", "full-e2e"),
        default="l0-cpu-gpu",
        help=(
            "Metric set to require. 'l0-cpu-gpu' checks the direct L1 CPU to "
            "L0 GPU proof; 'full-e2e' also checks L2 and L0 block-allocation "
            "metrics."
        ),
    )
    args = parser.parse_args(argv)

    required = FULL_E2E_REQUIRED if args.scope == "full-e2e" else L0_CPU_GPU_REQUIRED
    positive_samples = parse_positive_samples(read_scrape(args.scrape, sys.stdin))
    missing = missing_required_metrics(positive_samples, required)
    if not missing:
        print(f"OK: {args.scope} metrics have positive samples")
        return 0

    print(f"Missing positive {args.scope} metrics:", file=sys.stderr)
    for metric in missing:
        print(
            f"- {metric.sample_name}: {metric.description}",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
