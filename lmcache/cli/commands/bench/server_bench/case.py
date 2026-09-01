# SPDX-License-Identifier: Apache-2.0
"""Shared interfaces and results for server-bench cases."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    # First Party
    from lmcache.cli.commands.bench.server_bench.client import ServerBenchClient


@dataclass
class BenchResult:
    """Structured checks and measurements from one bench-case run."""

    case_name: str
    completed_runs: int = 0
    checks: dict[str, list[bool]] = field(default_factory=dict)
    latencies_ms: dict[str, list[float]] = field(default_factory=dict)
    interrupted: bool = False

    def record_check(self, name: str, passed: bool) -> None:
        """Append a correctness result under the given name."""
        self.checks.setdefault(name, []).append(passed)

    def record_latency(self, name: str, latency_ms: float) -> None:
        """Append a latency sample under the given name."""
        self.latencies_ms.setdefault(name, []).append(latency_ms)

    def passed_count(self, name: str) -> int:
        """Return the number of passing checks with the given name."""
        return sum(self.checks.get(name, ()))

    def failed_count(self, name: str) -> int:
        """Return the number of failing checks with the given name."""
        return sum(not passed for passed in self.checks.get(name, ()))

    @property
    def succeeded(self) -> bool:
        """Return whether all recorded checks passed."""
        return (
            not self.interrupted
            and bool(self.checks)
            and all(passed for results in self.checks.values() for passed in results)
        )


class BenchCase(Protocol):
    """Executable server-bench case contract."""

    @property
    def name(self) -> str:
        """Return the stable case name."""
        ...

    def run(
        self,
        client: ServerBenchClient,
        log: Callable[[str], None],
    ) -> BenchResult:
        """Run the case with an already-started client.

        Args:
            client: Started server-bench client.
            log: Progress logger.

        Returns:
            Structured case result.
        """
        ...
