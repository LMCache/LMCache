# SPDX-License-Identifier: Apache-2.0
"""Run one bench case per client concurrently and fold the results."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING
import threading

# First Party
from lmcache.cli.commands.bench.server_bench.cases.base import BenchCase, BenchResult

if TYPE_CHECKING:
    # First Party
    from lmcache.cli.commands.bench.server_bench.client import ServerBenchClient


def run_concurrently(
    clients: Sequence[ServerBenchClient],
    cases: Sequence[BenchCase],
    stop: threading.Event,
    log: Callable[[str], None],
) -> BenchResult:
    """Run ``cases[i]`` on ``clients[i]``, each in its own thread.

    Cases should observe ``stop`` between requests; it is set on Ctrl-C and
    when any case raises, so the remaining clients wind down instead of
    running on alone.

    Args:
        clients: Started clients, one per case.
        cases: Bench cases to run.
        stop: Event shared with the cases.
        log: Progress logger.

    Returns:
        The merged result of every case.

    Raises:
        ValueError: If ``clients`` and ``cases`` differ in length.
        Exception: The first exception raised by a case, after all threads
            have finished.
    """
    if len(clients) != len(cases):
        raise ValueError(
            f"got {len(clients)} clients for {len(cases)} cases; need one each"
        )

    results: list[BenchResult | None] = [None] * len(cases)
    errors: list[BaseException] = []

    def run_one(index: int) -> None:
        try:
            results[index] = cases[index].run(clients[index], log)
        except BaseException as exc:
            errors.append(exc)
            stop.set()

    threads = [
        threading.Thread(target=run_one, args=(index,), name=f"bench-client-{index}")
        for index in range(len(cases))
    ]
    for thread in threads:
        thread.start()

    interrupted = False
    try:
        for thread in threads:
            while thread.is_alive():
                thread.join(timeout=0.2)
    except KeyboardInterrupt:
        interrupted = True
        stop.set()
        for thread in threads:
            thread.join()

    if errors:
        raise errors[0]

    merged = BenchResult(case_name=cases[0].name if cases else "")
    for result in results:
        if result is not None:
            merged.merge(result)
    merged.interrupted = merged.interrupted or interrupted
    return merged
