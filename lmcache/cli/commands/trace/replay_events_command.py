# SPDX-License-Identifier: Apache-2.0
"""``lmcache trace replay-events`` — replay an events trace into a coordinator."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING
import argparse
import asyncio
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.metrics import Metrics, StreamHandler, get_formatter
from lmcache.logging import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.events_replay import EventsReplayResult


class ReplayEventsCommand(BaseCommand):
    """Replay an events trace, one file per server, into a coordinator."""

    def name(self) -> str:
        return "replay-events"

    def help(self) -> str:
        return "Replay an events trace (one file per server) into a coordinator."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "trace_path",
            metavar="FILE",
            nargs="+",
            help="Events-level .lct files, one per server; merged by wall-clock time.",
        )
        parser.add_argument(
            "--coordinator-url",
            required=True,
            metavar="URL",
            help="Coordinator base URL to deliver the cache-event stream to.",
        )
        parser.add_argument(
            "--speed",
            type=float,
            default=0.0,
            help=(
                "Pacing relative to the recording: 1 replays at the recorded "
                "rate, 2 twice as fast, 0 as fast as the coordinator takes "
                "batches (default: 0)."
            ),
        )
        parser.add_argument(
            "--heartbeat-interval",
            type=float,
            default=5.0,
            metavar="SECONDS",
            help=(
                "Heartbeat each registered server this often, below the "
                "coordinator's instance timeout; 0 sends none (default: 5)."
            ),
        )

    def execute(self, args: argparse.Namespace) -> None:
        run_events_replay(args)


def run_events_replay(args: argparse.Namespace) -> None:
    """Load the files in *args* and deliver their stream to the coordinator.

    Every record is logged as it goes; unless ``--quiet``, a summary follows:
    files, servers, records, batches and how the coordinator counted them.
    Exits with status 2 if a file is not an events trace this build replays,
    and with status 1 if the coordinator cannot be reached or refuses a
    call, at which point the replay stops: a partly delivered stream would
    leave views that look complete and are not.

    Args:
        args: Parsed CLI arguments.
    """
    # Third Party
    import httpx

    # First Party
    from lmcache.v1.mp_coordinator.events_replay import (
        CoordinatorTarget,
        EventsTrace,
        replay,
    )

    try:
        trace = EventsTrace.load(list(args.trace_path))
    except ValueError as e:
        logger.error("trace replay-events: %s", e)
        sys.exit(2)
    logger.info(
        "trace replay-events: files=%d servers=%d records=%d -> %s",
        len(trace.files),
        len(trace.instances),
        len(trace.records),
        args.coordinator_url,
    )

    async def _run() -> EventsReplayResult:
        async with (
            httpx.AsyncClient(timeout=30.0) as client,
            CoordinatorTarget(
                client,
                args.coordinator_url,
                heartbeat_interval=args.heartbeat_interval,
            ) as target,
        ):
            return await replay(trace, target, speed=args.speed)

    try:
        result = asyncio.run(_run())
    except httpx.HTTPError as e:
        logger.error(
            "trace replay-events: stopped, coordinator at %s failed a call: %s",
            args.coordinator_url,
            e,
        )
        sys.exit(1)
    if not getattr(args, "quiet", False):
        _emit_metrics(trace.files, trace.instances, result)


def _emit_metrics(
    files: list[str], instances: list[str], result: EventsReplayResult
) -> None:
    """Print the summary with the shared :class:`Metrics` renderer.

    Args:
        files: The files replayed.
        instances: The servers the stream named.
        result: What the replay delivered.
    """
    metrics = Metrics(title="Events Replay Result")
    metrics.add_handler(StreamHandler(get_formatter("terminal", width=64)))
    overall = metrics.add_section("overall", "Overall")
    overall.add("files", "Files", len(files))
    overall.add("servers", "Servers", len(instances))
    overall.add("records", "Records", result.records)
    overall.add("batches", "Batches delivered", result.batches)
    overall.add("applied", "Batches applied", result.ingested.applied)
    overall.add("duplicates", "Batches duplicate", result.ingested.duplicates)
    overall.add("stale", "Batches stale", result.ingested.stale)
    overall.add("starts", "Start marks", result.starts)
    overall.add("stops", "Stop marks", result.stops)
    overall.add("ended", "Streams cut without stop", result.ended)
    overall.add("skipped", "Records skipped", result.skipped)
    overall.add("duration", "Replay duration (s)", round(result.duration_s, 3))
    metrics.emit()
