# SPDX-License-Identifier: Apache-2.0
"""The memory orchestrator process: one authority per shared region.

Run as ``lmcache memory`` or ``python -m lmcache.v1.memory_orchestrator``.
Region state lives in memory only; the startup marker
``<state-dir>/<region_id>.marker`` keeps a second orchestrator off the region
and records whether the previous one stopped cleanly.
"""

# Standard
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
import argparse
import enum
import json
import math
import os
import re
import secrets
import signal
import socket
import sys
import threading

# Third Party
import grpc

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_orchestrator._proto_gen.memory_orchestrator_pb2_grpc import (
    add_MemoryOrchestratorServicer_to_server,
)
from lmcache.v1.memory_orchestrator.api import (
    DEFAULT_LAYOUT_FINGERPRINT,
    MAX_UINT64,
    VISIBILITY_MODES,
)
from lmcache.v1.memory_orchestrator.service import OrchestratorServicer
from lmcache.v1.memory_orchestrator.state import RegionState

logger = init_logger(__name__)

_EXIT_REFUSED = 2
_STOP_GRACE_S = 1.0
# Without this, gRPC binds with SO_REUSEPORT and a second orchestrator on the same
# port would silently split the clients between two authorities instead of failing.
_GRPC_OPTIONS = (("grpc.so_reuseport", 0),)
_SIZE_RE = re.compile(r"(\d+)([KMG]?)", re.IGNORECASE)
_SIZE_UNITS = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}


def _positive_int(text: str) -> int:
    """Parse a positive integer flag value."""
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def _positive_float(text: str) -> float:
    """Parse a positive, finite number flag value."""
    try:
        value = float(text)
    except ValueError:
        value = 0.0
    if not (value > 0 and math.isfinite(value)):
        raise argparse.ArgumentTypeError(f"expected a positive number, got {text!r}")
    return value


def _alignment_bytes(text: str) -> int:
    """Parse a power-of-two byte count with an optional K/M/G (1024-based) suffix."""
    match = _SIZE_RE.fullmatch(text)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"expected bytes with an optional K/M/G suffix, got {text!r}"
        )
    value = int(match.group(1)) * _SIZE_UNITS[match.group(2).upper()]
    if value <= 0 or value & (value - 1):
        raise argparse.ArgumentTypeError(
            f"alignment must be a power of two, got {text!r}"
        )
    return value


def _listen_address(text: str) -> str:
    """Check a ``HOST:PORT`` flag value; port 0 picks a free port."""
    host, _, port = text.rpartition(":")
    if not host or not port.isdigit() or int(port) > 65535:
        raise argparse.ArgumentTypeError(f"expected HOST:PORT, got {text!r}")
    return text


def _region_id(text: str) -> str:
    """Check a region id; it names the startup marker file."""
    if not text or "/" in text or "\0" in text or text in (".", ".."):
        raise argparse.ArgumentTypeError(
            f"region id must be usable as a file name, got {text!r}"
        )
    return text


def _pid_alive(pid: int) -> bool:
    """Return whether a process with this pid exists on this host."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists but belongs to another user
    return True


@dataclass(frozen=True)
class OrchestratorConfig:
    """Settings of one memory orchestrator process: the ``RegionState``
    arguments of the same names plus the ``--listen``, ``--state-dir`` and
    ``--max-workers`` flags.
    """

    region_id: str
    capacity_bytes: int
    alignment_bytes: int
    layout_fingerprint: bytes
    visibility_mode: str
    max_batch_entries: int
    listen: str
    state_dir: Path
    max_workers: int


@dataclass(frozen=True)
class _StartupMarker:
    """Contents of ``<state-dir>/<region_id>.marker``, stored as JSON."""

    region_id: str
    hostname: str
    pid: int
    region_epoch: int
    endpoint: str  # HOST:PORT the orchestrator is bound to
    started_at: str  # start time, ISO 8601 in UTC
    reset_required: bool


class _StartMode(enum.Enum):
    """How the startup marker lets this orchestrator start."""

    FRESH = enum.auto()  # no marker: serve normally and create the marker
    RESET_REQUIRED = enum.auto()  # marker of a dead orchestrator: refuse clients
    REFUSED = enum.auto()  # another orchestrator may own the region: do not start


def _read_marker(path: Path) -> _StartupMarker | None:
    """Return the startup marker at ``path``, or None if there is none.

    Raises:
        ValueError: If the marker exists but cannot be parsed.
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    try:
        return _StartupMarker(**json.loads(text))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"unreadable startup marker {path}: {exc}") from exc


def _start_mode(marker_path: Path, hostname: str, region_id: str) -> _StartMode:
    """Decide from the startup marker whether and how to start; logs why."""
    try:
        previous = _read_marker(marker_path)
    except ValueError:
        logger.exception(
            "Refusing to start: another orchestrator may be writing %s. If no "
            "orchestrator serves region %s, delete the marker.",
            marker_path,
            region_id,
        )
        return _StartMode.REFUSED
    if previous is None:
        return _StartMode.FRESH
    if previous.hostname != hostname:
        logger.error(
            "Refusing to start: startup marker %s names an orchestrator on host "
            "%s (pid %d) whose liveness cannot be checked from %s. If it is "
            "stopped and no MP server uses region %s, delete the marker.",
            marker_path,
            previous.hostname,
            previous.pid,
            hostname,
            region_id,
        )
        return _StartMode.REFUSED
    # A marker naming this very pid is stale: a restarted container can hand
    # the new orchestrator the pid of the old one.
    if previous.pid != os.getpid() and _pid_alive(previous.pid):
        logger.error(
            "Refusing to start: orchestrator pid %d still serves region %s "
            "(startup marker %s)",
            previous.pid,
            region_id,
            marker_path,
        )
        return _StartMode.REFUSED
    logger.error(
        "Region %s: the orchestrator recorded in %s (pid %d, started %s) did "
        "not stop cleanly, and MP servers of its epoch may still be using the "
        "region. Starting in RESET_REQUIRED: every client call is refused. To "
        "reset, stop every MP server using the region, stop this orchestrator, "
        "delete %s, and start the orchestrator again.",
        region_id,
        marker_path,
        previous.pid,
        previous.started_at,
        marker_path,
    )
    return _StartMode.RESET_REQUIRED


def _write_marker(path: Path, marker: _StartupMarker, mode: _StartMode) -> None:
    """Write and fsync the startup marker of a FRESH or RESET_REQUIRED start.

    FRESH creates it with ``O_EXCL``, so only one racing orchestrator can;
    RESET_REQUIRED atomically replaces the dead orchestrator's marker.

    Raises:
        FileExistsError: If a FRESH start finds the marker already created.
    """
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    if mode is _StartMode.FRESH:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    else:
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
    with os.fdopen(fd, "w") as marker_file:
        json.dump(asdict(marker), marker_file)
        marker_file.flush()
        os.fsync(marker_file.fileno())
    if mode is not _StartMode.FRESH:
        os.replace(tmp_path, path)


def _release_marker(path: Path, marker: _StartupMarker, state: RegionState) -> None:
    """Remove the startup marker after a clean stop, or log why it stays."""
    if state.contract().reset_required:
        reason = f"region {marker.region_id} still needs an offline reset"
    elif clients := state.registered_clients():
        reason = (
            f"{clients} clients are still registered and may still use region "
            f"{marker.region_id}, so the next start will require a reset"
        )
    else:
        try:
            still_ours = _read_marker(path) == marker
        except ValueError:
            still_ours = False
        if still_ours:
            path.unlink()
            logger.info("Removed startup marker %s after a clean stop", path)
            return
        reason = "it no longer names this orchestrator"
    logger.warning("Keeping startup marker %s: %s", path, reason)


def add_orchestrator_args(parser: argparse.ArgumentParser) -> None:
    """Add the orchestrator flags, shared by ``lmcache memory`` and
    ``python -m lmcache.v1.memory_orchestrator``, to ``parser``.

    Args:
        parser: The argument parser to add the flags to.
    """
    group = parser.add_argument_group(
        "Memory orchestrator",
        "Allocation and readiness authority for one shared memory region",
    )
    group.add_argument(
        "--region-id",
        type=_region_id,
        required=True,
        help="Name of the shared region. Clients must name the same region.",
    )
    capacity = group.add_mutually_exclusive_group(required=True)
    capacity.add_argument(
        "--capacity-gb",
        type=_positive_float,
        help="Region capacity in GiB, rounded down to the alignment.",
    )
    capacity.add_argument(
        "--capacity-bytes",
        type=_positive_int,
        help="Region capacity in bytes, rounded down to the alignment.",
    )
    group.add_argument(
        "--alignment",
        type=_alignment_bytes,
        default="2M",
        help="Extent alignment: bytes or a K/M/G suffix, a power of two. "
        "Default is %(default)s.",
    )
    group.add_argument(
        "--layout-fingerprint",
        default=DEFAULT_LAYOUT_FINGERPRINT.decode(),
        help="Layout fingerprint every client must present (sent as UTF-8). "
        "Default is %(default)s.",
    )
    group.add_argument(
        "--visibility-mode",
        choices=VISIBILITY_MODES,
        default="software_fenced",
        help="How writes become visible to other hosts; every client must use "
        "the same mode. Default is %(default)s.",
    )
    group.add_argument(
        "--max-batch-entries",
        type=_positive_int,
        default=4096,
        help="Most entries or tokens one request may carry. Default is %(default)s.",
    )
    group.add_argument(
        "--listen",
        type=_listen_address,
        default="0.0.0.0:7700",
        help="HOST:PORT to serve gRPC on; port 0 picks a free port, recorded "
        "in the startup marker. Default is %(default)s.",
    )
    group.add_argument(
        "--state-dir",
        type=Path,
        required=True,
        help="Directory for the startup marker <region-id>.marker; created if missing.",
    )
    group.add_argument(
        "--max-workers",
        type=_positive_int,
        default=16,
        help="Threads serving gRPC requests. Default is %(default)s.",
    )


def parse_args_to_orchestrator_config(args: argparse.Namespace) -> OrchestratorConfig:
    """Build the config from the flags added by ``add_orchestrator_args``.

    Args:
        args: Parsed arguments.

    Returns:
        The orchestrator configuration; its capacity is rounded down to a
        multiple of the alignment.

    Raises:
        ValueError: If the capacity is smaller than the alignment.
    """
    requested_bytes = args.capacity_bytes
    if requested_bytes is None:
        requested_bytes = int(args.capacity_gb * (1 << 30))
    capacity_bytes = requested_bytes // args.alignment * args.alignment
    if capacity_bytes == 0:
        raise ValueError(
            f"capacity of {requested_bytes} bytes is smaller than the alignment "
            f"of {args.alignment} bytes"
        )
    return OrchestratorConfig(
        region_id=args.region_id,
        capacity_bytes=capacity_bytes,
        alignment_bytes=args.alignment,
        layout_fingerprint=args.layout_fingerprint.encode(),
        visibility_mode=args.visibility_mode,
        max_batch_entries=args.max_batch_entries,
        listen=args.listen,
        state_dir=args.state_dir,
        max_workers=args.max_workers,
    )


def parse_args(argv: Sequence[str] | None) -> OrchestratorConfig:
    """Parse orchestrator command-line arguments.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` when None.

    Returns:
        The orchestrator configuration.

    Raises:
        SystemExit: With code 2 if the arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        prog="python -m lmcache.v1.memory_orchestrator",
        description="Run the LMCache memory orchestrator for one shared region.",
    )
    add_orchestrator_args(parser)
    args = parser.parse_args(argv)
    try:
        return parse_args_to_orchestrator_config(args)
    except ValueError as exc:
        parser.error(str(exc))


def serve(config: OrchestratorConfig) -> int:
    """Serve the region until SIGTERM or SIGINT.

    The startup marker is written after binding, so it records the real port,
    and is removed on stop only if no client is registered and no reset is
    pending. Must be called from the main thread: it installs signal handlers.

    Args:
        config: Orchestrator settings.

    Returns:
        0 after a signal-driven stop; 2 if another orchestrator may own the region.

    Raises:
        RuntimeError: If ``config.listen`` cannot be bound.
        OSError: If the state directory or the marker cannot be written.
    """
    config.state_dir.mkdir(parents=True, exist_ok=True)
    marker_path = config.state_dir / f"{config.region_id}.marker"
    hostname = socket.gethostname()
    mode = _start_mode(marker_path, hostname, config.region_id)
    if mode is _StartMode.REFUSED:
        return _EXIT_REFUSED

    state = RegionState(
        config.region_id,
        config.capacity_bytes,
        config.alignment_bytes,
        config.layout_fingerprint,
        config.visibility_mode,
        config.max_batch_entries,
        region_epoch=secrets.randbelow(MAX_UINT64) + 1,
        reset_required=mode is _StartMode.RESET_REQUIRED,
    )
    executor = ThreadPoolExecutor(
        max_workers=config.max_workers, thread_name_prefix="memory-orchestrator"
    )
    server = grpc.server(executor, options=_GRPC_OPTIONS)
    add_MemoryOrchestratorServicer_to_server(OrchestratorServicer(state), server)
    port = server.add_insecure_port(config.listen)
    contract = state.contract()
    marker = _StartupMarker(
        region_id=config.region_id,
        hostname=hostname,
        pid=os.getpid(),
        region_epoch=contract.region_epoch,
        endpoint=f"{config.listen.rpartition(':')[0]}:{port}",
        started_at=datetime.now(timezone.utc).isoformat(),
        reset_required=contract.reset_required,
    )

    stop = threading.Event()

    def request_stop(signum: int, frame: FrameType | None) -> None:
        logger.info("Received signal %d; stopping", signum)
        stop.set()

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    # The bound socket queues clients that read the marker before start().
    try:
        _write_marker(marker_path, marker, mode)
    except FileExistsError:
        logger.exception(
            "Refusing to serve region %s: another orchestrator created the "
            "startup marker %s while this one was starting",
            config.region_id,
            marker_path,
        )
        return _EXIT_REFUSED
    server.start()
    logger.info(
        "Memory orchestrator serving region %s on %s: region_epoch=%d "
        "capacity_bytes=%d alignment_bytes=%d layout_fingerprint=%s "
        "visibility_mode=%s max_batch_entries=%d reset_required=%s",
        contract.region_id,
        marker.endpoint,
        contract.region_epoch,
        contract.capacity_bytes,
        contract.alignment_bytes,
        contract.layout_fingerprint,
        contract.visibility_mode,
        contract.max_batch_entries,
        contract.reset_required,
    )
    stop.wait()
    server.stop(grace=_STOP_GRACE_S).wait()
    executor.shutdown(wait=False)
    _release_marker(marker_path, marker, state)
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    """Run the orchestrator from command-line arguments and exit with its code.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` when None.
    """
    sys.exit(serve(parse_args(argv)))
