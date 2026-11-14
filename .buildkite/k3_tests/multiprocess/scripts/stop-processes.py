# SPDX-License-Identifier: Apache-2.0
"""Stop tracked test processes and verify serving ports before a restart."""

# Standard
import argparse
import os
import signal
import socket
import subprocess
import time


def process_is_running(pid: int) -> bool:
    """Return whether a PID exists and has not become a zombie."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return bool(status) and not status.startswith("Z")


def signal_processes(pids: list[int], sig: signal.Signals) -> None:
    """Signal the tracked PIDs, allowing processes that have already exited."""
    for pid in pids:
        print(f"Sending {sig.name} to PID {pid}", flush=True)
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def wait_for_exit(pids: list[int], timeout: float) -> list[int]:
    """Return PIDs still running after a bounded wait, including non-children."""
    deadline = time.monotonic() + timeout
    remaining = pids
    while remaining:
        remaining = [pid for pid in remaining if process_is_running(pid)]
        if not remaining or time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    return remaining


def verify_ports_released(ports: list[int]) -> None:
    """Fail if a localhost TCP port cannot be rebound after shutdown."""
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
        print(f"Port {port} released", flush=True)


def main() -> None:
    """Stop test-owned PIDs gracefully, escalate on timeout, then check ports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pids", type=int, nargs="*")
    parser.add_argument("--ports", type=int, nargs="*", default=[])
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if any(pid <= 1 for pid in args.pids) or args.timeout < 0:
        parser.error("PIDs must be greater than 1 and timeout must be nonnegative")
    pids = [pid for pid in args.pids if process_is_running(pid)]
    signal_processes(pids, signal.SIGTERM)
    remaining = wait_for_exit(pids, args.timeout)
    if remaining:
        signal_processes(remaining, signal.SIGKILL)
        remaining = wait_for_exit(remaining, 10)
    if remaining:
        raise RuntimeError(f"Processes did not exit after SIGKILL: {remaining}")
    verify_ports_released(args.ports)


if __name__ == "__main__":
    main()
