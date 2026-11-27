# SPDX-License-Identifier: Apache-2.0
"""Wait until a server answers its health endpoint.

The rbln-serve image has no curl, so the smoke polls with the standard
library. Exits non-zero when the server process exits first or the endpoint
is not healthy within the timeout.

Usage: python wait_for_health.py URL PID [--timeout SECONDS]
"""

# Standard
import argparse
import os
import time
import urllib.error
import urllib.request


def _is_running(pid: int) -> bool:
    """Return whether process ``pid`` still exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _is_healthy(url: str) -> bool:
    """Return whether ``url`` answers with a 2xx status."""
    try:
        with urllib.request.urlopen(url, timeout=5):
            return True
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return False


def main() -> None:
    """Poll the health endpoint once per second until it answers.

    Raises:
        SystemExit: If the server process exits before it is healthy or the
            timeout elapses.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("pid", type=int)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        if not _is_running(args.pid):
            raise SystemExit("server exited before becoming healthy")
        if _is_healthy(args.url):
            print(f"{args.url} is healthy")
            return
        time.sleep(1)
    raise SystemExit(f"{args.url} not healthy within {args.timeout:.0f}s")


if __name__ == "__main__":
    main()
