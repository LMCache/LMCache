# SPDX-License-Identifier: Apache-2.0
"""Exercise restart cleanup with real listeners owned by another process."""

# Standard
from pathlib import Path
import subprocess
import sys
import time
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "stop-processes.py"
LISTENER = """
import signal
import socket
import sys
import time

sock = socket.socket()
sock.bind(('127.0.0.1', 0))
sock.listen()
if sys.argv[1] == 'slow':
    def stop(*args):
        time.sleep(1)
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
else:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
print(sock.getsockname()[1], flush=True)
while True:
    time.sleep(1)
"""


class StopProcessesTest(unittest.TestCase):
    """Verify graceful waiting, escalation and rejection of unrelated owners."""

    def setUp(self) -> None:
        """Start a listener that is a sibling of the cleanup subprocess."""
        self.process = subprocess.Popen(
            [sys.executable, "-c", LISTENER, self.mode],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert self.process.stdout is not None
        self.port = int(self.process.stdout.readline())

    def tearDown(self) -> None:
        """Release the listener even if cleanup or an assertion fails."""
        if self.process.poll() is None:
            self.process.kill()
        self.process.communicate(timeout=5)

    mode = "slow"

    def test_waits_for_non_child_to_release_port(self) -> None:
        """A graceful shutdown must finish before cleanup reports success."""
        started = time.monotonic()
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(self.process.pid),
                "--ports",
                str(self.port),
                "--timeout",
                "5",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreaterEqual(time.monotonic() - started, 1)
        self.assertNotIn("SIGKILL", result.stdout)
        self.assertIsNotNone(self.process.poll())

    def test_does_not_kill_untracked_port_owner(self) -> None:
        """An occupied port without a tracked PID fails instead of killing it."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--ports", str(self.port)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(self.process.poll())


class StubbornProcessTest(StopProcessesTest):
    """Check bounded escalation for a process that ignores SIGTERM."""

    mode = "stubborn"

    def test_waits_for_non_child_to_release_port(self) -> None:
        """Force shutdown after the deadline and verify the listener is gone."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(self.process.pid),
                "--ports",
                str(self.port),
                "--timeout",
                "0.2",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SIGKILL", result.stdout)
        self.assertIsNotNone(self.process.poll())


if __name__ == "__main__":
    unittest.main()
