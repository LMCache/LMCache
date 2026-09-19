# SPDX-License-Identifier: Apache-2.0
"""Exercise the local Dynamo demo's process lifecycle without Docker or GPUs."""

# Standard
from pathlib import Path
from typing import Any
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

FAKE_COMMAND = r"""
import json
import os
from pathlib import Path
import signal
import sys
import time


def record(event, **fields):
    payload = json.dumps({"event": event, "pid": os.getpid(), **fields}) + "\n"
    fd = os.open(
        os.environ["FAKE_EVENTS"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600
    )
    try:
        os.write(fd, payload.encode())
    finally:
        os.close(fd)


command = Path(sys.argv[0]).name
args = sys.argv[1:]
if command == "docker":
    record("docker", args=args, mode=os.environ.get("DYNAMO_MODE"))
    sys.exit(int(os.environ.get("FAKE_DOCKER_UP_STATUS", "0")) if "up" in args else 0)
if command == "curl":
    url = next(arg for arg in args if arg.startswith("http://"))
    healthy = os.environ.get("FAKE_UNHEALTHY_URL") != url
    record("health", url=url, healthy=healthy)
    sys.exit(0 if healthy else 22)
if command == "sleep":
    time.sleep(0.01)
    sys.exit(0)

kind = command if command == "lmcache" else args[args.index("-m") + 1]
if "--disaggregation-mode" in args:
    kind += ":" + args[args.index("--disaggregation-mode") + 1]


def stop(signum, frame):
    record("stop", kind=kind, signal=signum)
    sys.exit(0)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
record(
    "start",
    kind=kind,
    args=args,
    gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
    port=os.environ.get("DYN_SYSTEM_PORT"),
)
if os.environ.get("FAKE_FAIL_PROCESS") == kind:
    time.sleep(0.3)
    record("failure", kind=kind)
    sys.exit(17)
while True:
    time.sleep(0.05)
"""


class DynamoLocalLaunchTests(unittest.TestCase):
    """Verify the public launch commands and container process supervision."""

    def setUp(self) -> None:
        """Create isolated example files and fake external commands."""
        self.temporary = tempfile.TemporaryDirectory(prefix="dynamo launch tests ")
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        source = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "dynamo_integration"
            / "local"
        )
        self.local = root / "repository with spaces" / "local"
        shutil.copytree(source, self.local)
        self.cwd = root / "unrelated directory"
        self.cwd.mkdir()
        self.events = root / "events.jsonl"
        binaries = root / "bin"
        binaries.mkdir()
        for command in ("docker", "lmcache", "python3", "curl", "sleep"):
            executable = binaries / command
            executable.write_text(f"#!{sys.executable}\n{FAKE_COMMAND}")
            executable.chmod(0o755)
        self.environment = {
            **os.environ,
            "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
            "FAKE_EVENTS": str(self.events),
        }

    def test_launchers_select_mode_and_stop_compose_from_any_directory(self) -> None:
        """Host commands select their topology and find the bundled Compose file."""
        for script, mode in (
            ("agg_lmcache_mp.sh", "aggregated"),
            ("disagg_lmcache_mp.sh", "disaggregated"),
        ):
            with self.subTest(mode=mode):
                self.events.unlink(missing_ok=True)
                result = self._run(script)
                self.assertEqual(result.returncode, 0, result.stdout)
                calls = self._events()
                self.assertEqual(len(calls), 2, calls)
                start, stop = calls
                for call in calls:
                    self.assertEqual(call["event"], "docker")
                    self.assertEqual(call["mode"], mode)
                    args = call["args"]
                    self.assertEqual(args[0], "compose")
                    self.assertEqual(
                        Path(args[args.index("-f") + 1]),
                        self.local / "docker-compose.yml",
                    )
                self.assertIn("up", start["args"])
                self.assertIn("--abort-on-container-exit", start["args"])
                self.assertEqual(
                    start["args"][start["args"].index("--exit-code-from") + 1],
                    "dynamo",
                )
                self.assertEqual(
                    start["args"][start["args"].index("--profile") + 1],
                    "serving",
                )
                self.assertIn("stop", stop["args"])

    def test_failed_compose_start_stops_services_and_preserves_status(self) -> None:
        """A failed Compose launch still cleans up and reports its failure."""
        result = self._run(
            "agg_lmcache_mp.sh", extra_env={"FAKE_DOCKER_UP_STATUS": "23"}
        )
        self.assertEqual(result.returncode, 23, result.stdout)
        self.assertIn("stop", self._events()[-1]["args"])

    def test_aggregated_worker_uses_cache_and_failure_stops_peers(self) -> None:
        """One GPU worker uses the ready cache server and failures stop its peers."""
        self._check_worker_mode("aggregated", {"dynamo.vllm": ("0", "8081")})

    def test_disaggregated_workers_share_cache_on_separate_gpus(self) -> None:
        """Prefill and decode use distinct GPUs and one ready LMCache server."""
        self._check_worker_mode(
            "disaggregated",
            {
                "dynamo.vllm:decode": ("0", "8081"),
                "dynamo.vllm:prefill": ("1", "8082"),
            },
        )

    def test_unhealthy_service_prevents_inference_startup(self) -> None:
        """An unavailable dependency or cache prevents frontend and worker startup."""
        for url in (
            "http://localhost:8222/healthz",
            "http://localhost:2379/health",
            "http://localhost:8080/healthcheck",
        ):
            with self.subTest(url=url):
                self.events.unlink(missing_ok=True)
                result = self._run(
                    "serve.sh", "aggregated", extra_env={"FAKE_UNHEALTHY_URL": url}
                )
                self.assertNotEqual(result.returncode, 0, result.stdout)
                events = self._events()
                self.assertTrue(
                    any(
                        event["event"] == "health"
                        and event["url"] == url
                        and not event["healthy"]
                        for event in events
                    ),
                    events,
                )
                started = [event for event in events if event["event"] == "start"]
                self.assertFalse(
                    any(event["kind"].startswith("dynamo.") for event in started),
                    events,
                )
                self._assert_children_stopped(events)

    def test_termination_stops_all_started_processes(self) -> None:
        """Stopping the container entrypoint terminates all services it owns."""
        process = self._start("serve.sh", "disaggregated")
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if sum(event["event"] == "start" for event in self._events()) == 4:
                    break
                self.assertIsNone(process.poll(), "Entrypoint exited during startup")
                time.sleep(0.02)
            else:
                self.fail("The LMCache server, frontend, and workers did not start")
            process.terminate()
            output, _ = process.communicate(timeout=10)
            self.assertNotEqual(process.returncode, 0, output)
            self._assert_children_stopped(self._events())
        finally:
            self._kill_remaining(process)

    def _check_worker_mode(
        self, mode: str, workers: dict[str, tuple[str, str]]
    ) -> None:
        failed_worker = next(iter(workers))
        result = self._run(
            "serve.sh", mode, extra_env={"FAKE_FAIL_PROCESS": failed_worker}
        )
        self.assertEqual(result.returncode, 17, result.stdout)
        events = self._events()
        started = {
            event["kind"]: event for event in events if event["event"] == "start"
        }
        self.assertEqual(set(started), {"lmcache", "dynamo.frontend", *workers})
        cache_args = started["lmcache"]["args"]
        self.assertEqual(cache_args[0], "server")
        for option, value in (
            ("--l1-size-gb", "16"),
            ("--eviction-policy", "LRU"),
            ("--port", "5555"),
            ("--http-port", "8080"),
        ):
            self.assertEqual(cache_args[cache_args.index(option) + 1], value)
        ready_indexes = [
            index
            for index, event in enumerate(events)
            if event["event"] == "health" and event["healthy"]
        ]
        for url in (
            "http://localhost:8222/healthz",
            "http://localhost:2379/health",
            "http://localhost:8080/healthcheck",
        ):
            self.assertTrue(any(events[index]["url"] == url for index in ready_indexes))
        for worker, (gpu, port) in workers.items():
            event = started[worker]
            self.assertEqual(event["gpu"], gpu)
            self.assertEqual(event["port"], port)
            self.assertGreater(events.index(event), max(ready_indexes))
            args = event["args"]
            self.assertEqual(args[args.index("--model") + 1], "Qwen/Qwen3-0.6B")
            config = json.loads(args[args.index("--kv-transfer-config") + 1])
            self.assertEqual(config["kv_connector"], "LMCacheMPConnector")
            self.assertEqual(config["kv_role"], "kv_both")
            self.assertEqual(
                config["kv_connector_extra_config"]["lmcache.mp.port"], 5555
            )
        self._assert_children_stopped(events)

    def _assert_children_stopped(self, events: list[dict[str, Any]]) -> None:
        started = {event["pid"] for event in events if event["event"] == "start"}
        exited = {
            event["pid"] for event in events if event["event"] in ("stop", "failure")
        }
        self.assertEqual(started, exited, events)
        for pid in started:
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def _events(self) -> list[dict[str, Any]]:
        if not self.events.exists():
            return []
        return [json.loads(line) for line in self.events.read_text().splitlines()]

    def _start(
        self, script: str, *args: str, extra_env: dict[str, str] | None = None
    ) -> subprocess.Popen[str]:
        return subprocess.Popen(
            ["bash", str(self.local / script), *args],
            cwd=self.cwd,
            env={**self.environment, **(extra_env or {})},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    def _run(
        self, script: str, *args: str, extra_env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        process = self._start(script, *args, extra_env=extra_env)
        try:
            output, _ = process.communicate(timeout=20)
            return subprocess.CompletedProcess(process.args, process.returncode, output)
        finally:
            self._kill_remaining(process)

    def _kill_remaining(self, process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
