# SPDX-License-Identifier: Apache-2.0
# Standard
import importlib.util
import os
import time

# Third Party
import pytest

if importlib.util.find_spec("pytest_benchmark") is None:

    @pytest.fixture
    def benchmark():
        pytest.skip("pytest-benchmark is not installed")


@pytest.fixture(scope="session", autouse=True)
def disable_usage_tracking():
    """Keep the test suite from sending usage telemetry to the stats server.

    Tests that exercise the telemetry itself re-enable it per-test via
    monkeypatch with an injected transport.
    """
    os.environ["LMCACHE_TRACK_USAGE"] = "false"
    yield


@pytest.fixture(scope="session")
def bigtable_emulator():
    """Start or connect to the Bigtable emulator for integration testing."""
    # Standard
    import shutil
    import subprocess

    existing_host = os.environ.get("BIGTABLE_EMULATOR_HOST")
    if existing_host:
        print(f"\n[Fixture] Reusing existing Bigtable emulator at {existing_host}...")
        yield existing_host
        return

    if os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true":
        pytest.skip("Skipping Bigtable emulator integration tests in CI")

    if not shutil.which("gcloud"):
        pytest.skip(
            "gcloud CLI not found, skipping Bigtable Emulator integration tests"
        )

    port = "8899"
    host_port = f"localhost:{port}"

    print(f"\n[Fixture] Starting Bigtable emulator on {host_port}...")
    emulator_process = subprocess.Popen(
        [
            "gcloud",
            "beta",
            "emulators",
            "bigtable",
            "start",
            f"--host-port={host_port}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    os.environ["BIGTABLE_EMULATOR_HOST"] = host_port

    def is_port_open(h: str, p: int, t: float = 1.0) -> bool:
        # Standard
        import socket

        try:
            with socket.create_connection((h, p), timeout=t):
                return True
        except OSError:
            return False

    # Wait for the emulator to initialize
    port_num = int(port)
    start_time = time.time()
    success = False
    while time.time() - start_time < 10.0:
        if is_port_open("localhost", port_num):
            success = True
            break
        time.sleep(0.5)

    if not success:
        print(
            f"\n[Fixture] Bigtable emulator failed to bind to {host_port} within 10s!"
        )
        try:
            stdout, stderr = emulator_process.communicate(timeout=2.0)
            print(f"Stdout:\n{stdout.decode('utf-8', errors='ignore')}")
            print(f"Stderr:\n{stderr.decode('utf-8', errors='ignore')}")
        except Exception as e:
            print(f"Could not retrieve process logs: {e}")
        emulator_process.kill()
        pytest.fail("Failed to start Bigtable emulator")

    yield host_port

    print("\n[Fixture] Stopping Bigtable emulator...")
    emulator_process.terminate()
    try:
        emulator_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        emulator_process.kill()

    if "BIGTABLE_EMULATOR_HOST" in os.environ:
        del os.environ["BIGTABLE_EMULATOR_HOST"]
