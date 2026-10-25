# SPDX-License-Identifier: Apache-2.0
# Standard
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

# Third Party
import pytest

# First Party
from perf_regression_testing.runner import run
from perf_regression_testing.runner.config import load, resolve


def test_busy_gpus_make_the_run_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "pick_gpus", lambda count, wait_s: None)
    spec = resolve(load(), "agentic", "vllm_native")
    code = run.run(spec, tmp_path / "r", "img", "aiperf")
    summary = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert code == run.EXIT_GPU_BUSY
    assert "busy" in summary["invalid"][0]
    # The results page draws one line per model and load.
    assert summary["concurrency"] == spec.aiperf_args["concurrency"] == 340


def _fake_smi(
    monkeypatch: pytest.MonkeyPatch, busy: list[str]
) -> list[subprocess.CompletedProcess[str]]:
    """nvidia-smi with 8 GPUs (uuid GPU-<i>) and compute processes on ``busy``."""
    calls: list[subprocess.CompletedProcess[str]] = []

    def fake(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
        if "--query-compute-apps=gpu_uuid" in cmd:
            out = "".join(f"{u}\n" for u in busy)
        else:
            out = "".join(f"{i}, GPU-{i}\n" for i in range(8))
        r = subprocess.CompletedProcess(cmd, 0, out, "")
        calls.append(r)
        return r

    monkeypatch.setattr(run.subprocess, "run", fake)
    monkeypatch.setattr(run.time, "sleep", lambda s: None)
    return calls


def test_pick_gpus_skips_gpus_with_a_compute_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_smi(monkeypatch, ["GPU-0", "GPU-5", "GPU-0"])
    assert run.pick_gpus(4, wait_s=60) == [1, 2, 3, 4]


def test_pick_gpus_gives_up_when_too_few_are_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_smi(monkeypatch, [f"GPU-{i}" for i in range(5)])
    assert run.pick_gpus(4, wait_s=0) is None


def test_only_mooncake_clients_run_without_seccomp(tmp_path: Path) -> None:
    spec = resolve(load(), "agentic", "vllm_mooncake")
    cmds = run.build_commands(spec, [0, 1, 2, 3], "t", tmp_path, "img", "aiperf")
    for role, cmd in cmds.items():
        opts = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--security-opt"]
        if role.startswith("mooncake_client"):
            assert opts == ["seccomp=unconfined"]
        else:
            assert opts == []


@pytest.mark.parametrize(
    ("deployment", "dropped"),
    [("vllm_native", ["DAC_OVERRIDE"]), ("vllm_lmcache", [])],
)
def test_a_capped_vllm_cannot_override_dir_permissions(
    tmp_path: Path, deployment: str, dropped: list[str]
) -> None:
    spec = resolve(load(), "agentic", deployment)
    cmd = run.build_commands(spec, [0, 1, 2, 3], "t", tmp_path, "img", "aiperf")["vllm"]
    assert [cmd[i + 1] for i, a in enumerate(cmd) if a == "--cap-drop"] == dropped


def test_only_regions_this_run_created_are_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shm = tmp_path / "shm"
    shm.mkdir()
    ours, other = shm / "vllm_offload_ab-12.mmap", shm / "vllm_offload_cd-34.mmap"
    ours.touch()
    other.touch()
    log = (
        f"[shared_offload_region.py:142] Created mmap file /dev/shm/{ours.name}\n"
        "[shared_offload_region.py:128] Opened existing mmap file "
        f"/dev/shm/{other.name}\n"
    )
    (tmp_path / "vllm.log").write_text(log)
    monkeypatch.setattr(run, "SHM_DIR", shm)
    assert run.remove_shm_regions(tmp_path) == [ours.name]
    assert not ours.exists() and other.exists()


def test_no_log_removes_nothing(tmp_path: Path) -> None:
    assert run.remove_shm_regions(tmp_path) == []


def test_heartbeat_reports_the_latest_aiperf_phase(tmp_path: Path) -> None:
    now = datetime(2026, 10, 15, 21, 55, tzinfo=timezone.utc)
    log = tmp_path / "aiperf.log"
    log.write_text("21:47:41.852 INFO     Starting AIPerf System (_single_run.py:81)\n")
    assert run.aiperf_phase(log, now) == "loading the trace"
    with log.open("a") as f:
        f.write(
            "21:53:22.097 NOTICE   Phase agentic.warmup (warmup) started"
            " | target: 207 requests (runner.py:660)\n"
        )
    assert run.aiperf_phase(log, now) == "agentic.warmup started | target: 207 requests"
    with log.open("a") as f:
        f.write(
            "Warmup: 3/207 |   1% [00:33<38:31]21:54:22.111 INFO     Phase warmup"
            " progress | returned=4/207 | errors=0 (runner.py:1318)\n"
        )
    assert run.aiperf_phase(log, now) == "warmup progress | returned=4/207 | errors=0"


def test_heartbeat_counts_a_fixed_duration_phase_across_midnight(
    tmp_path: Path,
) -> None:
    log = tmp_path / "aiperf.log"
    log.write_text(
        "23:58:00.000 NOTICE   Phase profiling (profiling) started"
        " | target: 960.0s duration (runner.py:660)\n"
    )
    now = datetime(2026, 10, 16, 0, 4, tzinfo=timezone.utc)
    assert run.aiperf_phase(log, now) == (
        "profiling started | target: 960.0s duration (6 of 16 min)"
    )


@pytest.mark.parametrize(
    ("deployment", "limit"), [("vllm_simple_l2", ["1532g"]), ("vllm_lmcache", [])]
)
def test_only_a_deployment_with_a_memory_cap_gets_one(
    tmp_path: Path, deployment: str, limit: list[str]
) -> None:
    spec = resolve(load(), "agentic", deployment)
    cmd = run.build_commands(spec, [0, 1, 2, 3], "t", tmp_path, "img", "aiperf")["vllm"]
    assert [cmd[i + 1] for i, a in enumerate(cmd) if a == "--memory"] == limit
    # Swap counts toward the cap, so the page cache cannot spill past it.
    assert [cmd[i + 1] for i, a in enumerate(cmd) if a == "--memory-swap"] == limit


def test_io_bytes_sums_reads_and_writes_over_containers_and_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, stat in [
        ("a", "8:0 rbytes=10 wbytes=20 rios=1 wios=2\n259:0 rbytes=1 wbytes=2\n"),
        ("b", "259:1 rbytes=100 wbytes=200 dbytes=7\n"),
    ]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "io.stat").write_text(stat)
    monkeypatch.setattr(run.docker, "cgroup_dir", lambda name: tmp_path / name)
    assert run.io_bytes(["a", "b"]) == {"rbytes": 111, "wbytes": 222}


def test_gpu_holders_names_the_container_of_each_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "nvidia-smi":
            out = "GPU-0, 11, python3, 9000 MiB\nGPU-1, 22, python3, 10 MiB\n"
        else:
            out = "abc123 prt-old-lmcache\ndef456 other\n"
        return subprocess.CompletedProcess(cmd, 0, out, "")

    monkeypatch.setattr(run.subprocess, "run", fake)
    monkeypatch.setattr(run, "PROC", tmp_path)
    (tmp_path / "11").mkdir()
    (tmp_path / "11" / "cgroup").write_text("0::/system.slice/docker-abc123.scope\n")
    assert run.gpu_holders() == [
        "GPU-0, 11, python3, 9000 MiB, container=prt-old-lmcache",
        "GPU-1, 22, python3, 10 MiB",
    ]


def test_a_dead_vllm_is_named_as_exited(monkeypatch: pytest.MonkeyPatch) -> None:
    states = {"prt-x-lmcache": "running", "prt-x-vllm": "exited"}
    monkeypatch.setattr(run.docker, "status", states.get)
    assert run.exited(["prt-x-lmcache", "prt-x-vllm"]) == ["prt-x-vllm"]
