# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from types import SimpleNamespace
from typing import cast
import time

# Third Party
import pytest

# First Party
from perf_regression_testing.runner import disk_cap
from perf_regression_testing.runner.config import RunSpec
from perf_regression_testing.runner.disk_cap import DiskCap
from perf_regression_testing.runner.run import wipe_scratch

MiB = 2**20


def _fill(d: Path, n: int) -> None:
    for i in range(n):
        (d / f"b{i}.bin").write_bytes(b"x" * MiB)


def test_past_the_cap_new_files_fail_and_old_ones_still_read(tmp_path: Path) -> None:
    sub = tmp_path / "fs_r0" / "c74"
    sub.mkdir(parents=True)
    _fill(sub, 4)
    cap = DiskCap(tmp_path, 4 * MiB, poll_s=0.05)
    cap.start()
    deadline = time.time() + 10
    while cap.frozen_at is None and time.time() < deadline:
        time.sleep(0.05)
    with pytest.raises(PermissionError):
        (sub / "new.bin.tmp").write_bytes(b"y")
    with pytest.raises(PermissionError):
        (tmp_path / "fs_r0" / "c75").mkdir()
    assert (sub / "b0.bin").read_bytes() == b"x" * MiB
    res = cap.stop()
    assert res["frozen_at"] is not None
    assert res["peak_bytes"] >= 4 * MiB


def test_under_the_cap_nothing_is_frozen(tmp_path: Path) -> None:
    _fill(tmp_path, 1)
    cap = DiskCap(tmp_path, 100 * MiB, poll_s=0.05)
    cap.start()
    time.sleep(0.3)
    (tmp_path / "new.bin").write_bytes(b"y")
    assert cap.stop()["frozen_at"] is None


def test_a_frozen_dir_is_wiped(tmp_path: Path) -> None:
    root = tmp_path / "l2"
    sub = root / "fs_r0" / "c74"
    sub.mkdir(parents=True)
    _fill(sub, 2)
    cap = DiskCap(root, MiB, poll_s=0.05)
    cap.start()
    deadline = time.time() + 10
    while cap.frozen_at is None and time.time() < deadline:
        time.sleep(0.05)
    cap.stop()
    spec = SimpleNamespace(vllm_scratch=[{"path": str(root)}], services=[])
    wipe_scratch(cast(RunSpec, spec))
    assert root.is_dir() and not any(root.iterdir())


def test_a_failed_measurement_is_reported_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(path: Path) -> int:
        raise OSError("disk gone")

    monkeypatch.setattr(disk_cap, "allocated_bytes", boom)
    cap = DiskCap(tmp_path, MiB, poll_s=0.05)
    cap.start()
    time.sleep(0.3)
    monkeypatch.undo()
    assert "disk gone" in cap.stop()["error"]
