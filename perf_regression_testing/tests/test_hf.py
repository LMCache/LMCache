# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from typing import Any
import subprocess

# Third Party
import pytest

# First Party
from perf_regression_testing.runner import hf
from perf_regression_testing.runner.config import load


@pytest.mark.parametrize("rc", [0, 1])
def test_fetch_marks_only_a_finished_download_and_always_removes_its_container(
    rc: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load()
    m = cfg["models"]["glm53"]
    cache = cfg["host"]["hf_cache"]
    calls: list[list[str]] = []
    removed: list[str] = []

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        calls.append(args)
        kwargs["stdout"].write("fetching\n")
        return subprocess.CompletedProcess(args, rc)

    monkeypatch.setattr(hf.subprocess, "run", fake_run)

    def remove(names: list[str], stop_s: int) -> list[str]:
        removed.extend(names)
        return []

    monkeypatch.setattr(hf.docker, "remove", remove)
    log = tmp_path / "d.log"
    if rc:
        with pytest.raises(RuntimeError, match="GLM-5.3.*d.log"):
            hf.fetch(m, cache, "prt-serving:x", cfg["container"], log)
    else:
        hf.fetch(m, cache, "prt-serving:x", cfg["container"], log)
    (args,) = calls
    assert args[-3:] == [m["repo"], m["revision"], cache]
    assert f"{cache}:{cache}" in args and f"HF_HOME={cache}" in args
    assert removed == [args[args.index("--name") + 1]]
    assert Path(m["done"]).exists() == (rc == 0)
