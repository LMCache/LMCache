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
    m = cfg["models"]["qwen38"]
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
        with pytest.raises(RuntimeError, match="Qwen3.8-Flash-Next.*d.log"):
            hf.fetch(m, cache, "prt-serving:x", cfg["container"], log)
    else:
        hf.fetch(m, cache, "prt-serving:x", cfg["container"], log)
    (args,) = calls
    assert args[-3:] == [m["repo"], m["revision"], cache]
    assert f"{cache}:{cache}" in args and f"HF_HOME={cache}" in args
    assert removed == [args[args.index("--name") + 1]]
    assert Path(m["done"]).exists() == (rc == 0)


def _repo_dir(cache: Path, repo: str) -> Path:
    return cache / "hub" / ("models--" + repo.replace("/", "--"))


def _seed(cache: Path, repo: str) -> Path:
    (_repo_dir(cache, repo) / "snapshots").mkdir(parents=True)
    return _repo_dir(cache, repo)


def test_prune_removes_only_models_no_longer_configured() -> None:
    cfg = load()
    cache, root = cfg["host"]["hf_cache"], cfg["host"]["root"]
    models = list(cfg["models"].values())
    for m in models:
        _seed(Path(cache), m["repo"])
        Path(m["done"]).parent.mkdir(parents=True, exist_ok=True)
        Path(m["done"]).touch()
    gone = _seed(Path(cache), "XiaomiMiMo/MiMo-V2.6-Pro-MOPD")
    stale = Path(root, "state", "model-6a7f202e.done")
    stale.touch()
    assert hf.prune(cache, root, models) == ["models--XiaomiMiMo--MiMo-V2.6-Pro-MOPD"]
    assert not gone.exists() and not stale.exists()
    for m in models:
        assert _repo_dir(Path(cache), m["repo"]).is_dir()
        assert Path(m["done"]).exists()


@pytest.mark.parametrize("linked", [False, True])
def test_prune_leaves_a_cache_outside_prt_root_alone(
    tmp_path: Path, linked: bool
) -> None:
    cfg = load()
    shared = tmp_path.parent / f"shared-hf-{tmp_path.name}"
    _seed(shared, "someone/else")
    root = cfg["host"]["root"]
    cache = Path(root, "hf-link") if linked else shared
    if linked:
        cache.symlink_to(shared)
    assert hf.prune(str(cache), root, []) == []
    assert (shared / "hub" / "models--someone--else").is_dir()
