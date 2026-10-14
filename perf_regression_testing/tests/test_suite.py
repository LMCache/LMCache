# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import json

# Third Party
import pytest

# First Party
from perf_regression_testing import suite


def test_low_docker_disk_stops_the_suite_before_any_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Path(suite.load()["model"]["path"])
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    monkeypatch.setattr(suite.docker, "root_free_gb", lambda: 0.2)

    def no_build(cfg: object) -> Path:
        raise AssertionError("built with a full disk")

    monkeypatch.setattr(suite, "update_lmcache", no_build)
    assert suite.suite("weekly", tmp_path / "s") == suite.EXIT_SUITE_INVALID
    rep = json.loads((tmp_path / "s" / "report.json").read_text())
    assert "0.2 GB free" in rep["images"]["disk_full"]
    assert "NO IMAGES" in (tmp_path / "s" / "report.md").read_text()


def test_missing_model_stops_the_suite_before_any_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_build(cfg: object) -> Path:
        raise AssertionError("built without the model")

    monkeypatch.setattr(suite, "update_lmcache", no_build)
    assert suite.suite("nightly", tmp_path / "s") == suite.EXIT_SUITE_INVALID
    rep = json.loads((tmp_path / "s" / "report.json").read_text())
    assert "hf download" in rep["images"]["host_error"]
    assert "NOT RUN" in (tmp_path / "s" / "report.md").read_text()


def test_a_held_lock_times_out_and_a_released_one_is_taken(tmp_path: Path) -> None:
    held = suite.lock(tmp_path / "l", 0)
    assert held is not None
    assert suite.lock(tmp_path / "l", 0) is None
    held.close()
    assert suite.lock(tmp_path / "l", 0) is not None
