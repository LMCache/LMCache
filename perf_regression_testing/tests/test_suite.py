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

    def no_build(*args: object, **kwargs: object) -> dict:
        raise AssertionError("built with a full disk")

    monkeypatch.setattr(suite.images, "tonight", no_build)
    assert suite.suite("weekly", tmp_path / "s") == suite.EXIT_SUITE_INVALID
    rep = json.loads((tmp_path / "s" / "report.json").read_text())
    assert "0.2 GB free" in rep["images"]["error"]
    assert "NOT RUN" in (tmp_path / "s" / "report.md").read_text()


def test_missing_model_stops_the_suite_before_any_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_build(*args: object, **kwargs: object) -> dict:
        raise AssertionError("built without the model")

    monkeypatch.setattr(suite.images, "tonight", no_build)
    assert suite.suite("nightly", tmp_path / "s") == suite.EXIT_SUITE_INVALID
    rep = json.loads((tmp_path / "s" / "report.json").read_text())
    assert "hf download" in rep["images"]["error"]
    assert "NOT RUN" in (tmp_path / "s" / "report.md").read_text()


@pytest.mark.parametrize("name, pinned", [("nightly", True), ("weekly", False)])
def test_only_a_suite_that_ran_lmcache_pins_a_new_vllm(
    name: str, pinned: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = suite.load()
    model = Path(cfg["model"]["path"])
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    monkeypatch.setattr(suite.docker, "root_free_gb", lambda: 1e9)
    monkeypatch.setattr(suite.docker, "image_label", lambda image, key: "x")

    def tonight(root: Path, src: Path, out: Path, need_lmcache: bool) -> dict:
        im = {"usable": True, "vllm_source": "nightly", "time": 0.0}
        im |= {"serving": "prt-serving:new", "vllm_commit": "v"}
        lm = {"lmcache": "prt-lmcache:new", "lmcache_commit": "l"}
        return im | (lm if need_lmcache else {})

    def run(spec: object, out: Path, image: str, aiperf: str) -> int:
        out.mkdir(parents=True)
        summary = {"invalid": [], "vllm_commit": "v", "lmcache_commit": "l"}
        (out / "summary.json").write_text(json.dumps(summary))
        return suite.EXIT_VALID

    monkeypatch.setattr(suite.images, "tonight", tonight)
    monkeypatch.setattr(suite.images, "build", lambda args, log: f"prt-{args[0]}")
    monkeypatch.setattr(suite.images, "prune", lambda keep: [])
    monkeypatch.setattr(suite.docker, "save", lambda image, path: path.touch())
    monkeypatch.setattr(suite, "run", run)
    assert suite.suite(name, tmp_path / "s") == suite.EXIT_VALID
    pin = Path(cfg["host"]["root"]) / "prt-images" / "pinned.json"
    assert pin.exists() == pinned


def test_a_held_lock_times_out_and_a_released_one_is_taken(tmp_path: Path) -> None:
    held = suite.lock(tmp_path / "l", 0)
    assert held is not None
    assert suite.lock(tmp_path / "l", 0) is None
    held.close()
    assert suite.lock(tmp_path / "l", 0) is not None


def test_an_unknown_suite_is_a_usage_error_naming_the_suites(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["suite", "--suite", "daily"])
    with pytest.raises(SystemExit) as e:
        suite.main()
    assert e.value.code == suite.EXIT_SUITE_INVALID
    err = capsys.readouterr().err
    assert "'daily'" in err and "nightly" in err and "weekly" in err
