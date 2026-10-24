# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from typing import Any
import json

# Third Party
import pytest

# First Party
from perf_regression_testing import suite


def _fetch_models(cfg: dict[str, Any]) -> None:
    for m in cfg["models"].values():
        Path(m["done"]).parent.mkdir(parents=True, exist_ok=True)
        Path(m["done"]).touch()


def test_low_docker_disk_stops_the_suite_before_any_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fetch_models(suite.load())
    monkeypatch.setattr(suite.docker, "root_free_gb", lambda: 0.2)

    def no_build(*args: object, **kwargs: object) -> dict:
        raise AssertionError("built with a full disk")

    monkeypatch.setattr(suite.images, "tonight", no_build)
    assert suite.suite("weekly", tmp_path / "s") == suite.EXIT_SUITE_INVALID
    rep = json.loads((tmp_path / "s" / "report.json").read_text())
    assert "0.2 GB free" in rep["images"]["error"]
    assert "NOT RUN" in (tmp_path / "s" / "report.md").read_text()


def test_missing_models_are_downloaded_and_a_failed_one_skips_its_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(suite.docker, "root_free_gb", lambda: 1e9)
    monkeypatch.setattr(suite.docker, "image_label", lambda image, key: "x")

    def tonight(root: Path, src: Path, out: Path, need_lmcache: bool) -> dict:
        im = {"usable": True, "vllm_source": "nightly", "time": 0.0}
        im |= {"serving": "prt-serving:x", "vllm_commit": "v"}
        return im | {"lmcache": "prt-lmcache:x", "lmcache_commit": "l"}

    fetched: list[str] = []

    def fetch(model: dict, hf_cache: str, image: str, container: dict, log: Path):
        fetched.append(model["repo"])
        if model["repo"].startswith("zai-org/"):
            raise RuntimeError("disk full")
        Path(model["done"]).parent.mkdir(parents=True, exist_ok=True)
        Path(model["done"]).touch()

    ran: list[str] = []

    def run(spec: Any, out: Path, image: str, aiperf: str) -> int:
        ran.append(spec.deployment)
        out.mkdir(parents=True)
        summary = {"invalid": [], "vllm_commit": "v", "lmcache_commit": "l"}
        (out / "summary.json").write_text(json.dumps(summary))
        return suite.EXIT_VALID

    def prune(hf_cache: str, root: str, models: list[dict]) -> list[str]:
        raise PermissionError("root-owned snapshot")

    monkeypatch.setattr(suite.images, "tonight", tonight)
    monkeypatch.setattr(suite.images, "build", lambda args, log: f"prt-{args[0]}")
    monkeypatch.setattr(suite.images, "prune", lambda keep: [])
    monkeypatch.setattr(suite.hf, "fetch", fetch)
    monkeypatch.setattr(suite.hf, "prune", prune)
    monkeypatch.setattr(suite, "run", run)
    deps = ["vllm_simple", "vllm_lmcache_ssm"]
    assert suite.suite("weekly", tmp_path / "s", deps) == suite.EXIT_SUITE_INVALID
    assert fetched == [
        "arcee-ai/Trinity-Large-Thinking-FP8-Block",
        "zai-org/GLM-5.3-Flash",
    ]
    assert ran == ["vllm_simple"]
    rep = json.loads((tmp_path / "s" / "report.json").read_text())
    assert rep["prune_error"] == "root-owned snapshot"
    skipped = [r for r in rep["runs"] if r["deployment"] == "vllm_lmcache_ssm"]
    assert "disk full" in skipped[0]["summary"]["invalid"][0]
    root = Path(suite.load()["host"]["root"])
    history = (root / "prt-results" / "history.jsonl").read_text().splitlines()
    assert [json.loads(h)["suite"] for h in history] == ["weekly"]


@pytest.mark.parametrize(
    "name, deployments, pinned",
    [
        ("nightly", None, True),
        ("weekly", None, True),
        ("weekly", ["vllm_simple"], False),
    ],
)
def test_only_a_suite_that_ran_lmcache_pins_a_new_vllm(
    name: str,
    deployments: list[str] | None,
    pinned: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = suite.load()
    _fetch_models(cfg)
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
    assert suite.suite(name, tmp_path / "s", deployments) == suite.EXIT_VALID
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


def test_deployments_runs_only_the_named_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fetch_models(suite.load())
    monkeypatch.setattr(suite.docker, "root_free_gb", lambda: 1e9)
    monkeypatch.setattr(suite.docker, "image_label", lambda image, key: "x")
    builds: list[str] = []

    def tonight(root: Path, src: Path, out: Path, need_lmcache: bool) -> dict:
        im = {"usable": True, "vllm_source": "nightly", "time": 0.0}
        return im | {"serving": "prt-serving:x", "vllm_commit": "v"}

    def build(args: list[str], log: Path) -> str:
        builds.append(args[0])
        return f"prt-{args[0]}"

    ran: list[str] = []

    def run(spec: object, out: Path, image: str, aiperf: str) -> int:
        ran.append(out.name.rsplit("-", 1)[-1])
        out.mkdir(parents=True)
        summary = {"invalid": [], "vllm_commit": "v", "lmcache_commit": "l"}
        (out / "summary.json").write_text(json.dumps(summary))
        return suite.EXIT_VALID

    monkeypatch.setattr(suite.images, "tonight", tonight)
    monkeypatch.setattr(suite.images, "build", build)
    monkeypatch.setattr(suite.images, "prune", lambda keep: [])
    monkeypatch.setattr(suite, "run", run)
    assert suite.suite("weekly", tmp_path / "s", ["vllm_simple"]) == suite.EXIT_VALID
    assert ran == ["vllm_simple"]
    assert "mooncake" not in builds


def test_a_deployment_outside_the_suite_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["suite", "--suite", "weekly", "--deployments", "vllm_simple,vllm_lmcache"]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit) as e:
        suite.main()
    assert e.value.code == suite.EXIT_SUITE_INVALID
    err = capsys.readouterr().err
    assert "vllm_lmcache" in err and "vllm_native" in err


def test_a_suite_first_removes_containers_a_dead_suite_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    removed: list[list[str]] = []

    def remove(names: list[str], stop_s: int) -> list[str]:
        removed.append(names)
        return []

    monkeypatch.setattr(suite.docker, "prt_containers", lambda: ["prt-old-lmcache"])
    monkeypatch.setattr(suite.docker, "remove", remove)
    monkeypatch.setattr(suite, "suite", lambda name, out, deployments: 0)
    monkeypatch.setattr("sys.argv", ["suite", "--suite", "nightly"])
    with pytest.raises(SystemExit) as e:
        suite.main()
    assert e.value.code == 0
    assert removed == [["prt-old-lmcache"]]
