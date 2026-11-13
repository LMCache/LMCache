# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import json
import subprocess
import time

# Third Party
import pytest

# First Party
from perf_regression_testing.runner import images


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fake docker builds; returns the targets built, in order."""
    built: list[str] = []

    def build(args: list[str], log: Path) -> str:
        built.append(args[0])
        return "prt-serving:new" if args[0] == "serving" else f"prt-lmcache:{args[2]}"

    monkeypatch.setattr(images, "build", build)
    monkeypatch.setattr(images.docker, "image_label", lambda image, key: image)
    monkeypatch.setattr(images.docker, "image_exists", lambda image: True)
    monkeypatch.setattr(
        images.docker, "save", lambda image, path: path.write_text(image)
    )
    monkeypatch.setattr(images.docker, "load", lambda path: built.append("load"))
    return built


def _pin(root: Path, age_s: float) -> None:
    (root / "prt-images").mkdir()
    (root / "prt-images" / "pinned.json").write_text(
        json.dumps({"serving": "prt-serving:old", "time": time.time() - age_s})
    )


def _pinned(root: Path) -> str:
    return json.loads((root / "prt-images" / "pinned.json").read_text())["serving"]


def test_young_pin_is_reused_without_a_serving_build(
    builds: list[str], tmp_path: Path
) -> None:
    _pin(tmp_path, 3 * 86400)
    res = images.tonight(tmp_path, tmp_path, tmp_path / "out", need_lmcache=True)
    assert res["vllm_source"] == "pinned"
    assert res["usable"] and res["serving"] == "prt-serving:old"
    assert res["lmcache"] == "prt-lmcache:prt-serving:old"
    assert builds == ["lmcache"]


def test_suite_without_lmcache_builds_nothing_on_a_young_pin(
    builds: list[str], tmp_path: Path
) -> None:
    _pin(tmp_path, 3 * 86400)
    res = images.tonight(tmp_path, tmp_path, tmp_path / "out", need_lmcache=False)
    assert res["usable"] and res["serving"] == "prt-serving:old"
    assert "lmcache" not in res
    assert builds == []


def test_removed_pinned_image_is_loaded_from_its_saved_tar(
    builds: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(tmp_path, 8 * 86400)
    (tmp_path / "prt-images" / "prt-serving_old.tar").write_text("old")
    images.pin(
        tmp_path,
        images.tonight(tmp_path, tmp_path, tmp_path / "out", need_lmcache=True),
    )
    # Only the new pin's tar is kept.
    assert [t.name for t in (tmp_path / "prt-images").glob("*.tar")] == [
        "prt-serving_new.tar"
    ]
    builds.clear()
    monkeypatch.setattr(images.docker, "image_exists", lambda image: False)
    res = images.tonight(tmp_path, tmp_path, tmp_path / "out2", need_lmcache=True)
    assert res["vllm_source"] == "pinned" and res["serving"] == "prt-serving:new"
    assert builds == ["load", "lmcache"]


def test_a_tar_that_fails_to_load_builds_the_nightly(
    builds: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(tmp_path, 3 * 86400)
    (tmp_path / "prt-images" / "prt-serving_old.tar").write_text("corrupt")
    monkeypatch.setattr(images.docker, "image_exists", lambda image: False)

    def load(path: Path) -> None:
        raise subprocess.CalledProcessError(1, "docker load")

    monkeypatch.setattr(images.docker, "load", load)
    res = images.tonight(tmp_path, tmp_path, tmp_path / "out", need_lmcache=True)
    assert res["vllm_source"] == "nightly" and res["serving"] == "prt-serving:new"
    assert builds == ["serving", "lmcache"]


def test_removed_pinned_image_without_a_tar_builds_the_nightly(
    builds: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(tmp_path, 3 * 86400)
    monkeypatch.setattr(images.docker, "image_exists", lambda image: False)
    res = images.tonight(tmp_path, tmp_path, tmp_path / "out", need_lmcache=True)
    assert res["vllm_source"] == "nightly"
    assert res["usable"] and res["serving"] == "prt-serving:new"
    assert builds == ["serving", "lmcache"]


@pytest.mark.parametrize(
    "error", [subprocess.CalledProcessError(1, "docker save"), OSError(28, "full")]
)
def test_a_failed_save_still_pins_and_leaves_no_partial_tar(
    error: Exception, builds: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def save(image: str, path: Path) -> None:
        path.write_text("half")
        raise error

    monkeypatch.setattr(images.docker, "save", save)
    # A partial save left by an earlier, cancelled pin.
    (tmp_path / "prt-images").mkdir()
    (tmp_path / "prt-images" / "prt-serving_old.part").write_text("half")
    images.pin(tmp_path, {"vllm_source": "nightly", "serving": "s:1", "time": 0.0})
    assert _pinned(tmp_path) == "s:1"
    assert sorted(p.name for p in (tmp_path / "prt-images").iterdir()) == [
        "pinned.json"
    ]


def test_week_old_pin_builds_the_nightly_and_pins_it_only_when_told(
    builds: list[str], tmp_path: Path
) -> None:
    _pin(tmp_path, 8 * 86400)
    res = images.tonight(tmp_path, tmp_path, tmp_path / "out", need_lmcache=True)
    assert res["vllm_source"] == "nightly"
    assert res["serving"] == "prt-serving:new"
    assert builds == ["serving", "lmcache"]
    assert _pinned(tmp_path) == "prt-serving:old"
    images.pin(tmp_path, res)
    assert _pinned(tmp_path) == "prt-serving:new"


def test_reusing_the_pin_does_not_restart_its_week(
    builds: list[str], tmp_path: Path
) -> None:
    _pin(tmp_path, 3 * 86400)
    before = (tmp_path / "prt-images" / "pinned.json").read_text()
    images.pin(
        tmp_path,
        images.tonight(tmp_path, tmp_path, tmp_path / "out", need_lmcache=True),
    )
    assert (tmp_path / "prt-images" / "pinned.json").read_text() == before


def test_a_failed_build_leaves_no_images_and_no_fallback(
    builds: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(tmp_path, 8 * 86400)

    def build(args: list[str], log: Path) -> str:
        raise RuntimeError("no nightly wheel")

    monkeypatch.setattr(images, "build", build)
    res = images.tonight(tmp_path, tmp_path, tmp_path / "out", need_lmcache=True)
    assert not res["usable"] and "serving" not in res
    assert res["error"] == "no nightly wheel"


def test_no_pin_yet_builds_the_nightly(builds: list[str], tmp_path: Path) -> None:
    res = images.tonight(tmp_path, tmp_path, tmp_path / "out", need_lmcache=False)
    assert res["usable"] and res["vllm_source"] == "nightly"
    assert builds == ["serving"]
