# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import json
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
