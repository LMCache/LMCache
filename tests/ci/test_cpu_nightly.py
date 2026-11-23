# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the CPU nightly producer/consumer contract."""

# Standard
from datetime import datetime, timedelta, timezone
from pathlib import Path
import importlib.metadata as md
import importlib.util
import json
import os
import subprocess
import sys

# Third Party
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/cpu_nightly.py"
SPEC = importlib.util.spec_from_file_location("cpu_nightly", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
nightly = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = nightly
SPEC.loader.exec_module(nightly)
NOW = datetime(2026, 11, 19, 9, tzinfo=timezone.utc)


def distribution(
    tmp_path: Path, version: str, manifest: dict[str, object] | None
) -> md.Distribution:
    """Create installed distribution metadata without importing vLLM."""
    info = tmp_path / f"vllm_cpu_nightly-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Name: vllm-cpu-nightly\nVersion: {version}\n")
    if manifest is not None:
        (info / "cpu-nightly.json").write_text(json.dumps(manifest))
    return md.Distribution.at(info)


def test_legacy_wheel_cannot_be_revalidated_as_fresh(tmp_path: Path) -> None:
    """A successful test today cannot refresh a September build's age."""
    dist = distribution(tmp_path, "0.29.1.dev202609171111", None)
    build = nightly.read_build(dist)
    with pytest.raises(ValueError, match="stale"):
        nightly.check_freshness(build, NOW)


@pytest.mark.parametrize("age,passes", [(48, True), (48.01, False), (-0.26, False)])
def test_manifest_age_uses_build_time(tmp_path: Path, age: float, passes: bool) -> None:
    """The 48-hour boundary and future timestamps are enforced."""
    built_at = (NOW - timedelta(hours=age)).isoformat()
    build = nightly.read_build(
        distribution(
            tmp_path,
            "1.0.dev202611190900",
            {"source_sha": "a" * 40, "built_at": built_at, "direct_requirements": []},
        )
    )
    assert build.source_sha == "a" * 40
    if passes:
        nightly.check_freshness(build, NOW)
    else:
        with pytest.raises(ValueError, match="stale or future-dated"):
            nightly.check_freshness(build, NOW)


def test_url_dependencies_preserve_platform_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the URL matching this runtime is handed to pip as an argument."""
    log = tmp_path / "pip-args.json"
    fake_pip = tmp_path / "pip.py"
    fake_pip.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        f"Path({str(log)!r}).write_text(json.dumps(sys.argv[1:]))\n"
    )
    monkeypatch.setenv("PIP_BIN", f"{sys.executable} {fake_pip}")
    wanted = 'triton @ https://example.org/cpu.whl ; python_version >= "3.0"'
    build = nightly.NightlyBuild(
        "a" * 40,
        NOW,
        [wanted, 'triton @ https://example.org/other.whl ; python_version < "3.0"'],
    )
    nightly.install_dependencies(build)
    assert json.loads(log.read_text()) == ["install", wanted]


@pytest.mark.parametrize("status", ["tested", "failed"])
def test_cpu_pin_retains_build_provenance(status: str) -> None:
    """The pin record distinguishes the build time from validation time."""
    result = subprocess.run(
        ["bash", str(SCRIPT.with_name("pin-tested-vllm.sh"))],
        env={
            **os.environ,
            "CI_PLATFORM": "github_actions",
            "PIN_BACKEND": "cpu",
            "PIN_VLLM_DRY_RUN": "1",
            "PIN_VLLM_STATUS": status,
            "OS_PLATFORM": "ubuntu-22.04",
            "VLLM_VERSION": "1.0.dev202611190900",
            "VLLM_SOURCE_SHA": "a" * 40,
            "VLLM_BUILT_AT": NOW.isoformat(),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.splitlines()
    marker = next(i for i, line in enumerate(lines) if "would append to" in line)
    record = json.loads(lines[marker + 1])
    assert record["vllm_full_sha"] == "a" * 40
    assert record["built_at"] == NOW.isoformat()
    assert record["archive_index_url"] == ""
    assert ("would write latest_tested" in result.stdout) == (status == "tested")
