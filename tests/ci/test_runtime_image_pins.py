# SPDX-License-Identifier: Apache-2.0
"""Tests for image runtime pins: pin-tested-vllm.sh writes them and
resolve-pinned-vllm.sh reads them back for PR builds.

Run without loading the torch-dependent tests/conftest.py:
    uv run --no-sync --python 3.12 python -m pytest -q \
        --confcutdir=tests/ci tests/ci/test_runtime_image_pins.py
"""

# Standard
from pathlib import Path
import json
import os
import subprocess

# Third Party
import pytest

ROOT = Path(__file__).resolve().parents[2]
PIN_SCRIPT = ROOT / ".github/scripts/pin-tested-vllm.sh"
RESOLVER = ROOT / ".buildkite/k3_harness/resolve-pinned-vllm.sh"
RBLN_TAG = "harbor.example/rbln-serve:0.12.1-dev37"
RBLN_DIGEST = "sha256:" + "a" * 64


def _clean_env(**overrides: str) -> dict[str, str]:
    """Return an environment free of CI pin overrides plus ``overrides``."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("PIN_", "PINNED_", "VLLM_", "BUILDKITE_", "GITHUB_"))
    }
    env.update(overrides)
    return env


def _pin_record(backend: str, **env: str) -> dict:
    """Dry-run the pin script and return the record it would append."""
    result = subprocess.run(
        ["bash", str(PIN_SCRIPT)],
        env=_clean_env(PIN_BACKEND=backend, PIN_VLLM_DRY_RUN="1", **env),
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.splitlines()
    marker = next(i for i, line in enumerate(lines) if "would append to" in line)
    return json.loads(lines[marker + 1])


def _resolve_images(history: Path) -> dict[str, str]:
    """Source the resolver against ``history`` and return the image pins."""
    script = (
        f'source "{RESOLVER}" 2>/dev/null; '
        'printf "%s\\n%s\\n" "${PINNED_XPU_IMAGE}" "${PINNED_RBLN_IMAGE}"'
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env=_clean_env(
            USE_PINNED_VLLM="false",
            LMCACHE_RUNTIME_HISTORY_URL=history.as_uri(),
            XPU_RELEASE_IMAGE="xpu-fallback",
            RBLN_FALLBACK_IMAGE="rbln-fallback",
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    xpu, rbln = result.stdout.splitlines()
    return {"xpu": xpu, "rbln": rbln}


def _image_record(backend: str, runtime_id: str, tag: str, **fields: str) -> dict:
    """Build a tested_runtimes.jsonl image record as the pin script writes it."""
    record = {
        "backend": backend,
        "runtime_id": runtime_id,
        "installation_form": "image",
        "status": "tested",
        "validator": "buildkite",
        "image_tag": tag,
        "image_ref": f"{tag}@{RBLN_DIGEST}",
        "image_digest": RBLN_DIGEST,
    }
    record.update(fields)
    return record


def _write_history(path: Path, records: list[dict]) -> Path:
    """Write ``records`` to ``path`` as a tested_runtimes.jsonl history."""
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    return path


def test_rbln_pin_round_trips_to_pr_builds(tmp_path: Path) -> None:
    """A tested RBLN pin is exactly the image PR builds resolve next."""
    record = _pin_record(
        "rbln",
        VLLM_IMAGE_TAG=RBLN_TAG,
        VLLM_IMAGE_REF=f"{RBLN_TAG}@{RBLN_DIGEST}",
        VLLM_IMAGE_DIGEST=RBLN_DIGEST,
    )
    assert record["installation_form"] == "image"
    assert record["runtime_id"] == "linux-rebellions-rbln"

    history = _write_history(tmp_path / "history.jsonl", [record])
    assert _resolve_images(history)["rbln"] == f"{RBLN_TAG}@{RBLN_DIGEST}"


def test_rbln_pin_requires_a_digest() -> None:
    """A tested image pin without an immutable digest is refused."""
    result = subprocess.run(
        ["bash", str(PIN_SCRIPT)],
        env=_clean_env(
            PIN_BACKEND="rbln", PIN_VLLM_DRY_RUN="1", VLLM_IMAGE_TAG=RBLN_TAG
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "immutable digest" in result.stderr


@pytest.mark.parametrize(
    ("backend", "runtime_id"),
    [("rbln", "linux-rebellions-rbln"), ("xpu", "linux-intel-xpu")],
)
def test_resolver_picks_last_tested_image_of_its_runtime(
    tmp_path: Path, backend: str, runtime_id: str
) -> None:
    """Failed, unpinned and other-runtime records never become the pin."""
    other = "xpu" if backend == "rbln" else "rbln"
    other_runtime = "linux-intel-xpu" if backend == "rbln" else "linux-rebellions-rbln"
    history = _write_history(
        tmp_path / "history.jsonl",
        [
            _image_record(backend, runtime_id, "reg/img:old"),
            _image_record(backend, runtime_id, "reg/img:new"),
            _image_record(backend, runtime_id, "reg/img:failed", status="failed"),
            _image_record(
                backend, runtime_id, "reg/img:no-digest", image_ref="reg/img:no-digest"
            ),
            _image_record(other, other_runtime, "reg/img:other"),
        ],
    )

    assert _resolve_images(history)[backend] == f"reg/img:new@{RBLN_DIGEST}"


def test_resolver_falls_back_without_a_pin(tmp_path: Path) -> None:
    """Before any image is pinned, each runtime uses its fallback image."""
    history = _write_history(tmp_path / "history.jsonl", [])

    assert _resolve_images(history) == {"xpu": "xpu-fallback", "rbln": "rbln-fallback"}
