# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the XPU nightly Buildkite verification gate."""

# Standard
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import importlib.util

# Third Party
import pytest


@pytest.fixture
def verifier() -> ModuleType:
    """Load the standalone verifier without running its entry point."""
    path = (
        Path(__file__).resolve().parents[2]
        / ".buildkite/k3_tests/xpu/verify-xpu-ci-image.py"
    )
    spec = importlib.util.spec_from_file_location("verify_xpu_ci_image", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def output_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Configure the verifier with valid inputs and an isolated output file."""
    output = tmp_path / "github-output"
    for name, value in {
        "BUILDKITE_API_TOKEN": "test-token",
        "CANDIDATE_IMAGE": "lmcache/vllm-openai-xpu-ci@sha256:" + "a" * 64,
        "SOURCE_COMMIT": "b" * 40,
        "SOURCE_BRANCH": "dev",
        "WHEEL_ARTIFACT_ID": "11071540516",
        "GITHUB_OUTPUT": str(output),
    }.items():
        monkeypatch.setenv(name, value)
    return output


@pytest.mark.parametrize("prefix", ["", "sha256:"])
def test_artifact_digest_forwarded_to_both_pipelines(
    verifier: ModuleType,
    output_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
) -> None:
    """Both supported input formats reach both pipelines as prefixed SHA-256."""
    digest = "fcaaa5ed43e827d7d09c15202119f61e9bd34d1ef1f66da945376937d29260a8"
    monkeypatch.setenv("WHEEL_ARTIFACT_DIGEST", prefix + digest)
    with patch.object(
        verifier,
        "buildkite_request",
        side_effect=[
            {"number": 1, "web_url": "https://buildkite.com/lmcache/unit-tests-xpu/1"},
            {"number": 2, "web_url": "https://buildkite.com/lmcache/xpu-mp-test/2"},
            {"state": "passed"},
            {"state": "passed"},
        ],
    ) as request:
        verifier.main()

    assert request.call_count == 4
    for call in request.call_args_list[:2]:
        assert call.args[2]["env"]["XPU_WHEEL_ARTIFACT_DIGEST"] == "sha256:" + digest
        assert call.args[2]["env"]["XPU_WHEEL_ARTIFACT_ID"] == "11071540516"
        assert call.args[2]["env"]["XPU_SOURCE_COMMIT"] == "b" * 40
    assert output_path.read_text() == (
        "unit_build_url=https://buildkite.com/lmcache/unit-tests-xpu/1\n"
        "unit_build_number=1\n"
    )


@pytest.mark.parametrize(
    "digest",
    [
        "",
        "a" * 63,
        "a" * 65,
        "g" * 64,
        "sha512:" + "a" * 64,
        "sha256:sha256:" + "a" * 64,
    ],
)
def test_invalid_digest_rejected_before_buildkite(
    verifier: ModuleType,
    output_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    digest: str,
) -> None:
    """Malformed digests cannot trigger builds or produce promotion outputs."""
    monkeypatch.setenv("WHEEL_ARTIFACT_DIGEST", digest)
    with patch.object(verifier, "buildkite_request") as request:
        with pytest.raises(RuntimeError, match="XPU wheel artifact are required"):
            verifier.main()
    request.assert_not_called()
    assert not output_path.exists()
