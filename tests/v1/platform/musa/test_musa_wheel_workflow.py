# SPDX-License-Identifier: Apache-2.0
"""Static contract tests for the MUSA wheel release workflow."""

# Standard
from pathlib import Path
import stat

# Third Party
import yaml

ROOT = Path(__file__).resolve().parents[4]


def _load_workflow(relative_path: str) -> dict:
    """Load a GitHub Actions workflow without YAML 1.1 key coercion."""
    return yaml.load(
        (ROOT / relative_path).read_text(),
        Loader=yaml.BaseLoader,
    )


def test_musa_builder_script_is_executable_and_has_required_guards() -> None:
    """The container entrypoint must be runnable and fail closed by default."""
    script = ROOT / ".github/scripts/build_musa_wheel.sh"
    assert stat.S_IMODE(script.stat().st_mode) & stat.S_IXUSR
    content = script.read_text()
    assert 'MUSA_REQUIRE_TORCH_MUSA="${MUSA_REQUIRE_TORCH_MUSA:-1}"' in content
    assert "SETUPTOOLS_SCM_PRETEND_VERSION:-0.0.0.dev0+musa" in content
    assert "--exclude 'libmusa*.so*'" in content
    assert "wheel is missing the +musa local version" in content

    workflow = _load_workflow(".github/workflows/build_musa_artifacts.yml")
    cleanup = next(
        step
        for step in workflow["jobs"]["build-musa-artifacts"]["steps"]
        if step.get("name") == "Remove non-release tags"
    )
    assert "grep -vE" in cleanup["run"]
    assert "|| true" in cleanup["run"]


def test_musa_reusable_workflow_exposes_version_and_artifact_contract() -> None:
    """The reusable job output must match the artifact consumed by publish."""
    workflow = _load_workflow(".github/workflows/build_musa_artifacts.yml")
    assert workflow["env"]["MUSA_IMAGE"] == (
        "${{ vars.MUSA_IMAGE || "
        "'registry.mthreads.com/mcconline/musa-pytorch-release-public:"
        "rc5.2.0-v2.9.1.post1-S5000-py310' }}"
    )
    assert workflow["env"]["MUSA_REQUIRE_TORCH_MUSA"] == (
        "${{ vars.MUSA_REQUIRE_TORCH_MUSA || '0' }}"
    )
    assert workflow["env"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert workflow["env"]["SKIP_AUDITWHEEL_REPAIR"] == "0"
    assert workflow["env"]["MAX_JOBS"] == "2"
    call = workflow["on"]["workflow_call"]
    assert call["inputs"]["dev_version"]["type"] == "boolean"
    assert call["inputs"]["dev_version"]["default"] == "false"
    assert call["outputs"]["musa_version"]["value"] == (
        "${{ jobs.build-musa-artifacts.outputs.musa_version }}"
    )
    job = workflow["jobs"]["build-musa-artifacts"]
    assert job["outputs"]["musa_version"] == (
        "${{ steps.musa-version.outputs.musa_version }}"
    )
    assert not any(
        step.get("uses", "").startswith("docker/login-action@") for step in job["steps"]
    )
    upload_steps = [
        step
        for step in job["steps"]
        if "uses" in step and "upload-artifact" in step["uses"]
    ]
    assert upload_steps[0]["with"]["name"] == "release-musa-artifacts"
    smoke = next(
        step
        for step in job["steps"]
        if step.get("name", "").startswith("Smoke-check wheel installation")
    )
    assert "torch_musa" not in smoke["run"]
    assert "--no-deps" in smoke["run"]
    assert "MUSA_REQUIRE_TORCH_MUSA" in smoke["run"]

    version_step = next(
        step
        for step in job["steps"]
        if step.get("name") == "Resolve MUSA wheel version"
    )
    assert version_step["env"]["DEV_VERSION"] == "${{ inputs.dev_version }}"
    assert 'DEV_VERSION}" == "true"' in version_step["run"]


def test_publish_workflow_wires_musa_build_and_release() -> None:
    """Changes, build, and release jobs must all reference MUSA artifacts."""
    workflow = _load_workflow(".github/workflows/publish.yml")
    jobs = workflow["jobs"]
    build = jobs["build-musa"]
    assert build["uses"] == "./.github/workflows/build_musa_artifacts.yml"
    assert jobs["publish-musa-github-release"]["needs"] == [
        "changes",
        "build-musa",
        "test",
        "code-quality",
    ]
    assert "secrets" not in build
    filter_text = jobs["changes"]["steps"][1]["with"]["filters"]
    assert ".github/workflows/build_musa_artifacts.yml" in filter_text
    assert ".github/scripts/build_musa_wheel.sh" in filter_text

    nightly = _load_workflow(".github/workflows/nightly_build.yml")
    nightly_build = nightly["jobs"]["nightly-musa-wheel"]
    assert nightly_build["uses"] == "./.github/workflows/build_musa_artifacts.yml"
    assert nightly_build["with"]["dev_version"] == "true"
    nightly_publish = nightly["jobs"]["publish-nightly-musa"]
    assert nightly_publish["needs"] == "nightly-musa-wheel"
    download = next(
        step
        for step in nightly_publish["steps"]
        if "actions/download-artifact@" in step.get("uses", "")
    )
    assert download["with"]["name"] == "release-musa-artifacts"
    publish_step = next(
        step
        for step in nightly_publish["steps"]
        if step.get("name", "").startswith("Publish MUSA wheels")
    )
    assert "nightly-musa" in publish_step["run"]
    assert "--prerelease" in publish_step["run"]
    assert "MUSA_VERSION" in publish_step["env"]
