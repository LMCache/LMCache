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
    assert "import torch_musa" not in content
    assert "SETUPTOOLS_SCM_PRETEND_VERSION:-0.0.0.dev0+musa" in content
    assert "--exclude 'libmusa*.so*'" in content
    assert "wheel is missing the +musa local version" in content

    workflow = _load_workflow(".github/actions/build-artifacts/action.yml")
    cleanup = next(
        step
        for step in workflow["runs"]["steps"]
        if step.get("name") == "Remove non-release tags"
    )
    assert "grep -vE" in cleanup["run"]
    assert "|| true" in cleanup["run"]


def test_musa_reusable_workflow_exposes_version_and_artifact_contract() -> None:
    """The reusable job output must match the artifact consumed by publish."""
    workflow = _load_workflow(".github/workflows/build_artifacts.yml")
    action = _load_workflow(".github/actions/build-artifacts/action.yml")
    call = workflow["on"]["workflow_call"]
    assert call["inputs"]["dev_version"]["type"] == "boolean"
    assert call["inputs"]["dev_version"]["default"] == "false"
    assert (
        call["outputs"]["version"]["value"]
        == "${{ jobs.build-artifacts.outputs.version }}"
    )
    job = workflow["jobs"]["build-artifacts"]
    assert job["outputs"]["version"] == "${{ steps.build.outputs.version }}"
    assert "musa" in action["inputs"]["target"]["description"]

    musa_build = next(
        step
        for step in action["runs"]["steps"]
        if step.get("name") == "Build MUSA artifact"
    )
    assert musa_build["if"] == "inputs.target == 'musa'"
    assert musa_build["env"]["MUSA_IMAGE"] == (
        "${{ env.MUSA_IMAGE || "
        "'registry.mthreads.com/mcconline/musa-pytorch-release-public:"
        "rc5.1.0-v2.9.1-S5000-py310_tef' }}"
    )
    assert musa_build["env"]["SKIP_AUDITWHEEL_REPAIR"] == "0"
    assert musa_build["env"]["MAX_JOBS"] == "2"
    assert "torch_musa" not in musa_build["run"]
    assert "-e TORCH_DEVICE_BACKEND_AUTOLOAD=0" in musa_build["run"]
    assert "-e LMCACHE_LOG_LEVEL=ERROR" in musa_build["run"]
    assert "--no-deps" in musa_build["run"]

    version_step = next(
        step
        for step in action["runs"]["steps"]
        if step.get("name") == "Resolve platform wheel version"
    )
    assert version_step["env"]["DEV_VERSION"] == "${{ inputs.dev_version }}"
    assert 'musa) version="${base}+musa"' in version_step["run"]

    upload = next(
        step
        for step in action["runs"]["steps"]
        if "upload-artifact" in step.get("uses", "")
    )
    assert "release-{0}-artifacts" in upload["with"]["name"]


def test_publish_workflow_wires_musa_build_and_release() -> None:
    """Changes, build, and release jobs must all reference MUSA artifacts."""
    workflow = _load_workflow(".github/workflows/publish.yml")
    jobs = workflow["jobs"]
    build = jobs["build-artifacts"]
    assert build["uses"] == "./.github/workflows/build_artifacts.yml"
    assert "musa" in build["with"]["targets"]
    assert jobs["publish-musa-github-release"]["needs"] == [
        "changes",
        "build-artifacts",
        "test",
        "code-quality",
    ]
    assert build["secrets"] == "inherit"
    filter_text = jobs["changes"]["steps"][1]["with"]["filters"]
    assert ".github/workflows/build_artifacts.yml" in filter_text
    assert ".github/actions/build-artifacts/action.yml" in filter_text
    assert ".github/scripts/build_musa_wheel.sh" in filter_text

    nightly = _load_workflow(".github/workflows/nightly_build.yml")
    nightly_build = nightly["jobs"]["nightly-musa-wheel"]
    assert nightly_build["uses"] == "./.github/workflows/build_artifacts.yml"
    assert nightly_build["with"]["targets"] == '["musa"]'
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
