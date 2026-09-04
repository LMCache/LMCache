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
        "'sh-harbor.mthreads.com/ai-kv/kuae-lmcache-vllm-ci@sha256:"
        "75c8c1012cf49caf6dd99dbbfd33931ef100d035647083b999eaf0092d94edba' }}"
    )
    call = workflow["on"]["workflow_call"]
    assert call["outputs"]["musa_version"]["value"] == (
        "${{ jobs.build-musa-artifacts.outputs.musa_version }}"
    )
    job = workflow["jobs"]["build-musa-artifacts"]
    assert job["outputs"]["musa_version"] == (
        "${{ steps.musa-version.outputs.musa_version }}"
    )
    upload_steps = [
        step
        for step in job["steps"]
        if "uses" in step and "upload-artifact" in step["uses"]
    ]
    assert upload_steps[0]["with"]["name"] == "release-musa-artifacts"


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
    filter_text = jobs["changes"]["steps"][1]["with"]["filters"]
    assert ".github/workflows/build_musa_artifacts.yml" in filter_text
    assert ".github/scripts/build_musa_wheel.sh" in filter_text
