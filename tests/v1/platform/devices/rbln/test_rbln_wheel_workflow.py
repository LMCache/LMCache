# SPDX-License-Identifier: Apache-2.0
"""Static contract tests for the RBLN wheel build and release wiring."""

# Standard
from pathlib import Path
import os
import re

# Third Party
import yaml

ROOT = Path(__file__).resolve().parents[5]
BUILD_WORKFLOW = ".github/workflows/build_rbln_artifacts.yml"
BUILD_SCRIPT = ".github/scripts/build_rbln_wheel.sh"
ARTIFACT = "release-rbln-artifacts"


def _load_workflow(relative_path: str) -> dict:
    """Load a GitHub Actions workflow without YAML 1.1 key coercion."""
    return yaml.load((ROOT / relative_path).read_text(), Loader=yaml.BaseLoader)


def _step(steps: list[dict], name: str) -> dict:
    return next(step for step in steps if step.get("name") == name)


def _referenced_outputs(text: str, job: str) -> set[str]:
    """Return the ``needs.<job>.outputs.*`` names that ``text`` uses."""
    return set(re.findall(rf"needs\.{re.escape(job)}\.outputs\.(\w+)", text))


def _declared_outputs() -> set[str]:
    """Return the outputs the reusable RBLN build workflow declares."""
    workflow = _load_workflow(BUILD_WORKFLOW)
    return set(workflow["on"]["workflow_call"]["outputs"])


def test_build_workflow_uploads_the_wheel_its_script_builds() -> None:
    """The reusable workflow runs the build script and uploads its +rbln wheel."""
    assert os.access(ROOT / BUILD_SCRIPT, os.X_OK)
    job = _load_workflow(BUILD_WORKFLOW)["jobs"]["build-rbln-artifacts"]

    build = _step(job["steps"], "Build RBLN wheel in the manylinux image")
    assert f"/work/LMCache/{BUILD_SCRIPT}" in build["run"]
    prepare = _step(job["steps"], "Prepare artifact build")
    assert prepare["with"]["version_suffix"] == "+rbln"
    assert prepare["with"]["dev_version"] == "${{ inputs.dev_version }}"
    upload = next(s for s in job["steps"] if "upload-artifact" in s.get("uses", ""))
    assert upload["with"]["name"] == ARTIFACT


def test_release_publishes_the_rbln_wheel_to_its_own_github_release() -> None:
    """A published release uploads the RBLN artifact to a v{tag}-rbln release."""
    jobs = _load_workflow(".github/workflows/publish.yml")["jobs"]
    assert jobs["build-rbln"]["uses"] == f"./{BUILD_WORKFLOW}"

    release = jobs["publish-rbln-github-release"]
    assert "build-rbln" in release["needs"]
    fetch = _step(release["steps"], "Fetch RBLN release artifacts")
    assert fetch["with"]["name"] == ARTIFACT
    publish = _step(release["steps"], "Publish RBLN wheel to dedicated GitHub Release")
    assert 'gh release create "${{ github.ref_name }}-rbln"' in publish["run"]
    used = _referenced_outputs(publish["run"], "build-rbln")
    assert used and used <= _declared_outputs()

    path_filter = next(s for s in jobs["changes"]["steps"] if s.get("id") == "filter")
    assert BUILD_WORKFLOW in path_filter["with"]["filters"]
    assert BUILD_SCRIPT in path_filter["with"]["filters"]


def test_nightly_publishes_a_dev_versioned_rbln_wheel() -> None:
    """The nightly build publishes a .devN RBLN wheel to the nightly-rbln release."""
    jobs = _load_workflow(".github/workflows/nightly_build.yml")["jobs"]
    build = jobs["nightly-rbln-wheel"]
    assert build["uses"] == f"./{BUILD_WORKFLOW}"
    assert build["with"]["dev_version"] == "true"

    publish = _step(jobs["publish-nightly-rbln"]["steps"], "Publish nightly artifacts")
    assert publish["with"]["artifact_name"] == ARTIFACT
    assert publish["with"]["release_tag"] == "nightly-rbln"
    used = _referenced_outputs(publish["with"]["release_notes"], "nightly-rbln-wheel")
    assert used and used <= _declared_outputs()
