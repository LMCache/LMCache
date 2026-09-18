# SPDX-License-Identifier: Apache-2.0
"""Static contract tests for the MUSA wheel release workflow."""

# Standard
from pathlib import Path
import stat
import subprocess
import sys
import zipfile

# Third Party
import yaml

ROOT = Path(__file__).resolve().parents[4]


def _extract_musa_wheel_metadata_verifier() -> str:
    script = (ROOT / ".github/scripts/build_musa_wheel.sh").read_text()
    start = script.index("<<'PY'\n") + len("<<'PY'\n")
    end = script.index('\nPY\n\necho "=== bundled MUSA/torch libraries', start)
    return script[start:end]


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

    install = next(
        step
        for step in workflow["runs"]["steps"]
        if step.get("name") == "Install Python build dependencies"
    )
    assert "PIP_INDEX_URL" not in install["env"]
    assert install["env"]["CUSTOM_PIP_INDEX_URL"] == "${{ inputs.pip_index_url }}"
    assert '--index-url "${CUSTOM_PIP_INDEX_URL}"' in install["run"]


def test_musa_reusable_workflow_exposes_version_and_artifact_contract() -> None:
    """The reusable job output must match the artifact consumed by publish."""
    workflow = _load_workflow(".github/workflows/build_musa_artifacts.yml")
    call = workflow["on"]["workflow_call"]
    assert call["inputs"]["dev_version"]["type"] == "boolean"
    assert call["inputs"]["dev_version"]["default"] == "false"
    assert (
        call["outputs"]["musa_version"]["value"]
        == "${{ jobs.build-musa-artifacts.outputs.musa_version }}"
    )
    job = workflow["jobs"]["build-musa-artifacts"]
    assert job["outputs"]["musa_version"] == "${{ steps.prepare.outputs.version }}"

    musa_build = next(
        step
        for step in job["steps"]
        if step.get("name") == "Build MUSA wheel in the public MUSA image"
    )
    assert workflow["env"]["SKIP_AUDITWHEEL_REPAIR"] == "0"
    assert workflow["env"]["MAX_JOBS"] == "2"
    assert "torch_musa" not in musa_build["run"]
    assert (
        '-e TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD}"'
        in musa_build["run"]
    )

    version_step = next(
        step for step in job["steps"] if step.get("name") == "Prepare artifact build"
    )
    assert version_step["uses"] == "./.github/actions/build-artifacts"
    assert version_step["with"]["dev_version"] == "${{ inputs.dev_version }}"
    assert version_step["with"]["version_suffix"] == "+${{ env.MUSA_LOCAL_VERSION }}"

    upload = next(
        step for step in job["steps"] if "upload-artifact" in step.get("uses", "")
    )
    assert upload["with"]["name"] == "release-musa-artifacts"


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
    assert ".github/actions/build-artifacts/action.yml" in filter_text
    assert ".github/scripts/build_musa_wheel.sh" in filter_text

    nightly = _load_workflow(".github/workflows/nightly_build.yml")
    nightly_build = nightly["jobs"]["nightly-musa-wheel"]
    assert nightly_build["uses"] == "./.github/workflows/build_musa_artifacts.yml"
    assert nightly_build["with"]["dev_version"] == "true"
    nightly_publish = nightly["jobs"]["publish-nightly-musa"]
    assert nightly_publish["needs"] == "nightly-musa-wheel"
    publish_step = next(
        step
        for step in nightly_publish["steps"]
        if step.get("name") == "Publish nightly artifacts"
    )
    assert publish_step["uses"] == "./.github/actions/publish-artifacts"
    assert publish_step["with"]["artifact_name"] == "release-musa-artifacts"
    assert publish_step["with"]["release_tag"] == "nightly-musa"
    assert (
        "needs.nightly-musa-wheel.outputs.musa_version"
        in publish_step["with"]["release_notes"]
    )
    publish_action = _load_workflow(".github/actions/publish-artifacts/action.yml")
    publish_command = next(
        step["run"]
        for step in publish_action["runs"]["steps"]
        if step.get("name") == "Publish rolling prerelease"
    )
    assert "--prerelease" in publish_command


def test_musa_wheel_metadata_verifier_accepts_pep440_normalized_version(
    tmp_path: Path,
) -> None:
    """Wheel metadata may canonicalize release segments such as 1.0001 -> 1.1."""
    wheel = tmp_path / "lmcache-1.1+musa-cp310-cp310-manylinux_2_35_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "lmcache-1.1+musa.dist-info/METADATA",
            "\n".join(
                [
                    "Metadata-Version: 2.4",
                    "Name: lmcache",
                    "Version: 1.1+musa",
                    "",
                ]
            ),
        )
        archive.writestr(
            "lmcache-1.1+musa.dist-info/WHEEL",
            "\n".join(
                [
                    "Wheel-Version: 1.0",
                    "Generator: test",
                    "Root-Is-Purelib: false",
                    "Tag: cp310-cp310-manylinux_2_35_x86_64",
                    "",
                ]
            ),
        )

    subprocess.run(
        [
            sys.executable,
            "-c",
            _extract_musa_wheel_metadata_verifier(),
            str(wheel),
            str(tmp_path / "check"),
            "1.0001+musa",
            "manylinux_2_35_x86_64",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
