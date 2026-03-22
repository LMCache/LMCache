# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


DOCKERFILE = Path(__file__).resolve().parents[1] / "docker" / "Dockerfile"


def _section_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def test_runtime_dependencies_switch_to_cuda13_packages() -> None:
    dockerfile = DOCKERFILE.read_text()
    runtime_section = _section_between(
        dockerfile,
        "# Install runtime dependencies",
        "# CUDA arch list used by torch",
    )

    assert 'if [ "$CUDA_MAJOR" -ge 13 ]; then \\' in runtime_section
    assert "uv pip uninstall cupy-cuda12x nixl nixl-cu12 || true" in runtime_section
    assert "uv pip install cupy-cuda13x nixl-cu13" in runtime_section
    assert "uv pip install -r cuda.txt" in runtime_section


def test_runtime_dependencies_preserve_cuda12_defaults() -> None:
    dockerfile = DOCKERFILE.read_text()
    runtime_section = _section_between(
        dockerfile,
        "# Install runtime dependencies",
        "# CUDA arch list used by torch",
    )
    cuda13_branch = runtime_section.split(
        'if [ "$CUDA_MAJOR" -ge 13 ]; then \\',
        maxsplit=1,
    )[1].split("fi", maxsplit=1)[0]

    assert "uv pip install -r cuda.txt" in runtime_section
    assert "cupy-cuda12x" in runtime_section
    assert "nixl-cu12" in runtime_section
    assert "cupy-cuda13x" in cuda13_branch
    assert "nixl-cu13" in cuda13_branch


def test_release_path_builds_lmcache_from_source_for_cuda13() -> None:
    dockerfile = DOCKERFILE.read_text()
    release_section = _section_between(
        dockerfile,
        "# Install vLLM stable release and LMCache.",
        "WORKDIR /workspace",
    )

    assert 'if [ "$CUDA_MAJOR" -ge 13 ]; then \\' in release_section
    assert "COPY ./requirements/build.txt /workspace/build.txt" in dockerfile
    assert "uv pip install -r /workspace/build.txt" in dockerfile
    assert "apt-get install -y --no-install-recommends ${BUILD_PKGS}" in release_section
    assert "uv pip install . --verbose --no-build-isolation --no-deps" in release_section
    assert "uv pip install lmcache --verbose" in release_section
    cuda13_branch = release_section.split(
        'if [ "$CUDA_MAJOR" -ge 13 ]; then \\',
        maxsplit=1,
    )[1].split("else \\", maxsplit=1)[0]
    cuda12_branch = release_section.split("else \\", maxsplit=1)[1]
    assert "uv pip install lmcache --verbose" not in cuda13_branch
    assert "cu12" not in cuda13_branch.lower()
    assert "uv pip install lmcache --verbose" in cuda12_branch
    assert "--no-build-isolation" not in cuda12_branch
