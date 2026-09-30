# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for wheel-first imports in XPU CI."""

# Standard
from pathlib import Path
import os
import shutil
import subprocess
import textwrap
import venv

# Third Party
import pytest


@pytest.mark.parametrize("existing_pythonpath", ["", "."])
def test_wheel_setup_exposes_repo_helpers_without_shadowing_wheel(
    tmp_path: Path, existing_pythonpath: str
) -> None:
    """Keep repo helpers and the installed wheel importable in spawned workers."""
    repo = tmp_path / "repo"
    harness = repo / ".buildkite/k3_harness"
    harness.mkdir(parents=True)
    script = harness / "setup-lmcache-only-env.sh"
    shutil.copyfile(
        Path(__file__).resolve().parents[2]
        / ".buildkite/k3_harness/setup-lmcache-only-env.sh",
        script,
    )
    helpers = repo / ".buildkite/k3_tests/common_scripts"
    helpers.mkdir(parents=True)
    (helpers / "helpers.sh").write_text(
        "check_gpu_health() { :; }\nmerge_pr_base_branch() { :; }\n"
    )
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    wheel = artifact / "test.whl"
    wheel.touch()
    xpu = repo / ".buildkite/k3_tests/xpu"
    xpu.mkdir()
    (xpu / "download-wheel.py").write_text(f"print({str(wheel)!r})\n")

    environment = tmp_path / "venv"
    venv.EnvBuilder(system_site_packages=True).create(environment)
    python = environment / "bin/python"
    site_packages = Path(
        subprocess.check_output(
            [
                str(python),
                "-I",
                "-c",
                'import sysconfig; print(sysconfig.get_path("purelib"))',
            ],
            text=True,
        ).strip()
    )
    (site_packages / "test-dependencies.pth").write_text(
        f"{Path(pytest.__file__).resolve().parents[1]}\n"
    )
    installed = site_packages / "lmcache"
    installed.mkdir()
    (installed / "__init__.py").write_text('ORIGIN = "wheel"\n')
    conflicting_tests = site_packages / "tests"
    conflicting_tests.mkdir()
    (conflicting_tests / "__init__.py").write_text("")
    source = repo / "lmcache"
    source.mkdir()
    (source / "__init__.py").write_text(
        'raise RuntimeError("Imported source instead of wheel")\n'
    )
    benchmarks = repo / "benchmarks"
    benchmarks.mkdir()
    (benchmarks / "helper.py").write_text("VALUE = 42\n")
    setup_extensions = repo / "setup_extensions"
    setup_extensions.mkdir()
    (setup_extensions / "__init__.py").write_text("VALUE = 43\n")
    test_package = repo / "tests"
    test_package.mkdir()
    (test_package / "__init__.py").write_text("")
    test_module = test_package / "v1"
    test_module.mkdir()
    (test_module / "__init__.py").write_text("")
    (test_module / "test_imports.py").write_text(
        textwrap.dedent(
            """\
            from pathlib import Path
            import multiprocessing as mp
            import sys
            import sysconfig
            import lmcache
            import tests
            from benchmarks.helper import VALUE
            from setup_extensions import VALUE as SETUP_VALUE

            def check_imports() -> None:
                assert lmcache.ORIGIN == "wheel"
                assert Path(lmcache.__file__).resolve().is_relative_to(
                    sysconfig.get_path("purelib")
                )
                assert Path(tests.__file__).resolve() == (
                    Path.cwd() / "tests/__init__.py"
                )
                assert VALUE == 42
                assert SETUP_VALUE == 43
                assert str(Path.cwd()) not in sys.path
                assert "." not in sys.path

            def test_imports() -> None:
                worker = mp.get_context("spawn").Process(target=check_imports)
                worker.start()
                try:
                    worker.join(timeout=20)
                    assert worker.exitcode == 0
                finally:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join()
                check_imports()
            """
        )
    )

    uv_args = tmp_path / "uv-args"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'uv() { printf "%s\\n" "$@" > "$UV_ARGS_FILE"; }\nsource "$1"\n'
            "python -m pytest --import-mode=importlib -q tests/v1/test_imports.py",
            "test-wheel-imports",
            str(script),
        ],
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{environment / 'bin'}:{os.environ['PATH']}",
            "PYTHONPATH": existing_pythonpath,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "XPU_WHEEL_ARTIFACT_ID": "123",
            "UV_ARGS_FILE": str(uv_args),
            "TMPDIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert uv_args.read_text().splitlines() == ["pip", "install", str(wheel)]
    assert "LMCache loaded from wheel:" in result.stdout
    assert "1 passed" in result.stdout


@pytest.mark.parametrize("expected_commit", ["current", "wrong", "missing"])
def test_candidate_checkout_requires_wheel_commit(expected_commit: str) -> None:
    """Reject mismatched revisions without invoking the PR-base merge path."""
    repo = Path(__file__).resolve().parents[2]
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    env = {
        **os.environ,
        "XPU_CANDIDATE_VALIDATION": "1",
        "LMCACHE_PR_BASE_MERGE": "always",
    }
    if expected_commit != "missing":
        env["XPU_SOURCE_COMMIT"] = actual if expected_commit == "current" else "0" * 40
    else:
        env.pop("XPU_SOURCE_COMMIT", None)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; merge_pr_base_branch',
            "test-xpu-candidate",
            str(repo / ".buildkite/k3_tests/common_scripts/helpers.sh"),
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if expected_commit == "current":
        assert result.returncode == 0, result.stderr
        assert "skipping PR-base pre-merge" in result.stdout
    else:
        assert result.returncode != 0
        assert "XPU" in result.stderr
