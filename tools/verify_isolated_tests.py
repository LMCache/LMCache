# SPDX-License-Identifier: Apache-2.0
"""Manual runner regressions: python -m pytest tools/verify_isolated_tests.py."""

# Standard
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

# Third Party
import pytest


@pytest.fixture(scope="module", params=["source", "wheel", "editable"])
def environment(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, Path, Path, str]:
    """Prepare a tiny checkout and install it only for wheel/editable checks."""
    tmp = tmp_path_factory.mktemp("isolated-runner")
    checkout = tmp / "checkout # 100%"
    (checkout / "tools").mkdir(parents=True)
    (checkout / "tests").mkdir()
    (checkout / "lmcache").mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    (checkout / ".gitignore").write_text("build/\n*.egg-info/\n")
    runner = Path(__file__).with_name("run_isolated_tests.py")
    shutil.copy2(runner, checkout / "tools")
    (checkout / "tests/__init__.py").write_text("")
    (checkout / "lmcache/__init__.py").write_text("VALUE = 42\n")
    (checkout / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n'
        '[project]\nname = "lmcache"\nversion = "1.0"\n'
        '[tool.setuptools]\npackages = ["lmcache"]\n'
    )
    (checkout / "tests/test_smoke.py").write_text(
        "import subprocess, sys\n"
        "from pathlib import Path\n"
        "import lmcache\n"
        "def test_origin():\n"
        "    assert lmcache.VALUE == 42\n"
        "    assert not (Path.cwd() / 'lmcache').exists()\n"
        "    subprocess.run([sys.executable, '-c', "
        "'import lmcache; assert lmcache.VALUE == 42'], check=True)\n"
    )
    subprocess.run([sys.executable, "-m", "venv", str(tmp / "venv")], check=True)
    python = tmp / "venv/bin/python"
    site = Path(
        subprocess.check_output(
            [
                str(python),
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            text=True,
        ).strip()
    )
    (site / "test-deps.pth").write_text(
        str(Path(pytest.__file__).resolve().parents[1]) + "\n"
    )
    mode = str(request.param)
    if mode != "source":
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                *(["-e"] if mode == "editable" else []),
                str(checkout),
            ],
            cwd=tmp,
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            check=True,
        )
    return checkout, python, site, mode


def _run(
    environment: tuple[Path, Path, Path, str], *pytest_args: str
) -> subprocess.CompletedProcess[str]:
    checkout, python, _, mode = environment
    return subprocess.run(
        [
            str(python),
            str(checkout / "tools/run_isolated_tests.py"),
            "--mode",
            "source" if mode == "editable" else mode,
            *(["--check-editable"] if mode == "editable" else []),
            "--",
            "-q",
            "--import-mode=importlib",
            *pytest_args,
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(checkout),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )


def test_parent_and_child_imports(
    environment: tuple[Path, Path, Path, str], tmp_path: Path
) -> None:
    """Exercise all three import methods and preserve an absolute report path."""
    report = tmp_path / "results.xml"
    result = _run(environment, "tests/test_smoke.py", f"--junitxml={report}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert report.is_file()
    if environment[3] == "editable":
        assert "Editable install verified:" in result.stdout


def test_rejects_checkout_test_paths(
    environment: tuple[Path, Path, Path, str],
) -> None:
    result = _run(environment, str(environment[0] / "tests/test_smoke.py"))
    assert result.returncode != 0
    assert "Test is outside the staged tree" in result.stdout + result.stderr


def test_rejects_wrong_package_origin(
    environment: tuple[Path, Path, Path, str],
) -> None:
    """Ensure path injection cannot mask broken or mismatched installations."""
    checkout, _, site, mode = environment
    if mode == "editable":
        target = next(site.glob("__editable__*.pth"))
        original = target.read_text()
        target.write_text("")
        error = "Editable check: expected"
    elif mode == "wheel":
        target = next(site.glob("lmcache-*.dist-info/direct_url.json"))
        original = target.read_text()
        target.write_text(json.dumps({"dir_info": {"editable": True}}))
        error = "wheel mode does not support editable installs"
    else:
        target = checkout / "lmcache/__init__.py"
        original = target.read_text()
        target.unlink()
        error = "Cannot find lmcache"
    try:
        result = _run(environment, "tests/test_smoke.py")
    finally:
        target.write_text(original)
    assert result.returncode != 0
    assert error in result.stdout + result.stderr
