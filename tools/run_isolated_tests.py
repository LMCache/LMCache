# SPDX-License-Identifier: Apache-2.0
"""Run checkout tests against either the checkout or an installed wheel."""

# Standard
from importlib import import_module, metadata, util
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

# Third Party
import pytest

_expected: Path | None = None
_staged: Path | None = None


def _check_editable(checkout: Path) -> None:
    """Verify editable metadata and import the checkout without PYTHONPATH."""
    try:
        distribution = metadata.distribution("lmcache")
    except metadata.PackageNotFoundError as exc:
        raise SystemExit("Editable check: lmcache is not installed") from exc
    try:
        origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Editable check: invalid direct_url.json: {exc}") from exc
    if (
        not isinstance(origin, dict)
        or not isinstance(origin.get("dir_info"), dict)
        or origin["dir_info"].get("editable") is not True
    ):
        raise SystemExit("Editable check: lmcache is not installed editable")
    url = origin.get("url")
    if not isinstance(url, str):
        raise SystemExit("Editable check: installation URL is missing")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "file"
        or parsed.netloc not in ("", "localhost")
        or parsed.query
        or parsed.fragment
    ):
        raise SystemExit(f"Editable check: invalid local installation URL: {url}")
    installed = Path(url2pathname(parsed.path)).resolve()
    if installed != checkout:
        raise SystemExit(
            f"Editable check: installed from {installed}, expected {checkout}"
        )
    spec = util.find_spec("lmcache")
    expected = checkout / "lmcache/__init__.py"
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
        raise SystemExit(f"Editable check: expected {expected}, found {spec}")
    module = import_module("lmcache")
    if module.__file__ is None or Path(module.__file__).resolve() != expected:
        raise SystemExit(f"Editable check: unexpected imported module {module}")
    print(f"Editable install verified: {module.__file__}", flush=True)


def pytest_sessionstart(session: pytest.Session) -> None:
    """Reject unexpected package origins before collecting isolated tests."""
    global _expected, _staged
    mode = os.environ["LMCACHE_TEST_IMPORT_MODE"]
    checkout = Path(os.environ["LMCACHE_TEST_CHECKOUT"]).resolve()
    _staged = Path(os.environ["LMCACHE_TEST_STAGED"]).resolve()
    test_spec = util.find_spec("tests")
    if test_spec is None or test_spec.origin is None:
        raise pytest.UsageError("Cannot find the staged tests package")
    if Path(test_spec.origin).resolve() != _staged / "tests/__init__.py":
        raise pytest.UsageError(f"Unexpected tests package: {test_spec.origin}")
    spec = util.find_spec("lmcache")
    if spec is None or spec.origin is None:
        raise pytest.UsageError("Cannot find lmcache in the selected environment")
    actual = Path(spec.origin).resolve()
    if mode == "source":
        _expected = checkout / "lmcache/__init__.py"
    else:
        try:
            distribution = metadata.distribution("lmcache")
        except metadata.PackageNotFoundError as exc:
            raise pytest.UsageError("wheel mode requires an installed lmcache") from exc
        direct_url = distribution.read_text("direct_url.json")
        if direct_url and json.loads(direct_url).get("dir_info", {}).get("editable"):
            raise pytest.UsageError("wheel mode does not support editable installs")
        if distribution.read_text("WHEEL") is None or not any(
            str(file) == "lmcache/__init__.py" for file in distribution.files or ()
        ):
            raise pytest.UsageError("lmcache is not installed from a recorded wheel")
        _expected = Path(distribution.locate_file("lmcache/__init__.py")).resolve()
        if actual.is_relative_to(checkout) or actual.is_relative_to(_staged):
            raise pytest.UsageError(f"wheel mode imported checkout code: {actual}")
    if actual != _expected:
        raise pytest.UsageError(f"Expected lmcache at {_expected}, found {actual}")
    print(f"\nLMCache test mode: {mode}; package: {actual}")


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Reject tests or imports that escaped isolation after collection."""
    if _staged is None or _expected is None:
        raise pytest.UsageError("The isolated import check was not initialized")
    if not item.path.resolve().is_relative_to(_staged):
        raise pytest.UsageError(f"Test is outside the staged tree: {item.path}")
    spec = util.find_spec("lmcache")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != _expected:
        raise pytest.UsageError(f"LMCache import origin changed after startup: {spec}")


def main() -> int:
    """Stage repository resources without the product package and invoke pytest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("source", "wheel"), required=True)
    parser.add_argument(
        "--check-editable",
        action="store_true",
        help="verify the editable install before running source-mode tests",
    )
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.check_editable and args.mode != "source":
        parser.error("--check-editable requires --mode source")
    root = Path(__file__).resolve().parents[1]
    files = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ]
    ).split(b"\0")

    with tempfile.TemporaryDirectory(prefix="lmcache-tests-") as directory:
        staged = Path(directory)
        for raw in files:
            if not raw:
                continue
            relative = Path(os.fsdecode(raw))
            if relative.parts[0] == "lmcache":
                continue
            source = root / relative
            if not source.is_file() and not source.is_symlink():
                continue
            destination = staged / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                destination.symlink_to(os.readlink(source))
            else:
                shutil.copy2(source, destination)

        env = os.environ.copy()
        env["LMCACHE_TEST_IMPORT_MODE"] = args.mode
        env["LMCACHE_TEST_CHECKOUT"] = str(root)
        env["LMCACHE_TEST_STAGED"] = str(staged)
        runner = str(staged / "tools" / Path(__file__).name)
        if args.check_editable:
            probe_env = env.copy()
            probe_env.pop("PYTHONPATH", None)
            result = subprocess.call(
                [
                    sys.executable,
                    runner,
                    "--_check-editable",
                    str(root),
                ],
                cwd=staged,
                env=probe_env,
            )
            if result:
                return result
        paths = [str(staged)]
        if args.mode == "source":
            paths.append(str(root))
        env["PYTHONPATH"] = os.pathsep.join(paths)
        pytest_args = args.pytest_args
        if pytest_args and pytest_args[0] == "--":
            pytest_args = pytest_args[1:]
        command = [
            sys.executable,
            runner,
            "--_pytest",
            *pytest_args,
        ]
        return subprocess.call(command, cwd=staged, env=env)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--_check-editable"]:
        _check_editable(Path(sys.argv[2]).resolve())
    elif sys.argv[1:2] == ["--_pytest"]:
        sys.exit(pytest.main(sys.argv[2:], plugins=[sys.modules[__name__]]))
    else:
        sys.exit(main())
