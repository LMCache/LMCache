# SPDX-License-Identifier: Apache-2.0
"""Regressions for installation-independent paths and the pre-commit guard."""

# Standard
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
import importlib.util
import os
import subprocess
import sys

# Third Party
import pytest

# First Party
from tests import import_utils

_CHECK_PATH = import_utils.repo_path("tools", "check_test_imports.py")
_spec = importlib.util.spec_from_file_location("_test_import_checker", _CHECK_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load test import check from {_CHECK_PATH}")
_checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_checker)

_FILENAME = import_utils.repo_path("tests", "v1", "test_example.py")
_HEADER = (
    "from pathlib import Path\n"
    "import os, sys, subprocess\n"
    "root = Path(__file__).resolve().parents[2]\n"
)


@pytest.mark.parametrize("location", ["checkout", "editable checkout", "site-packages"])
def test_selected_package_paths_and_subprocess(
    location: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolve source/install-shaped paths and import the same package in a child."""
    root = tmp_path / location
    package = root / "lmcache"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    init.write_text("VALUE = 42\n", encoding="utf-8")
    module = ModuleType("lmcache")
    module.__spec__ = ModuleSpec("lmcache", loader=None, origin=str(init))
    monkeypatch.setitem(sys.modules, "lmcache", module)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "test-helpers"))
    monkeypatch.setenv("LMCACHE_IMPORT_TEST_SENTINEL", "preserved")
    original = os.environ.copy()

    assert import_utils.lmcache_path("integration", "adapter.py") == (
        package / "integration/adapter.py"
    )
    env = import_utils.lmcache_subprocess_env()
    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(root),
        str(tmp_path / "test-helpers"),
    ]
    assert env["LMCACHE_IMPORT_TEST_SENTINEL"] == "preserved"
    assert os.environ == original
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import lmcache; assert lmcache.VALUE == 42; print(lmcache.__file__)",
        ],
        cwd=import_utils.lmcache_path().parent,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert Path(result.stdout.strip()) == init


@pytest.mark.parametrize("spec", [None, ModuleSpec("lmcache", loader=None)])
def test_missing_package_is_explicit(
    spec: ModuleSpec | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(import_utils, "find_spec", lambda _: spec)
    with pytest.raises(ImportError, match="Cannot locate"):
        import_utils.lmcache_path()
    with pytest.raises(ImportError, match="Cannot locate"):
        import_utils.lmcache_subprocess_env()


def test_repo_resources_are_independent_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert import_utils.repo_path("tests") == Path(__file__).resolve().parent


@pytest.mark.parametrize(
    ("source", "code"),
    [
        (
            'path = Path(__file__).parents[2] / "lmcache" / "integration"',
            "TCI001",
        ),
        (
            "_REPO_ROOT = Path(__file__).resolve().parents[2]\n"
            "def run():\n    subprocess.run(['python'], cwd=_REPO_ROOT)",
            "TCI003",
        ),
        ('path = root / "lmcache" / "integration" / "adapter.py"', "TCI001"),
        ('path = root.joinpath("lmcache", "integration", "adapter.py")', "TCI001"),
        (
            "from tests.import_utils import repo_path as resource\n"
            'path = resource("lmcache", "integration/adapter.py")',
            "TCI001",
        ),
        ("sys.path.insert(0, str(root))", "TCI002"),
        ("sys.path.extend([str(root)])", "TCI002"),
        ("from sys import path as paths\npaths.append(str(root))", "TCI002"),
        ('env["PYTHONPATH"] = str(root)', "TCI002"),
        ('env = {"PYTHONPATH": os.pathsep.join([str(root), "helpers"])}', "TCI002"),
        ("env.update(PYTHONPATH=str(root))", "TCI002"),
        ("subprocess.run(['python', '-c', 'import lmcache'], cwd=root)", "TCI003"),
        ('saved = root\nroot = unknown()\npath = saved / "lmcache"', "TCI001"),
        ('saved: Path = root\npath = saved / "lmcache"', "TCI001"),
        (
            "def test_one():\n"
            "    root = unknown()\n"
            "def test_two():\n"
            '    path = root / "lmcache"',
            "TCI001",
        ),
    ],
)
def test_guard_rejects_known_mistakes(source: str, code: str) -> None:
    errors = _checker.check_test_imports(_HEADER + source, _FILENAME)
    assert errors
    assert any(message.startswith(code) for _, message in errors)
    assert all(line >= 4 for line, _ in errors)


@pytest.mark.parametrize(
    "source",
    [
        'path = root / "tests" / "data.json"',
        "from tests.import_utils import lmcache_path\n"
        'path = lmcache_path("integration", "adapter.py")',
        "import lmcache\npath = Path(lmcache.__file__).parent / 'integration'",
        "subprocess.run(['python'], cwd=tmp_path)",
        "sys.path.insert(0, str(tmp_path))",
        'sys.path.insert(0, str(root / "tests"))',
        'env["PYTHONPATH"] = str(root / "test-helpers")',
        'subprocess.run(["python"], cwd=root / "tests")',
        "# root / 'lmcache'\ntext = \"root / 'lmcache'\"",
        "# test-import: allow -- Scan checkout sources for an AST contract.\n"
        'path = root / "lmcache"',
        'path = root / "lmcache"  # test-import: allow -- Build source files.',
        "# test-import: allow -- Read checkout sources, not installed code.\n"
        'def test_source_contract():\n    path = root / "lmcache"',
        "def test_one(root):\n    sys.path.insert(0, str(root))",
        'class Example:\n    root = unknown()\npath = root / "tests"',
    ],
)
def test_guard_allows_resources_and_explicit_exemptions(source: str) -> None:
    assert _checker.check_test_imports(_HEADER + source, _FILENAME) == []


def test_exemption_requires_a_reason() -> None:
    errors = _checker.check_test_imports(
        _HEADER + '# test-import: allow\npath = root / "lmcache"', _FILENAME
    )
    assert errors


def test_inline_exemption_does_not_hide_the_next_statement() -> None:
    errors = _checker.check_test_imports(
        _HEADER
        + 'path = root / "lmcache"  # test-import: allow -- Build source files.\n'
        + 'other = root / "lmcache"',
        _FILENAME,
    )
    assert len(errors) == 1
    assert errors[0][0] == 5


def test_guard_cli_reports_violations_and_bad_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "test_example.py"
    path.write_text(
        "from tests.import_utils import repo_path\n"
        'path = repo_path("lmcache/integration/adapter.py")\n',
        encoding="utf-8",
    )
    assert _checker.main([str(path)]) == 1
    assert "TCI001" in capsys.readouterr().out
    path.write_text("def broken(", encoding="utf-8")
    assert _checker.main([str(path)]) == 1
    assert "test_example.py" in capsys.readouterr().out
    assert _checker.main([str(tmp_path / "missing.py")]) == 1
    assert "missing.py" in capsys.readouterr().out
