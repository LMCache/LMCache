# SPDX-License-Identifier: Apache-2.0
"""Directory discovery and lazy imports, including fresh-interpreter checks."""

# Standard
from pathlib import Path
import subprocess
import sys
import textwrap

# Third Party
import pytest

# First Party
from lmcache.v1.gpu_connector import gds_backends
from lmcache.v1.gpu_connector._gds_backends import available_backends, create_backend


@pytest.mark.parametrize(
    ("selection", "platform", "expected"),
    [
        ("import_only", "none", None),
        ("discover_only", "none", None),
        ("missing", "none", None),
        ("cufile", "cuda", "cufile"),
        ("hipfile", "hip", "hipfile"),
        ("ugds", "cuda", "ugds"),
        ("phx", "hip", "phx"),
        ("auto", "cuda", "cufile"),
        ("auto", "hip", "hipfile"),
    ],
)
def test_lazy_imports_in_fresh_interpreter(
    selection: str, platform: str, expected: str | None
) -> None:
    # Other test modules import concrete backends during collection. A fresh
    # interpreter ensures those imports cannot hide an eager factory import.
    script = textwrap.dedent(
        """
        from unittest.mock import Mock
        import ctypes
        import sys
        import torch
        from lmcache.v1.gpu_connector import _gds_backends as factory

        prefix = "lmcache.v1.gpu_connector.gds_backends."
        def loaded():
            return {
                name.removeprefix(prefix) for name in sys.modules
                if name.startswith(prefix)
                and not name.removeprefix(prefix).startswith("_")
            }

        assert loaded() == set(), loaded()
        ctypes.CDLL = Mock(side_effect=AssertionError("native library loaded"))
        selection, platform, expected = sys.argv[1:]
        torch.version.cuda = "test" if platform == "cuda" else None
        torch.version.hip = "test" if platform == "hip" else None
        if selection == "import_only":
            assert loaded() == set()
        else:
            names = factory.available_backends()
            assert names and all(not name.startswith("_") for name in names)
            assert loaded() == set(), loaded()
            if selection == "missing":
                try:
                    factory.create_backend(selection)
                except ValueError:
                    pass
                else:
                    raise AssertionError("unknown backend accepted")
                assert loaded() == set(), loaded()
            elif selection != "discover_only":
                backend = factory.create_backend(selection)
                assert backend.name == expected
                assert factory.create_backend(selection) is not backend
                candidates = (
                    set(names[:names.index(expected) + 1])
                    if selection == "auto" else {expected}
                )
                assert loaded() == candidates, loaded()
                backend.close_driver()
        assert "cufile.bindings" not in sys.modules
        assert "hipfile" not in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, selection, platform, expected or ""],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_discovery_does_not_execute_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "unavailable.py").write_text("raise RuntimeError('must stay lazy')\n")
    (tmp_path / "_helper.py").write_text("raise RuntimeError('private helper')\n")
    monkeypatch.setattr(gds_backends, "__path__", [str(tmp_path)])
    assert available_backends() == ("unavailable",)
    assert f"{gds_backends.__name__}.unavailable" not in sys.modules


@pytest.mark.parametrize("name", ["_helper", "__init__", "../outside", "missing"])
def test_only_discovered_public_names_can_be_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    (tmp_path / "_helper.py").write_text("raise RuntimeError('private helper')\n")
    monkeypatch.setattr(gds_backends, "__path__", [str(tmp_path)])
    with pytest.raises(ValueError, match="unsupported GDS L1 backend"):
        create_backend(name)


@pytest.mark.parametrize("source", ["", "Backend = 42\n", "class Backend: pass\n"])
def test_backend_module_must_export_an_interface_subclass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    (tmp_path / "invalid.py").write_text(source)
    monkeypatch.setattr(gds_backends, "__path__", [str(tmp_path)])
    try:
        with pytest.raises(TypeError, match="GDS backend 'invalid' must export"):
            create_backend("invalid")
    finally:
        sys.modules.pop(f"{gds_backends.__name__}.invalid", None)
