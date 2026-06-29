# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``tools/gen_engine_kv_format.py``.

The generator renders ``lmcache/_engine_kv_format.py`` (the static, typed Python
mirror of the C++ ``EngineKVFormat`` enum) from the X-macro spec in
``csrc/engine_kv_format.def``. These tests exercise its public functions
(``parse_members`` / ``render``) and verify the committed generated file is
up to date -- all without a GPU or the compiled extension.
"""

# Standard
from pathlib import Path
from types import ModuleType
import importlib.util

# Third Party
import pytest

_GEN_PATH = Path(__file__).resolve().parent.parent / "tools" / "gen_engine_kv_format.py"


def _load_generator() -> ModuleType:
    """Load the standalone generator script as a module (it is not a package)."""
    spec = importlib.util.spec_from_file_location("gen_engine_kv_format", _GEN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generator from {_GEN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_generator()


def test_parse_members_basic_order_and_values() -> None:
    """Members are returned in file order with their integer values."""
    members = gen.parse_members("X(A, 0)\nX(B, 1)\nX(C, 2)\n")
    assert [(name, value) for name, value, _ in members] == [
        ("A", 0),
        ("B", 1),
        ("C", 2),
    ]


def test_parse_members_attaches_preceding_comments() -> None:
    """``//`` comment lines directly above an entry attach to it."""
    members = gen.parse_members("// used by: foo\n// shape: bar\nX(A, 0)\n")
    assert members[0][2] == ["used by: foo", "shape: bar"]


def test_parse_members_blank_line_resets_comments() -> None:
    """A blank line detaches a comment block from a following entry."""
    members = gen.parse_members("// header\n\nX(A, 0)\n")
    assert members[0][2] == []


def test_parse_members_rejects_duplicate_value() -> None:
    """Two members sharing an integer value is an error."""
    with pytest.raises(ValueError):
        gen.parse_members("X(A, 0)\nX(B, 0)\n")


def test_parse_members_rejects_duplicate_name() -> None:
    """Two members sharing a name is an error."""
    with pytest.raises(ValueError):
        gen.parse_members("X(A, 0)\nX(A, 1)\n")


def test_parse_members_allows_trailing_comment() -> None:
    """An inline ``// ...`` trailing comment on an ``X(...)`` line is tolerated."""
    members = gen.parse_members("X(A, 0)  // inline note\n")
    assert [(name, value) for name, value, _ in members] == [("A", 0)]


def test_parse_members_rejects_no_members() -> None:
    """A spec with no ``X(...)`` entries is an error."""
    with pytest.raises(ValueError):
        gen.parse_members("// only comments, no entries\n")


def test_render_produces_valid_python_with_alias() -> None:
    """Rendered output compiles and contains the enum, members, and alias."""
    code = gen.render(gen.parse_members("// c\nX(A, 0)\nX(B, 1)\n"))
    compile(code, "<generated>", "exec")  # raises SyntaxError if malformed
    assert "class EngineKVFormat(IntEnum):" in code
    assert "    A = 0" in code
    assert "    B = 1" in code
    assert "GPUKVFormat = EngineKVFormat" in code


def test_committed_generated_file_is_up_to_date() -> None:
    """The checked-in mirror matches a fresh regeneration from the .def."""
    expected = gen.render(gen.parse_members(gen.DEF_PATH.read_text(encoding="utf-8")))
    actual = gen.OUT_PATH.read_text(encoding="utf-8")
    assert actual == expected, (
        "lmcache/_engine_kv_format.py is stale; "
        "run `python tools/gen_engine_kv_format.py` and commit the result."
    )
