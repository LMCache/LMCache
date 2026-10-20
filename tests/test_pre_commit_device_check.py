# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the direct torch device pre-commit check."""

# Standard
from pathlib import Path
import sys

# Third Party
import pytest

# First Party
from tools.check_direct_torch_device import direct_device_lines, main


def test_ignores_textual_device_references() -> None:
    source = (
        "# torch.cuda.synchronize()\n"
        '"""torch.xpu.synchronize()"""\n'
        'target = "torch.cuda.Stream"\n'
    )
    assert direct_device_lines(source, {"cuda", "xpu"}) == set()


def test_rejects_executable_device_references() -> None:
    source = "torch.cuda.synchronize()\nvalue = torch.xpu.current_device()\n"
    assert direct_device_lines(source, {"cuda", "xpu"}) == {1, 2}


def test_rejects_import_aliases() -> None:
    source = (
        "import torch as pytorch\n"
        "pytorch.cuda.synchronize()\n"
        "from torch import xpu as accelerator\n"
        "accelerator.synchronize()\n"
    )
    assert direct_device_lines(source, {"cuda", "xpu"}) == {2, 3, 4}


def test_rejects_dynamic_device_access() -> None:
    source = 'import torch\ngetattr(torch, "cuda").synchronize()\n'
    assert direct_device_lines(source, {"cuda"}) == {2}


def test_ignores_shadowed_torch() -> None:
    source = (
        "import torch\n"
        "torch = object()\n"
        "torch.cuda = 1\n"
        "def check(torch):\n"
        "    return torch.xpu\n"
    )
    assert direct_device_lines(source, {"cuda", "xpu"}) == set()


def test_ignores_function_local_shadowing_before_assignment() -> None:
    source = (
        "import torch\n"
        "def check():\n"
        "    result = torch.cuda\n"
        "    torch = object()\n"
        "    return result\n"
    )
    assert direct_device_lines(source, {"cuda"}) == set()


@pytest.mark.parametrize(
    "statement",
    [
        "stream: torch.cuda.Stream",
        "stream: torch.cuda.Stream = None",
        "stream: torch.cuda.synchronize() = None",
        "def check(stream: torch.cuda.Stream): pass",
        "async def check(stream: torch.cuda.Stream): pass",
        "def check(stream: torch.cuda.Stream, /): pass",
        "def check(*, stream: torch.cuda.Stream): pass",
        "def check(*streams: torch.cuda.Stream): pass",
        "def check(**streams: torch.cuda.Stream): pass",
        "def check() -> torch.cuda.Event: pass",
        "def check(torch: torch.cuda.Stream): pass",
        "class Check(metaclass=torch.cuda.Stream): pass",
        "class Check(backend=torch.cuda.synchronize()): pass",
    ],
)
def test_rejects_device_references_in_definition_fields(statement: str) -> None:
    assert direct_device_lines(f"import torch\n{statement}\n", {"cuda"}) == {2}


@pytest.mark.parametrize(
    "statement",
    [
        "torch: object",
        "if False:\n    torch = None",
        "if condition:\n    torch = None\nelse:\n    pass",
        "if condition:\n    pass\nelse:\n    torch = None",
        "if condition:\n    if other:\n        torch = None",
        "for torch in ():\n    pass",
        "for item in ():\n    torch = None",
        "while False:\n    torch = None",
        "for item in ():\n    break\nelse:\n    torch = None",
        "while condition:\n    break\nelse:\n    torch = None",
        "condition and (torch := None)",
        "condition or (torch := None)",
        "(torch := None) if condition else None",
        "None if condition else (torch := None)",
        "try:\n    may_fail()\n    torch = None\nexcept Exception:\n    pass",
        "try:\n    may_fail()\nexcept Exception:\n    torch = None",
        "match value:\n    case 0:\n        torch = None",
    ],
)
def test_conditional_binding_preserves_possible_torch(statement: str) -> None:
    source = f"import torch\n{statement}\ntorch.cuda.synchronize()\n"
    assert direct_device_lines(source, {"cuda"}) == {len(source.splitlines())}


@pytest.mark.parametrize(
    "statement",
    [
        "if condition:\n    import torch as backend\nelse:\n    backend = None",
        "if condition:\n    backend = None\nelse:\n    import torch as backend",
        "for item in items:\n    import torch as backend",
        "try:\n    import torch as backend\n    may_fail()\nexcept Exception:\n"
        "    backend = None",
    ],
)
def test_conditional_import_preserves_possible_torch(statement: str) -> None:
    source = f"{statement}\nbackend.cuda.synchronize()\n"
    assert direct_device_lines(source, {"cuda"}) == {len(source.splitlines())}


def test_conditional_branches_have_independent_bindings() -> None:
    source = (
        "import torch\n"
        "if condition:\n"
        "    torch = None\n"
        "else:\n"
        "    torch.cuda.synchronize()\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {5}


def test_exception_handler_sees_bindings_before_failure() -> None:
    source = (
        "import torch\n"
        "try:\n"
        "    may_fail()\n"
        "    torch = None\n"
        "except Exception:\n"
        "    torch.cuda.synchronize()\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {6}


def test_finally_sees_bindings_before_failure() -> None:
    source = (
        "import torch\n"
        "try:\n"
        "    may_fail()\n"
        "    torch = None\n"
        "finally:\n"
        "    torch.cuda.synchronize()\n"
        "torch.cuda\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {6}


@pytest.mark.skipif(sys.version_info < (3, 11), reason="requires exception groups")
def test_exception_group_preserves_possible_torch() -> None:
    source = (
        "import torch\n"
        "try:\n"
        "    may_fail()\n"
        "    torch = None\n"
        "except* Exception:\n"
        "    pass\n"
        "torch.cuda.synchronize()\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {7}


@pytest.mark.skipif(sys.version_info < (3, 12), reason="requires type parameters")
@pytest.mark.parametrize(
    "definition",
    [
        "def check[T: torch.cuda.Stream](): pass",
        "class Check[T: torch.cuda.Stream]: pass",
    ],
)
def test_rejects_device_references_in_type_parameters(definition: str) -> None:
    assert direct_device_lines(f"import torch\n{definition}\n", {"cuda"}) == {2}


def test_async_loop_preserves_possible_torch() -> None:
    source = (
        "async def check():\n"
        "    import torch\n"
        "    async for torch in items:\n"
        "        pass\n"
        "    torch.cuda.synchronize()\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {5}


def test_branch_merge_preserves_module_and_backend_bindings() -> None:
    source = (
        "if condition:\n"
        "    import torch as backend\n"
        "else:\n"
        "    from torch import cuda as backend\n"
        "backend.cuda.synchronize()\n"
        "backend.synchronize()\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {4, 5, 6}


@pytest.mark.parametrize(
    "statement",
    [
        "torch = None",
        "torch: object = None",
        "if condition:\n    torch = None\nelse:\n    torch = object()",
        "try:\n    may_fail()\nfinally:\n    torch = None",
    ],
)
def test_definite_rebinding_still_shadows_torch(statement: str) -> None:
    source = f"import torch\n{statement}\ntorch.cuda\n"
    assert direct_device_lines(source, {"cuda"}) == set()


@pytest.mark.parametrize("definition", ["def", "async def"])
def test_method_skips_class_bindings(definition: str) -> None:
    source = (
        "import torch\n"
        "class Check:\n"
        "    torch = None\n"
        f"    {definition} check(self):\n"
        "        torch.cuda.synchronize()\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {5}


def test_nested_class_skips_outer_class_bindings() -> None:
    source = (
        "import torch\n"
        "class Outer:\n"
        "    torch = None\n"
        "    class Inner:\n"
        "        stream = torch.cuda.Stream\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {5}


def test_method_resolves_enclosing_function_binding() -> None:
    source = (
        "def outer():\n"
        "    import torch as backend\n"
        "    class Check:\n"
        "        backend = None\n"
        "        def check(self):\n"
        "            backend.cuda.synchronize()\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {6}


def test_class_body_and_method_definition_use_class_bindings() -> None:
    source = (
        "import torch\n"
        "class Check:\n"
        "    torch = other\n"
        "    stream = torch.cuda\n"
        "    def check(self, stream: torch.cuda = torch.cuda) -> torch.cuda:\n"
        "        pass\n"
    )
    assert direct_device_lines(source, {"cuda"}) == set()


def test_class_annotation_does_not_shadow_global_torch() -> None:
    source = (
        "import torch\n"
        "class Check:\n"
        "    torch: object\n"
        "    stream = torch.cuda.Stream\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {4}


def test_method_local_binding_still_shadows_global_torch() -> None:
    source = (
        "import torch\n"
        "class Check:\n"
        "    def check(self, torch):\n"
        "        return torch.cuda\n"
        "    def other(self):\n"
        "        stream = torch.cuda\n"
        "        torch = None\n"
    )
    assert direct_device_lines(source, {"cuda"}) == set()


@pytest.mark.parametrize(
    "source, line",
    [
        ("import torch\nstream: torch.cuda.Stream\n", 2),
        ("import torch\nif False:\n    torch = None\ntorch.cuda.synchronize()\n", 4),
        (
            "import torch\nclass Check:\n    torch = None\n"
            "    def check(self):\n        torch.cuda.synchronize()\n",
            5,
        ),
        ("import torch\ntorch = object()\nvalue = torch.cuda\n", None),
    ],
)
def test_main_reports_violations_and_exit_status(
    source: str,
    line: int | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    path = tmp_path / "test_device_check.py"
    path.write_text(source, encoding="utf-8")
    status = main([str(path)])
    output = capsys.readouterr().out
    if line is None:
        assert status == 0
        assert output == ""
    else:
        assert status == 1
        assert "BANNED: Direct torch device API usage" in output
        assert f"{path}:{line}: {source.splitlines()[line - 1]}" in output
