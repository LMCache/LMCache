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


@pytest.mark.parametrize(
    "setup, expression",
    [
        ("t = torch", "t.cuda.synchronize()"),
        ("a = torch\nt = a", "t.xpu.synchronize()"),
        ("a = t = torch", "t.cuda.synchronize()"),
        ("t: object = torch", "t.cuda.synchronize()"),
        ("", "(t := torch).cuda.synchronize()"),
        ("(t := torch)", "t.cuda.synchronize()"),
        ("t = torch if condition else None", "t.cuda.synchronize()"),
        ("t = None\nif condition:\n    t = torch", "t.cuda.synchronize()"),
        ("t = torch\ntorch = None", "t.cuda.synchronize()"),
        ("", "[torch.cuda.Stream()]"),
        ("modules = [torch]", "modules[0].cuda.synchronize()"),
        ("modules = [None, torch]", "modules[-1].cuda.synchronize()"),
        ("modules = [None, torch]", "modules[index].cuda.synchronize()"),
        ("modules = [[torch]]", "modules[0][0].cuda.synchronize()"),
        ("modules = [torch]\nother = modules", "other[0].cuda.synchronize()"),
        ("modules = [torch]", 'getattr(modules[0], "cuda").synchronize()'),
        ("modules = [None, torch][1:]", "modules[0].cuda.synchronize()"),
        ("modules = [torch]\nexpanded = [*modules]", "expanded[0].cuda"),
        ("modules = [None, torch]\nexpanded = [*modules]", "expanded[1].cuda"),
        ("modules = [None, torch][::-1]", "modules[0].cuda"),
        ("[t] = [torch]", "t.cuda.synchronize()"),
        ("_, t = None, torch", "t.cuda.synchronize()"),
        ("[_, [t]] = [None, [torch]]", "t.cuda.synchronize()"),
        ("*rest, t = [None, torch]", "t.cuda.synchronize()"),
        ("_, *rest = [None, torch]", "rest[0].cuda.synchronize()"),
        ("t = None\nfor t in [torch]:\n    pass", "t.cuda.synchronize()"),
        ("", "[t.cuda.synchronize() for t in [torch]]"),
        ("", "{t.cuda.Stream for t in (torch,)}"),
        ("", "(t.cuda.Stream for t in [torch])"),
        ("", "{t.cuda.Stream: None for t in [torch]}"),
        ("", "{None: t.cuda.Stream for t in [torch]}"),
        ("", "[None for t in [torch] if t.cuda.is_available()]"),
        ("", "[t.cuda.Stream for group in [[torch]] for t in group]"),
        ("", "[t.cuda.Stream for _, t in [(None, torch)]]"),
        ("modules = [t for t in [torch]]", "modules[0].cuda.Stream"),
        ("", "(lambda: torch.cuda.synchronize())"),
        ("", "(lambda t=torch: t.cuda.synchronize())"),
        ("", "(lambda *, t=torch: t.cuda.synchronize())"),
        ("", "(lambda: (t := torch).cuda.synchronize())"),
    ],
)
def test_rejects_alias_and_container_device_access(setup: str, expression: str) -> None:
    source = f"import torch\n{setup}\n{expression}\n"
    assert direct_device_lines(source, {"cuda", "xpu"}) == {len(source.splitlines())}


@pytest.mark.parametrize(
    "source",
    [
        "t = torch\nt = object()\nt.cuda",
        "modules = [None, torch]\nmodules[0].cuda",
        "modules = [torch, None]\nmodules[-1].cuda",
        "modules = [torch, None][1:]\nmodules[0].cuda",
        "modules = [None, torch][:-1]\nmodules[0].cuda",
        "modules = [None, torch]\nexpanded = [*modules]\nexpanded[0].cuda",
        "modules = [torch]\nmodules.cuda",
        "t, other = [None, torch]\nt.cuda",
        "f = lambda torch: torch.cuda",
        "f = lambda torch, /: torch.cuda",
        "f = lambda *, torch: torch.cuda",
        "f = lambda *torch: torch.cuda",
        "f = lambda **torch: torch.cuda",
        "f = lambda: (torch.cuda, (torch := None))",
        "[torch.cuda for torch in objects]",
        "{torch.cuda for torch in objects}",
        "(torch.cuda for torch in objects)",
        "{torch.cuda: torch.xpu for torch in objects}",
        "[torch.cuda for torch in objects if torch.xpu]",
        "[torch.cuda for item in objects for torch in objects]",
        "class C:\n    import torch as local\n    f = lambda: local.cuda",
    ],
)
def test_alias_and_expression_scope_do_not_create_false_positives(
    source: str,
) -> None:
    assert direct_device_lines(f"import torch\n{source}\n", {"cuda", "xpu"}) == set()


@pytest.mark.parametrize(
    "expression",
    [
        "lambda torch: torch.cuda",
        "[torch.cuda for torch in objects]",
        "{torch.cuda for torch in objects}",
        "(torch.cuda for torch in objects)",
        "{torch.cuda: None for torch in objects}",
        "[(torch := None) for item in ()]",
        "lambda: (torch := None)",
        "[(lambda torch: torch.cuda)(item) for item in objects]",
    ],
)
def test_expression_scope_preserves_outer_binding(expression: str) -> None:
    source = f"import torch\nvalue = {expression}\ntorch.cuda.synchronize()\n"
    assert direct_device_lines(source, {"cuda"}) == {3}


def test_comprehension_outer_iterable_uses_enclosing_scope() -> None:
    source = "import torch\nvalues = [torch for torch in torch.cuda.devices]\n"
    assert direct_device_lines(source, {"cuda"}) == {2}


def test_class_expression_scopes_skip_class_namespace() -> None:
    source = (
        "import torch\n"
        "class C:\n"
        "    torch = None\n"
        "    f = lambda: torch.cuda\n"
        "    values = [torch.cuda for item in objects]\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {4, 5}


def test_class_comprehension_iterable_uses_class_namespace() -> None:
    source = (
        "class C:\n    import torch as local\n    values = [t.cuda for t in [local]]\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {3}


def test_nested_comprehension_scopes_do_not_leak() -> None:
    source = (
        "import torch\n"
        "values = [(torch.cuda, [torch.cuda for torch in objects]) "
        "for torch in [torch]]\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {2}


def test_comprehension_walrus_binds_enclosing_function() -> None:
    source = (
        "def check():\n"
        "    import torch\n"
        "    values = [(t := torch) for item in objects]\n"
        "    t.cuda.synchronize()\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {4}


def test_loop_and_unpack_keep_alias_values_before_rebinding() -> None:
    source = (
        "import torch\n"
        "torch, other = None, torch\n"
        "for _, t in [(None, other)]:\n"
        "    t.cuda.synchronize()\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {4}


def test_comprehension_short_circuit_walrus_keeps_possible_outer_binding() -> None:
    source = (
        "import torch\n"
        "values = [torch.cuda for item in objects "
        "if condition or (torch := None)]\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {2}


def test_lambda_default_is_scanned_in_enclosing_scope() -> None:
    source = "import torch\nf = lambda torch=torch.cuda.Stream: torch.cuda\n"
    assert direct_device_lines(source, {"cuda"}) == {2}


def test_alias_backend_keeps_import_and_usage_diagnostics() -> None:
    source = "from torch import cuda as backend\nalias = backend\nalias.synchronize()\n"
    assert direct_device_lines(source, {"cuda"}) == {1, 2, 3}


def test_walrus_in_comprehension_is_a_function_local_binding() -> None:
    source = (
        "import torch\n"
        "def check():\n"
        "    torch.cuda\n"
        "    values = [(torch := None) for item in objects]\n"
    )
    assert direct_device_lines(source, {"cuda"}) == set()


@pytest.mark.parametrize(
    "source",
    [
        "t = torch\nt.cuda.synchronize()",
        "modules = [torch]\nmodules[0].cuda.synchronize()",
        "for t in [torch]:\n    t.cuda.synchronize()",
        "[t] = [torch]\nt.cuda.synchronize()",
        "[t.cuda.synchronize() for t in [torch]]",
        "f = lambda t=torch: t.cuda.synchronize()",
    ],
)
def test_main_rejects_alias_and_list_access(
    source: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    path = tmp_path / "test_device_alias.py"
    path.write_text(f"import torch\n{source}\n", encoding="utf-8")
    assert main([str(path)]) == 1
    assert "BANNED: Direct torch device API usage" in capsys.readouterr().out


@pytest.mark.parametrize(
    "source, lines",
    [
        (
            "import torch\n"
            "def run(torch=torch):\n"
            "    torch.cuda.synchronize()\n"
            "run()\n",
            {3},
        ),
        (
            "import torch\n"
            "def outer():\n"
            "    torch = None\n"
            "    def run():\n"
            "        global torch\n"
            "        torch.cuda.synchronize()\n"
            "    run()\n"
            "outer()\n",
            {6},
        ),
        (
            "import torch as original\n"
            "torch = None\n"
            "for i in range(2):\n"
            "    if torch is not None:\n"
            "        torch.cuda.synchronize()\n"
            "    torch = original\n",
            {5},
        ),
        (
            "torch = None\n"
            "for _ in [0]:\n"
            "    import torch\n"
            "    break\n"
            "    torch = None\n"
            "torch.cuda.synchronize()\n",
            {6},
        ),
        (
            "import torch as original\n"
            "torch = None\n"
            "match original:\n"
            "    case torch:\n"
            "        torch.cuda.synchronize()\n",
            {5},
        ),
        (
            "import torch\n"
            "class C:\n"
            "    torch = None\n"
            "    del torch\n"
            "    torch.cuda.synchronize()\n",
            {5},
        ),
    ],
    ids=["default", "global", "back-edge", "break", "match", "delete"],
)
def test_upstream_direct_reference_regressions(
    source: str,
    lines: set[int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert direct_device_lines(source, {"cuda"}) == lines
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    path = tmp_path / "test_upstream_regression.py"
    path.write_text(source, encoding="utf-8")
    assert main([str(path)]) == 1
    output = capsys.readouterr().out
    for line in lines:
        assert f"{path}:{line}:" in output


@pytest.mark.parametrize(
    "signature",
    ["torch=torch", "torch=torch, /", "*, torch=torch"],
)
@pytest.mark.parametrize("definition", ["def", "async def"])
def test_function_defaults_preserve_torch_alias(
    signature: str, definition: str
) -> None:
    source = f"import torch\n{definition} run({signature}):\n    torch.cuda\n"
    assert direct_device_lines(source, {"cuda"}) == {3}


def test_global_declaration_applies_to_nested_closure() -> None:
    source = (
        "import torch\n"
        "def outer():\n"
        "    torch = None\n"
        "    def middle():\n"
        "        global torch\n"
        "        def inner():\n"
        "            torch.cuda\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {7}


def test_nonlocal_skips_class_binding() -> None:
    source = (
        "def outer():\n"
        "    import torch\n"
        "    class C:\n"
        "        torch = None\n"
        "        def run(self):\n"
        "            nonlocal torch\n"
        "            torch.cuda\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {7}


@pytest.mark.parametrize("declaration", ["global torch", "nonlocal torch"])
def test_declared_binding_is_updated_within_function(declaration: str) -> None:
    source = (
        "import torch as original\n"
        "torch = None\n"
        "def outer():\n"
        "    torch = None\n"
        "    def run():\n"
        f"        {declaration}\n"
        "        torch = original\n"
        "        torch.cuda\n"
        "    torch.cuda\n"
        "torch.cuda\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {8}


@pytest.mark.parametrize("loop", ["for _ in items:", "while condition:"])
def test_loop_propagates_alias_through_multiple_iterations(loop: str) -> None:
    source = (
        "import torch\n"
        "a = b = c = None\n"
        f"{loop}\n"
        "    a.cuda\n"
        "    a = b\n"
        "    b = c\n"
        "    c = torch\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {4}


def test_continue_keeps_binding_for_next_iteration() -> None:
    source = (
        "import torch as original\n"
        "torch = None\n"
        "while condition:\n"
        "    torch.cuda\n"
        "    torch = original\n"
        "    continue\n"
        "    torch = None\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {4}


def test_conditional_break_preserves_exit_binding() -> None:
    source = (
        "import torch as original\n"
        "torch = None\n"
        "for _ in items:\n"
        "    if condition:\n"
        "        torch = original\n"
        "        break\n"
        "    torch = None\n"
        "torch.cuda\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {8}


def test_break_bypasses_loop_else() -> None:
    source = (
        "import torch as original\n"
        "torch = None\n"
        "while condition:\n"
        "    torch = original\n"
        "    break\n"
        "else:\n"
        "    torch = None\n"
        "torch.cuda\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {8}


def test_inner_loop_break_does_not_exit_outer_body() -> None:
    source = (
        "import torch\n"
        "for _ in items:\n"
        "    for _ in items:\n"
        "        break\n"
        "    torch = None\n"
        "    torch.cuda\n"
    )
    assert direct_device_lines(source, {"cuda"}) == set()


def test_function_local_delete_does_not_fall_back_to_global() -> None:
    source = "import torch\ndef run():\n    del torch\n    torch.cuda\n"
    assert direct_device_lines(source, {"cuda"}) == set()


def test_conditional_class_delete_preserves_global_fallback() -> None:
    source = (
        "import torch\n"
        "class C:\n"
        "    torch = None\n"
        "    if condition:\n"
        "        del torch\n"
        "    torch.cuda\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {6}


@pytest.mark.parametrize(
    "subject, pattern",
    [("original", "torch"), ("original", "_ as torch"), ("[original]", "[torch]")],
)
def test_match_capture_propagates_subject(subject: str, pattern: str) -> None:
    source = (
        "import torch as original\n"
        "torch = None\n"
        f"match {subject}:\n"
        f"    case {pattern} if torch.cuda.is_available():\n"
        "        torch.cuda\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {4, 5}


def test_match_capture_shadows_global_in_function() -> None:
    source = (
        "import torch\n"
        "def run(value):\n"
        "    torch.cuda\n"
        "    match value:\n"
        "        case torch:\n"
        "            torch.cuda\n"
    )
    assert direct_device_lines(source, {"cuda"}) == set()


@pytest.mark.parametrize("control", ["break", "continue"])
def test_loop_exit_runs_finally_before_propagating_binding(control: str) -> None:
    source = (
        "import torch as original\n"
        "torch = None\n"
        "for _ in items:\n"
        "    try:\n"
        f"        {control}\n"
        "        torch = None\n"
        "    finally:\n"
        "        torch = original\n"
        "torch.cuda\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {9}


def test_finally_can_clear_break_binding() -> None:
    source = (
        "import torch\n"
        "for _ in items:\n"
        "    try:\n"
        "        break\n"
        "    finally:\n"
        "        torch = None\n"
        "else:\n"
        "    torch = None\n"
        "torch.cuda\n"
    )
    assert direct_device_lines(source, {"cuda"}) == set()


@pytest.mark.parametrize("control", ["return", "raise RuntimeError()"])
def test_function_exit_does_not_visit_unreachable_assignment(control: str) -> None:
    source = (
        "def run():\n"
        "    import torch\n"
        f"    {control}\n"
        "    torch.cuda\n"
        "import torch\n"
        "torch.cuda\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {6}


def test_loop_container_growth_reaches_finite_summary() -> None:
    source = (
        "import torch\n"
        "modules = [torch]\n"
        "for _ in items:\n"
        "    modules = [modules]\n"
        "    modules[0][0].cuda\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {5}


def test_global_rebinding_is_visible_to_nested_closure_only() -> None:
    source = (
        "import torch as original\n"
        "torch = None\n"
        "def outer():\n"
        "    torch = None\n"
        "    def run():\n"
        "        global torch\n"
        "        torch = original\n"
        "        def inner():\n"
        "            torch.cuda\n"
        "torch.cuda\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {9}


def test_function_default_rebinding_still_shadows_alias() -> None:
    source = "import torch\ndef run(torch=torch):\n    torch = None\n    torch.cuda\n"
    assert direct_device_lines(source, {"cuda"}) == set()


def test_match_alternatives_preserve_captured_alias() -> None:
    source = (
        "import torch as original\n"
        "match [original]:\n"
        "    case [torch] | (torch,):\n"
        "        torch.cuda\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {4}


def test_nested_class_delete_falls_back_to_module_not_outer_class() -> None:
    source = (
        "import torch\n"
        "class Outer:\n"
        "    torch = None\n"
        "    class Inner:\n"
        "        torch = None\n"
        "        del torch\n"
        "        torch.cuda\n"
    )
    assert direct_device_lines(source, {"cuda"}) == {7}
