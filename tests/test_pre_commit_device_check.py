# SPDX-License-Identifier: Apache-2.0
"""Core lint regressions and explicit boundaries of the bounded alias analysis."""

# Standard
from collections.abc import Callable
from itertools import product
from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib.util
import subprocess
import sys

# Third Party
import pytest

_CHECK_PATH = Path(__file__).resolve().parents[1] / "tools/check_direct_torch_device.py"
_spec = importlib.util.spec_from_file_location(
    "_lmcache_precommit_device_check", _CHECK_PATH
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load device check from {_CHECK_PATH}")
_device_check = importlib.util.module_from_spec(_spec)
# Dataclass annotation resolution needs the module registered during execution.
sys.modules[_spec.name] = _device_check
_spec.loader.exec_module(_device_check)
direct_device_lines: Callable[[str, set[str]], set[int]] = (
    _device_check.direct_device_lines
)
main: Callable[[list[str]], int] = _device_check.main


@pytest.fixture
def device_check_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide an isolated CLI working directory with a known CUDA backend."""
    root = tmp_path / "device-check"
    (root / "lmcache/v1/platform/devices/cuda").mkdir(parents=True)
    monkeypatch.chdir(root)
    return root


@pytest.mark.parametrize(
    "source, lines",
    [
        ('# torch.cuda\n"""torch.xpu"""\ntarget = "torch.cuda.Stream"', set()),
        ("torch.cuda.synchronize()\nvalue = torch.xpu.current_device()", {1, 2}),
        (
            "import torch as pytorch\npytorch.cuda.synchronize()\n"
            "from torch import xpu as accelerator\naccelerator.synchronize()",
            {2, 3, 4},
        ),
        ("import torch.cuda as backend\nbackend.synchronize()", {1}),
        ("from torch.cuda import Stream\nStream()", {1}),
        ("import torch\ngetattr(torch, 'cuda').synchronize()", {2}),
        ("import torch\nt = getattr(torch, 'cuda')\nt.synchronize()", {2, 3}),
        ("import torch\ngetattr = custom\ngetattr(torch, 'cuda')", set()),
        ("import torch\ndef f(getattr):\n    getattr(torch, 'cuda')", set()),
        (
            "import torch\nclass C:\n    getattr = custom\n"
            "    def f(self):\n        getattr(torch, 'cuda')",
            {5},
        ),
        (
            "import torch\nclass C:\n    getattr = custom\n"
            "    value = getattr(torch, 'cuda')",
            set(),
        ),
        (
            "import torch\ntorch = object()\ntorch.cuda = 1\n"
            "def check(torch):\n    return torch.xpu",
            set(),
        ),
        ("import torch\ndef f():\n    torch.cuda\n    torch = None", set()),
        (
            "from torch import cuda as backend\nalias = backend\nalias.synchronize()",
            {1, 2, 3},
        ),
    ],
)
def test_direct_references_and_imports(source: str, lines: set[int]) -> None:
    assert direct_device_lines(source, {"cuda", "xpu"}) == lines


@pytest.mark.parametrize(
    "statement",
    [
        "stream: torch.cuda.Stream",
        "stream: torch.cuda.Stream = None",
        "def f(x: torch.cuda.Stream): pass",
        "async def f(x: torch.cuda.Stream): pass",
        "def f(x: torch.cuda.Stream, /): pass",
        "def f(*, x: torch.cuda.Stream): pass",
        "def f(*x: torch.cuda.Stream): pass",
        "def f(**x: torch.cuda.Stream): pass",
        "def f() -> torch.cuda.Event: pass",
        "def f(torch: torch.cuda.Stream): pass",
        "def f(x=torch.cuda.Stream): pass",
        "@torch.cuda.decorator\ndef f(): pass",
        "class C(torch.cuda.Stream): pass",
        "class C(metaclass=torch.cuda.Stream): pass",
        "class C(backend=torch.cuda.synchronize()): pass",
    ],
)
def test_definition_fields_are_scanned(statement: str) -> None:
    assert direct_device_lines(f"import torch\n{statement}", {"cuda"}) == {2}


@pytest.mark.skipif(sys.version_info < (3, 12), reason="requires type parameters")
@pytest.mark.parametrize(
    "definition",
    ["def f[T: torch.cuda.Stream](): pass", "class C[T: torch.cuda.Stream]: pass"],
)
def test_type_parameters_are_scanned(definition: str) -> None:
    assert direct_device_lines(f"import torch\n{definition}", {"cuda"}) == {2}


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
def test_conditional_binding_keeps_possible_torch(statement: str) -> None:
    source = f"import torch\n{statement}\ntorch.cuda.synchronize()"
    assert direct_device_lines(source, {"cuda"}) == {len(source.splitlines())}


@pytest.mark.parametrize(
    "statement",
    [
        "torch = None",
        "torch: object = None",
        "if condition:\n    torch = None\nelse:\n    torch = object()",
        "try:\n    may_fail()\nfinally:\n    torch = None",
    ],
)
def test_definite_rebinding_shadows_torch(statement: str) -> None:
    assert (
        direct_device_lines(f"import torch\n{statement}\ntorch.cuda", {"cuda"}) == set()
    )


@pytest.mark.parametrize(
    "source, lines",
    [
        ("import torch\nif flag:\n    torch = None\nelse:\n    torch.cuda", {5}),
        ("if flag:\n    import torch as t\nelse:\n    t = None\nt.cuda", {5}),
        ("for item in items:\n    import torch as t\nt.cuda", {3}),
        (
            "if flag:\n    import torch as t\nelse:\n    from torch import cuda as t\n"
            "t.cuda\nt.synchronize()",
            {4, 5, 6},
        ),
        (
            "import torch\ntry:\n    may_fail()\n    torch = None\n"
            "except Exception:\n    torch.cuda",
            {6},
        ),
        (
            "import torch\ntry:\n    may_fail()\n    torch = None\n"
            "finally:\n    torch.cuda\ntorch.cuda",
            {6},
        ),
        (
            "async def f():\n    import torch\n    async for torch in items:\n"
            "        pass\n    torch.cuda",
            {5},
        ),
    ],
)
def test_branch_and_exception_bindings(source: str, lines: set[int]) -> None:
    assert direct_device_lines(source, {"cuda"}) == lines


@pytest.mark.skipif(sys.version_info < (3, 11), reason="requires exception groups")
def test_exception_groups() -> None:
    source = (
        "import torch\ntry:\n    may_fail()\n    torch = None\n"
        "except* Exception:\n    pass\ntorch.cuda"
    )
    assert direct_device_lines(source, {"cuda"}) == {7}


@pytest.mark.parametrize(
    "setup, expression",
    [
        ("t = torch", "t.cuda"),
        ("a = torch\nt = a", "t.xpu"),
        ("a = t = torch", "t.cuda"),
        ("t: object = torch", "t.cuda"),
        ("", "(t := torch).cuda"),
        ("(t := torch)", "t.cuda"),
        ("t = torch if flag else None", "t.cuda"),
        ("t = None\nif flag:\n    t = torch", "t.cuda"),
        ("t = torch\ntorch = None", "t.cuda"),
        ("", "[torch.cuda.Stream()]"),
        ("modules = [torch]", "modules[0].cuda"),
        ("modules = [None, torch]", "modules[-1].cuda"),
        ("modules = [None, torch]", "modules[index].cuda"),
        ("modules = [[torch]]", "modules[0][0].cuda"),
        ("modules = [torch]\nother = modules", "other[0].cuda"),
        ("modules = [torch]", 'getattr(modules[0], "cuda")'),
        ("modules = [None, torch]\nexpanded = [*modules]", "expanded[1].cuda"),
        ("[t] = [torch]", "t.cuda"),
        ("_, t = None, torch", "t.cuda"),
        ("[_, [t]] = [None, [torch]]", "t.cuda"),
        ("*rest, t = [None, torch]", "t.cuda"),
        ("_, *rest = [None, torch]", "rest[0].cuda"),
        ("t = None\nfor t in [torch]:\n    pass", "t.cuda"),
        ("", "[t.cuda for t in [torch]]"),
        ("", "{t.cuda for t in (torch,)}"),
        ("", "(t.cuda for t in [torch])"),
        ("", "{t.cuda: None for t in [torch]}"),
        ("", "{None: t.cuda for t in [torch]}"),
        ("", "[None for t in [torch] if t.cuda]"),
        ("", "[t.cuda for group in [[torch]] for t in group]"),
        ("", "[t.cuda for _, t in [(None, torch)]]"),
        ("modules = [t for t in [torch]]", "modules[0].cuda"),
        ("", "lambda: torch.cuda"),
        ("", "lambda t=torch: t.cuda"),
        ("", "lambda *, t=torch: t.cuda"),
        ("", "lambda: (t := torch).cuda"),
    ],
)
def test_simple_aliases_and_literal_sequences(setup: str, expression: str) -> None:
    source = f"import torch\n{setup}\n{expression}"
    assert direct_device_lines(source, {"cuda", "xpu"}) == {len(source.splitlines())}


@pytest.mark.parametrize(
    "source",
    [
        "t = torch\nt = object()\nt.cuda",
        "modules = [None, torch]\nmodules[0].cuda",
        "modules = [torch, None]\nmodules[-1].cuda",
        "modules = [None, torch]\nexpanded = [*modules]\nexpanded[0].cuda",
        "modules = [torch]\nmodules[9].cuda",
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
def test_shadowing_and_sequence_positions(source: str) -> None:
    assert direct_device_lines(f"import torch\n{source}", {"cuda", "xpu"}) == set()


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
def test_expression_scopes_do_not_leak(expression: str) -> None:
    assert direct_device_lines(
        f"import torch\nvalue = {expression}\ntorch.cuda", {"cuda"}
    ) == {3}


@pytest.mark.parametrize(
    "source, lines",
    [
        (
            "import torch\nclass C:\n    torch = None\n"
            "    def f(self):\n        torch.cuda",
            {5},
        ),
        (
            "import torch\nclass C:\n    torch = None\n"
            "    async def f(self):\n        torch.cuda",
            {5},
        ),
        (
            "import torch\nclass Outer:\n    torch = None\n"
            "    class Inner:\n        x = torch.cuda",
            {5},
        ),
        (
            "def outer():\n    import torch as t\n    class C:\n        t = None\n"
            "        def f(self):\n            t.cuda",
            {6},
        ),
        (
            "import torch\nclass C:\n    torch = other\n    x = torch.cuda\n"
            "    def f(self, x: torch.cuda = torch.cuda) -> torch.cuda:\n        pass",
            set(),
        ),
        ("import torch\nclass C:\n    torch: object\n    x = torch.cuda", {4}),
        (
            "import torch\nclass C:\n"
            "    def f(self, torch):\n        return torch.cuda\n"
            "    def g(self):\n        torch.cuda\n        torch = None",
            set(),
        ),
        ("import torch\nvalues = [torch for torch in torch.cuda.devices]", {2}),
        (
            "import torch\nclass C:\n    torch = None\n    f = lambda: torch.cuda\n"
            "    xs = [torch.cuda for item in objects]",
            {4, 5},
        ),
        (
            "class C:\n    import torch as local\n    xs = [t.cuda for t in [local]]",
            {3},
        ),
        (
            "import torch\nxs = [(torch.cuda, [torch.cuda for torch in objects]) "
            "for torch in [torch]]",
            {2},
        ),
        (
            "def f():\n    import torch\n"
            "    xs = [(t := torch) for item in objects]\n    t.cuda",
            {4},
        ),
        (
            "import torch\n"
            "xs = [torch.cuda for item in objects if flag or (torch := None)]",
            {2},
        ),
        ("import torch\nf = lambda torch=torch.cuda.Stream: torch.cuda", {2}),
        (
            "import torch\ndef f():\n    torch.cuda\n"
            "    xs = [(torch := None) for item in objects]",
            set(),
        ),
        (
            "import torch\ntorch, other = None, torch\n"
            "for _, t in [(None, other)]:\n    t.cuda",
            {4},
        ),
    ],
)
def test_lexical_scopes(source: str, lines: set[int]) -> None:
    assert direct_device_lines(source, {"cuda"}) == lines


@pytest.mark.parametrize(
    "source, lines",
    [
        ("import torch\ndef run(torch=torch):\n    torch.cuda\nrun()", {3}),
        (
            "import torch\ndef outer():\n    torch = None\n    def run():\n"
            "        global torch\n        torch.cuda\n    run()\nouter()",
            {6},
        ),
        (
            "import torch as original\ntorch = None\nfor i in range(2):\n"
            "    if torch is not None:\n        torch.cuda\n    torch = original",
            {5},
        ),
        (
            "torch = None\nfor _ in [0]:\n    import torch\n"
            "    break\n    torch = None\ntorch.cuda",
            {6},
        ),
        (
            "import torch as original\ntorch = None\n"
            "match original:\n    case torch:\n        torch.cuda",
            {5},
        ),
        (
            "import torch\nclass C:\n    torch = None\n    del torch\n    torch.cuda",
            {5},
        ),
    ],
    ids=["default", "global", "back-edge", "break", "match", "delete"],
)
@pytest.mark.usefixtures("device_check_root")
def test_upstream_direct_reference_regressions(
    source: str, lines: set[int], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert direct_device_lines(source, {"cuda"}) == lines
    path = tmp_path / "upstream_regression.py"
    path.write_text(source, encoding="utf-8")
    assert main([str(path)]) == 1
    output = capsys.readouterr().out
    for line in lines:
        assert f"{path}:{line}:" in output


@pytest.mark.parametrize(
    "signature", ["torch=torch", "torch=torch, /", "*, torch=torch"]
)
@pytest.mark.parametrize("definition", ["def", "async def"])
def test_function_defaults(signature: str, definition: str) -> None:
    assert direct_device_lines(
        f"import torch\n{definition} f({signature}):\n    torch.cuda", {"cuda"}
    ) == {3}


@pytest.mark.parametrize(
    "source, lines",
    [
        (
            "import torch\ndef outer():\n    torch = None\n    def middle():\n"
            "        global torch\n        def inner():\n            torch.cuda",
            {7},
        ),
        (
            "def outer():\n    import torch\n    class C:\n        torch = None\n"
            "        def f(self):\n            nonlocal torch\n            torch.cuda",
            {7},
        ),
        ("import torch\ndef f():\n    del torch\n    torch.cuda", set()),
        (
            "import torch\nclass C:\n    torch = None\n"
            "    if flag:\n        del torch\n    torch.cuda",
            {6},
        ),
        (
            "import torch\nclass Outer:\n    torch = None\n    class Inner:\n"
            "        torch = None\n        del torch\n        torch.cuda",
            {7},
        ),
        (
            "import torch as original\ntorch = None\nmatch original:\n"
            "    case _ as torch if torch.cuda:\n        torch.cuda",
            {4, 5},
        ),
        (
            "import torch\ndef f(value):\n    torch.cuda\n"
            "    match value:\n        case torch:\n            torch.cuda",
            set(),
        ),
        ("import torch\ndef f(torch=torch):\n    torch = None\n    torch.cuda", set()),
        (
            "import torch as original\ntorch = None\ndef outer():\n    torch = None\n"
            "    def f():\n        global torch\n        torch = original\n"
            "        def inner():\n            torch.cuda\ntorch.cuda",
            {9},
        ),
    ],
)
def test_declarations_and_captures(source: str, lines: set[int]) -> None:
    assert direct_device_lines(source, {"cuda"}) == lines


@pytest.mark.parametrize("declaration", ["global torch", "nonlocal torch"])
def test_function_analysis_does_not_mutate_enclosing_scope(declaration: str) -> None:
    source = (
        "import torch as original\ntorch = None\ndef outer():\n    torch = None\n"
        f"    def f():\n        {declaration}\n"
        "        torch = original\n        torch.cuda\n"
        "    torch.cuda\ntorch.cuda"
    )
    assert direct_device_lines(source, {"cuda"}) == {8}


@pytest.mark.parametrize(
    "source, lines",
    [
        (
            "import torch\na = b = c = None\nwhile flag:\n"
            "    a.cuda\n    a = b\n    b = c\n    c = torch",
            {4},
        ),
        (
            "import torch\na = b = c = None\nfor _ in items:\n"
            "    a.cuda\n    a = b\n    b = c\n    c = torch",
            {4},
        ),
        (
            "import torch as original\ntorch = None\nwhile flag:\n"
            "    torch.cuda\n    torch = original\n    continue\n    torch = None",
            {4},
        ),
        (
            "import torch as original\ntorch = None\nfor _ in items:\n"
            "    if flag:\n        torch = original\n        break\n"
            "    torch = None\ntorch.cuda",
            {8},
        ),
        (
            "import torch as original\ntorch = None\nwhile flag:\n"
            "    torch = original\n    break\nelse:\n    torch = None\ntorch.cuda",
            {8},
        ),
        (
            "import torch\nfor _ in items:\n    for _ in items:\n"
            "        break\n    torch = None\n    torch.cuda",
            set(),
        ),
        (
            "import torch\nfor _ in items:\n    try:\n        break\n"
            "    finally:\n        torch = None\nelse:\n    torch = None\ntorch.cuda",
            set(),
        ),
    ],
)
def test_loop_control_flow(source: str, lines: set[int]) -> None:
    assert direct_device_lines(source, {"cuda"}) == lines


@pytest.mark.parametrize("control", ["break", "continue"])
def test_loop_exit_runs_finally(control: str) -> None:
    source = (
        "import torch as original\ntorch = None\nfor _ in items:\n    try:\n"
        f"        {control}\n        torch = None\n"
        "    finally:\n        torch = original\ntorch.cuda"
    )
    assert direct_device_lines(source, {"cuda"}) == {9}


@pytest.mark.parametrize("control", ["return", "raise RuntimeError()"])
def test_function_exit_skips_unreachable_code(control: str) -> None:
    source = (
        f"def f():\n    import torch\n    {control}\n"
        "    torch.cuda\nimport torch\ntorch.cuda"
    )
    assert direct_device_lines(source, {"cuda"}) == {6}


@pytest.mark.parametrize(
    "source, lines",
    [
        (
            "import torch\n\nfn = lambda: (torch := None)\ntorch.cuda.synchronize()\n",
            {4},
        ),
        (
            'import torch\n\nclass Item:\n    cuda = "not PyTorch"\n\n'
            "fn = lambda torch: torch.cuda\nvalue = fn(Item())\n",
            set(),
        ),
        (
            'import torch\n\nclass Item:\n    cuda = "not PyTorch"\n\n'
            "objects = [Item()]\nvalues = [torch.cuda for torch in objects]\n",
            set(),
        ),
        (
            "torch = None\n\ndef run():\n    torch.cuda.synchronize()\n"
            "\nimport torch\n\nrun()\n",
            {4},
        ),
    ],
    ids=["lambda-walrus", "lambda-shadow", "comprehension-shadow", "late-global"],
)
@pytest.mark.usefixtures("device_check_root")
def test_reviewer_examples(
    source: str, lines: set[int], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert direct_device_lines(source, {"cuda"}) == lines
    path = tmp_path / "reviewer.py"
    path.write_text(source, encoding="utf-8")
    assert main([str(path)]) == int(bool(lines))
    output = capsys.readouterr().out
    assert bool(output) == bool(lines)


@pytest.mark.parametrize(
    "source, lines",
    [
        ("t = None\ndef f():\n    t.cuda\nimport torch as t", {3}),
        ("import torch\nt = None\ndef f():\n    t.cuda\nt = torch", {4}),
        (
            "import torch as original\ntorch = None\n"
            "def f():\n    torch.cuda\nf()\ntorch = original",
            {4},
        ),
        ("import torch\ntorch = None\ndef f():\n    torch.cuda", {4}),
        (
            "import torch\nt = None\ndef f(t=t):\n    t.cuda\nt = torch",
            set(),
        ),
        (
            "import torch\ndef f():\n    global torch\n"
            "    torch = None\n    torch.cuda",
            set(),
        ),
    ],
)
def test_deferred_globals_are_conservative_not_call_sensitive(
    source: str, lines: set[int]
) -> None:
    assert direct_device_lines(source, {"cuda"}) == lines


@pytest.mark.parametrize(
    "source",
    [
        "def f(t):\n    t.cuda\nf(torch)",
        "def f(t):\n    t.cuda\nf(*[torch])",
        "def f(t):\n    t.cuda\nf(**{'t': torch})",
        "def f():\n    return torch\nf().cuda",
        "def outer(t):\n    return lambda: t.cuda\nouter(torch)()",
        "modules = [None]\nmodules[0] = torch\nmodules[0].cuda",
        "modules = []\nalias = modules\nalias.append(torch)\nmodules[0].cuda",
        "modules = []\nmodules.extend([torch])\nmodules[0].cuda",
        "modules = [None]\nmodules[:] = [torch]\nmodules[0].cuda",
        "modules = {'backend': torch}\nmodules['backend'].cuda",
        "modules = [] + [torch]\nmodules[0].cuda",
        "modules = [torch] * 2\nmodules[1].cuda",
        "modules = []\nmodules += [torch]\nmodules[0].cuda",
        "modules = [torch][:]\nmodules[0].cuda",
        "for _, t in enumerate([torch]):\n    t.cuda",
        "for t, _ in zip([torch], [None]):\n    t.cuda",
        "for t in iter([torch]):\n    t.cuda",
        "for t in reversed([torch]):\n    t.cuda",
        "for t in {'backend': torch}.values():\n    t.cuda",
        "gen = (t for t in [torch])\nfor value in gen:\n    value.cuda",
        "match [torch]:\n    case [t]:\n        t.cuda",
    ],
)
def test_documented_indirect_flows_are_not_inferred(source: str) -> None:
    assert direct_device_lines(f"import torch\n{source}", {"cuda"}) == set()


@pytest.mark.parametrize(
    "source",
    [
        "modules = [torch]\nmodules[0] = None\nmodules[0].cuda",
        "modules = [torch]\nalias = modules\nalias.clear()\nmodules[0].cuda",
        "def clear():\n    global torch\n    torch = None\nclear()\ntorch.cuda",
    ],
)
def test_ignored_mutations_can_leave_conservative_matches(source: str) -> None:
    result = direct_device_lines(f"import torch\n{source}", {"cuda"})
    assert result == {len(source.splitlines()) + 1}


@pytest.mark.parametrize(
    "initial, before, after",
    list(product(["t = other", "t = torch"], ["other", "torch"], ["other", "torch"])),
)
def test_supported_alias_flows_match_runtime_branches(
    initial: str, before: str, after: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class TorchModule(ModuleType):
        cuda: SimpleNamespace

    observed: set[int] = set()

    def synchronize() -> None:
        observed.add(sys._getframe(1).f_lineno)

    fake = TorchModule("torch")
    fake.cuda = SimpleNamespace(synchronize=synchronize)
    other = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setitem(sys.modules, "torch", fake)
    source = (
        f"import torch\n{initial}\nif flag:\n    t = {before}\nelse:\n    t = {after}\n"
        "t.cuda.synchronize()\n"
        "modules = [other, torch]\nfor t in modules:\n    t.cuda.synchronize()\n"
        "[t.cuda.synchronize() for t in [torch]]\n"
    )
    for flag in (False, True):
        exec(compile(source, "<bounded-flow>", "exec"), {"other": other, "flag": flag})
    assert direct_device_lines(source, {"cuda"}) == observed


def test_recursive_calls_and_container_growth_finish_within_timeout(
    tmp_path: Path, device_check_root: Path
) -> None:
    source = (
        "import torch\nmodules = [torch]\nwhile flag:\n    modules = [modules]\n"
        "def f():\n    f()\n    torch.cuda\n"
    )
    for index in range(18):
        previous = "f" if index == 0 else f"f{index - 1}"
        source += f"def f{index}():\n    {previous}()\n    {previous}()\n"
    source += "f17()\n"
    path = tmp_path / "recursive.py"
    path.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(_CHECK_PATH), str(path)],
        text=True,
        capture_output=True,
        timeout=5,
        cwd=device_check_root,
    )
    assert result.returncode == 1, result.stderr
    assert result.stdout.splitlines() == [
        "BANNED: Direct torch device API usage",
        "Use lmcache.v1.platform abstractions instead",
        f"{path}:7:     torch.cuda",
    ]
    assert result.stderr == ""


@pytest.mark.parametrize("preloaded", [False, True])
def test_scanner_import_ignores_foreign_tools_package(
    preloaded: bool, tmp_path: Path
) -> None:
    package = tmp_path / "tools"
    package.mkdir()
    (package / "__init__.py").write_text("FOREIGN_TOOLS = True\n", encoding="utf-8")
    script = (
        "import runpy, sys\n"
        "if sys.argv[2] == 'True':\n"
        "    import tools\n"
        "    assert tools.FOREIGN_TOOLS\n"
        "original = sys.modules.get('tools')\n"
        "tests = runpy.run_path(sys.argv[1])\n"
        "assert tests['direct_device_lines']('torch.cuda', {'cuda'}) == {1}\n"
        "assert sys.modules.get('tools') is original\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve()), str(preloaded)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.usefixtures("device_check_root")
def test_cli_reports_syntax_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "invalid.py"
    path.write_text("def invalid(:\n", encoding="utf-8")
    assert main([str(path)]) == 1
    output = capsys.readouterr().out
    assert f"{path}:1:" in output
    assert "BANNED" not in output


def test_cli_discovers_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lmcache/v1/platform/devices/custom").mkdir(parents=True)
    (tmp_path / "lmcache/v1/platform/devices/__pycache__").mkdir()
    path = tmp_path / "source.py"
    path.write_text("import torch\ntorch.custom.synchronize()\n", encoding="utf-8")
    assert main([str(path)]) == 1
    assert f"{path}:2:" in capsys.readouterr().out


def test_cli_warns_when_no_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lmcache/v1/platform/devices").mkdir(parents=True)
    assert main([]) == 0
    assert "WARNING: no device dirs found" in capsys.readouterr().out


@pytest.mark.usefixtures("device_check_root")
def test_cli_read_errors_are_not_hidden(tmp_path: Path) -> None:
    path = tmp_path / "missing.py"
    with pytest.raises(FileNotFoundError) as error:
        main([str(path)])
    assert error.value.filename == str(path)
