# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the direct torch device pre-commit check."""

# First Party
from tools.check_direct_torch_device import direct_device_lines


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
