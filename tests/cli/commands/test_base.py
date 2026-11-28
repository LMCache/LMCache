# SPDX-License-Identifier: Apache-2.0
"""Tests for the CLI helpers in ``lmcache.cli.commands.base``."""

# Standard
from pathlib import Path
from unittest.mock import patch

# Third Party
import pytest

# First Party
from lmcache.cli.commands import base

_TRANSPORT = "lmcache.v1.transport.grpc._proto_gen"
_ORCHESTRATOR = "lmcache.v1.memory_orchestrator._proto_gen"


def _cannot_import_name() -> ImportError:
    return ImportError(
        f"cannot import name 'p2p_service_pb2' from '{_TRANSPORT}'",
        name=_TRANSPORT,
        path=f"{_TRANSPORT}/__init__.py",
    )


@pytest.mark.parametrize(
    "error",
    [
        _cannot_import_name(),
        ModuleNotFoundError(
            f"No module named '{_ORCHESTRATOR}.memory_orchestrator_pb2_grpc'",
            name=f"{_ORCHESTRATOR}.memory_orchestrator_pb2_grpc",
        ),
    ],
    ids=["cannot-import-name", "no-module-named"],
)
def test_missing_binding_imports_get_the_generator(error: ImportError) -> None:
    hint = base.missing_grpc_bindings_hint(error)
    assert hint is not None
    assert "_generate.py" in hint


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError(
            "import of lmcache.v1.distributed.config halted; None in sys.modules",
            name="lmcache.v1.distributed.config",
        ),
        ModuleNotFoundError("No module named", name=f"{_ORCHESTRATOR}._generate"),
        ImportError(
            "cannot import name 'config' from 'lmcache.v1.distributed'",
            name="lmcache.v1.distributed",
            path="lmcache/v1/distributed/__init__.py",
        ),
        ImportError("bare"),
    ],
    ids=["other-module", "not-a-binding", "other-package", "no-name"],
)
def test_other_import_errors_get_no_hint(error: ImportError) -> None:
    assert base.missing_grpc_bindings_hint(error) is None


def test_no_hint_outside_a_source_checkout(tmp_path: Path) -> None:
    installed = tmp_path / "site-packages" / "lmcache" / "cli" / "commands" / "base.py"
    with patch.object(base, "__file__", str(installed)):
        assert base.missing_grpc_bindings_hint(_cannot_import_name()) is None
