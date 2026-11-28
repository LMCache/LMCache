# SPDX-License-Identifier: Apache-2.0

# Standard
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import patch
import argparse

# Third Party
import pytest

# First Party
from lmcache.cli.commands import memory
from lmcache.cli.commands.memory import MemoryCommand
from lmcache.v1.memory_orchestrator.server import OrchestratorConfig, parse_args

_FailingImport = Callable[[str, ImportError], AbstractContextManager[None]]


def _missing_orchestrator_binding() -> ModuleNotFoundError:
    return ModuleNotFoundError(
        "No module named 'lmcache.v1.memory_orchestrator._proto_gen."
        "memory_orchestrator_pb2_grpc'",
        name="lmcache.v1.memory_orchestrator._proto_gen.memory_orchestrator_pb2_grpc",
    )


def test_memory_command_reports_ungenerated_bindings_at_registration(
    failing_import: _FailingImport,
) -> None:
    """A build that skipped the gRPC generator names it, not a reinstall."""
    with (
        failing_import(
            "lmcache.v1.memory_orchestrator.server", _missing_orchestrator_binding()
        ),
        patch.object(memory.logger, "warning") as mock_warning,
    ):
        MemoryCommand().add_arguments(argparse.ArgumentParser())

    logged = " ".join(map(str, mock_warning.call_args.args))
    assert "_generate.py" in logged


def test_memory_command_explains_ungenerated_bindings(
    failing_import: _FailingImport, capsys: pytest.CaptureFixture[str]
) -> None:
    """A build that skipped the gRPC generator points at the generator."""
    command = MemoryCommand()
    command.register(argparse.ArgumentParser().add_subparsers())
    with (
        failing_import(
            "lmcache.v1.memory_orchestrator.server", _missing_orchestrator_binding()
        ),
        pytest.raises(SystemExit) as excinfo,
    ):
        command.execute(argparse.Namespace())

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "_generate.py" in err


def test_memory_command_serves_the_parsed_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    argv = ["--region-id", "r", "--capacity-bytes", str(1 << 20)]
    argv += ["--alignment", "4K", "--state-dir", str(tmp_path)]
    argv += ["--write-ttl-seconds", "12.5"]
    parser = argparse.ArgumentParser()
    command = MemoryCommand()
    command.register(parser.add_subparsers())
    args = parser.parse_args([command.name(), *argv])
    served: list[OrchestratorConfig] = []

    def fake_serve(config: OrchestratorConfig) -> int:
        served.append(config)
        return 2

    monkeypatch.setattr("lmcache.v1.memory_orchestrator.server.serve", fake_serve)
    with pytest.raises(SystemExit) as excinfo:
        args.func(args)
    assert excinfo.value.code == 2
    assert served == [parse_args(argv)]
    assert served[0].write_ttl_seconds == 12.5
