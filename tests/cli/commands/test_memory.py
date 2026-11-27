# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import argparse
import importlib.util
import sys

# Third Party
import pytest

# First Party
from lmcache.cli.commands import memory
from lmcache.cli.commands.memory import MemoryCommand
from lmcache.v1.memory_orchestrator import _proto_gen as orchestrator_proto_gen
from lmcache.v1.memory_orchestrator.server import OrchestratorConfig, parse_args
from lmcache.v1.transport.grpc import _proto_gen as transport_proto_gen


def test_memory_command_reports_ungenerated_bindings_at_registration(
    tmp_path: Path,
) -> None:
    """A build that skipped the gRPC generator names it, not a reinstall."""
    with (
        patch.dict(sys.modules, {"lmcache.v1.memory_orchestrator.server": None}),
        patch.object(orchestrator_proto_gen, "__path__", [str(tmp_path)]),
        patch.object(memory.logger, "warning") as mock_warning,
    ):
        MemoryCommand().add_arguments(argparse.ArgumentParser())

    logged = " ".join(map(str, mock_warning.call_args.args))
    assert "_generate.py" in logged


def test_memory_command_explains_ungenerated_bindings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A build that skipped the gRPC generator points at the generator."""
    command = MemoryCommand()
    command.register(argparse.ArgumentParser().add_subparsers())
    with (
        patch.dict(sys.modules, {"lmcache.v1.memory_orchestrator.server": None}),
        patch.object(orchestrator_proto_gen, "__path__", [str(tmp_path)]),
        pytest.raises(SystemExit) as excinfo,
    ):
        command.execute(argparse.Namespace())

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "_generate.py" in err


@pytest.mark.skipif(
    importlib.util.find_spec("lmcache.v1.transport.grpc._proto_gen.p2p_service_pb2")
    is None,
    reason="Requires the generated transport gRPC bindings",
)
@pytest.mark.parametrize(
    "package",
    [orchestrator_proto_gen, transport_proto_gen],
    ids=["orchestrator", "transport"],
)
def test_memory_command_names_the_generator_when_either_bindings_are_missing(
    package: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Hiding either package the orchestrator imports still points at the generator."""
    command = MemoryCommand()
    command.register(argparse.ArgumentParser().add_subparsers())
    with (
        patch.dict(sys.modules, {"lmcache.v1.memory_orchestrator.server": None}),
        patch.object(package, "__path__", [str(tmp_path)]),
        pytest.raises(SystemExit) as excinfo,
    ):
        command.execute(argparse.Namespace())

    assert excinfo.value.code == 1
    assert "_generate.py" in capsys.readouterr().err


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
