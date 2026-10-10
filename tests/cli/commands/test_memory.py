# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import argparse

# Third Party
import pytest

# First Party
from lmcache.cli.commands.memory import MemoryCommand
from lmcache.v1.memory_orchestrator.server import OrchestratorConfig, parse_args


def test_memory_command_serves_the_parsed_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    argv = ["--region-id", "r", "--capacity-bytes", str(1 << 20)]
    argv += ["--alignment", "4K", "--state-dir", str(tmp_path)]
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
