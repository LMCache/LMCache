# SPDX-License-Identifier: Apache-2.0
"""Subprocess tests for directed CLI command registration (contract:
docs/design/cli/framework-and-metrics.md)."""

# Standard
import json
import subprocess
import sys

_PROBE = """
import contextlib, io, json, sys

spec = json.loads(sys.argv[1])
sys.argv = ["lmcache"] + spec["ambient"]

import lmcache.cli.commands as pkg
import lmcache.cli.main as main_mod

wrappers_at_import = sorted(
    m for m in sys.modules
    if m.startswith("lmcache.cli.commands.") and m != "lmcache.cli.commands.base"
)
shim_at_import = "ALL_COMMANDS" in pkg.__dict__

out, err = io.StringIO(), io.StringIO()
exit_code = 0
try:
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        main_mod.main(spec["argv"]) if spec["explicit"] else main_mod.main()
except SystemExit as exc:
    exit_code = exc.code if isinstance(exc.code, int) else 1

mods = {name: name in sys.modules for name in spec["want"]}
ambient = list(sys.argv[1:])

from lmcache.cli.commands import ALL_COMMANDS
all_names = sorted(cmd.name() for cmd in ALL_COMMANDS)
shim_cached = ALL_COMMANDS is pkg.ALL_COMMANDS

print(json.dumps({
    "exit": exit_code,
    "text": out.getvalue() + err.getvalue(),
    "wrappers_at_import": wrappers_at_import,
    "shim_at_import": shim_at_import,
    "shim_cached": shim_cached,
    "all_names": all_names,
    "mods": mods,
    "ambient": ambient,
}))
"""


def _run(tokens, *, explicit=False, decoy=(), want=()) -> dict:
    """Run one probe subprocess: route through the real main(), parse the report."""
    spec = {
        "ambient": list(tokens if not explicit else decoy),
        "argv": list(tokens),
        "explicit": explicit,
        "want": list(want),
    }
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(spec)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.splitlines()[-1])


_SERVER_RUNTIME = "lmcache.v1.multiprocess.config"
_L2_WRAPPER = "lmcache.cli.commands.bench.l2_adapter_bench"
_L2_RUNTIME = "lmcache.cli.commands.bench.l2_adapter_bench.command"


def test_package_import_pulls_no_concrete_wrappers() -> None:
    """Importing commands/main is inert and ALL_COMMANDS is not computed at import."""
    res = _run([])
    assert res["wrappers_at_import"] == []
    assert not res["shim_at_import"]


def test_root_bare_and_unknown_list_every_choice() -> None:
    """Bare, -h, and invalid-choice paths keep the full top-level choice list."""
    expected = {"": 1, "-h": 0, "--help": 0, "nope": 2}
    for token, exit_code in expected.items():
        res = _run([token] if token else [])
        assert res["exit"] == exit_code, token
        assert res["all_names"], token
        for name in res["all_names"]:
            assert name in res["text"], (token, name)
        if token == "nope":
            assert "invalid choice" in res["text"]


def test_bench_help_routes_keep_choices_and_bound_registration() -> None:
    """A help token stops selection; only the selected leaf pays add_arguments."""
    for tokens, leaf_expected in (
        (["bench", "-h"], False),
        (["bench", "--help", "l2"], False),
        (["bench", "l2", "-h"], True),
    ):
        res = _run(tokens, want=(_L2_WRAPPER, _L2_RUNTIME, _SERVER_RUNTIME))
        assert res["exit"] == 0, tokens
        assert res["mods"][_L2_WRAPPER] is True, tokens
        assert res["mods"][_L2_RUNTIME] is leaf_expected, tokens
        assert res["mods"][_SERVER_RUNTIME] is False, tokens
        if leaf_expected:
            assert "bench l2" in res["text"], tokens
        else:
            for child in ("engine", "server", "l2"):
                assert child in res["text"], tokens


def test_executable_leaf_route() -> None:
    """mock runs end to end while unselected commands' runtimes stay unimported."""
    res = _run(
        ["mock", "--name", "t", "--num-items", "1"],
        want=("lmcache.cli.commands.server", _SERVER_RUNTIME),
    )
    assert res["exit"] == 0
    assert "Mock Result" in res["text"]
    assert res["mods"]["lmcache.cli.commands.server"] is True
    assert res["mods"][_SERVER_RUNTIME] is False


def test_nested_tool_route() -> None:
    """Two composite levels: selected path registers, other groups stay unscanned."""
    res = _run(
        ["tool", "cache-simulator", "-h"],
        want=(
            "lmcache.cli.commands.tool.cache_simulator.simulate_command",
            _L2_WRAPPER,
        ),
    )
    assert res["exit"] == 0
    for child in ("simulate", "sweep", "gen-dataset"):
        assert child in res["text"]
    assert res["mods"]["lmcache.cli.commands.tool.cache_simulator.simulate_command"]
    assert not res["mods"][_L2_WRAPPER]


def test_explicit_argv_ignores_decoy_ambient_argv() -> None:
    """main(tokens) registers and parses tokens, never the process sys.argv."""
    decoy = ["mock", "--name", "decoy"]
    res = _run(["bench", "-h"], explicit=True, decoy=decoy)
    assert res["exit"] == 0
    assert "engine" in res["text"]
    assert "Mock Result" not in res["text"]
    assert res["ambient"] == decoy


def test_lazy_all_commands_shim() -> None:
    """ALL_COMMANDS resolves on first access, caches, and holds the full registry."""
    res = _run([])
    assert not res["shim_at_import"]
    assert res["shim_cached"]
    assert {"mock", "bench", "server", "tool"} <= set(res["all_names"])
