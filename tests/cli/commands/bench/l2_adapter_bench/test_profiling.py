# SPDX-License-Identifier: Apache-2.0
"""Test for the L2 adapter benchmark flame-graph profiler."""

# Standard
from pathlib import Path
import os
import shutil

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.l2_adapter_bench.profiling import (
    FlameProfiler,
    ProfileError,
    check_profiling_deps,
)


def test_check_profiling_deps_missing_tool_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing tool names the tool and how to recover.

    Runs on any machine: the toolchain is faked absent via monkeypatch,
    so this never skips and always exercises the error path.
    """
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    with pytest.raises(ProfileError) as excinfo:
        check_profiling_deps("on-cpu")

    message = str(excinfo.value)
    assert "perf" in message
    # Actionable: points at the fallback mode.
    assert "off-cpu" in message


def test_flame_profiler_builds_with_real_toolchain(tmp_path: Path) -> None:
    if shutil.which("perf") is None:
        pytest.skip("perf not installed; flame-graph profiling unavailable")

    scripts_dir = tmp_path / "FlameGraph"
    scripts_dir.mkdir()
    (scripts_dir / "flamegraph.pl").write_text("#!/usr/bin/perl\n")
    output = tmp_path / "flames" / "out.svg"

    FlameProfiler(
        mode="on-cpu",
        output=str(output),
        flamegraph_dir=str(scripts_dir),
        pid=os.getpid(),
        title="test",
    )

    # The component validated the toolchain and prepared the output dir.
    assert output.parent.is_dir()
