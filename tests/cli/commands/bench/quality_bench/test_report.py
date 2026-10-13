# SPDX-License-Identifier: Apache-2.0
"""Tests for ``bench quality`` result directories."""

# Standard
from datetime import datetime
import os

# First Party
from lmcache.cli.commands.bench.quality_bench.report import create_run_dir

_STARTED = datetime(2026, 10, 12, 18, 30, 5)


class TestCreateRunDir:
    def test_named_for_the_start_time(self, tmp_path) -> None:
        path = create_run_dir(str(tmp_path), _STARTED)
        assert path == os.path.join(tmp_path, "20261012-183005")
        assert os.path.isdir(path)

    def test_creates_a_missing_output_dir(self, tmp_path) -> None:
        root = tmp_path / "nested" / "results"
        path = create_run_dir(str(root), _STARTED)
        assert os.path.isdir(path)

    def test_same_second_runs_get_their_own_directories(self, tmp_path) -> None:
        paths = [create_run_dir(str(tmp_path), _STARTED) for _ in range(3)]
        assert [os.path.basename(p) for p in paths] == [
            "20261012-183005",
            "20261012-183005-2",
            "20261012-183005-3",
        ]

    def test_leaves_existing_files_alone(self, tmp_path) -> None:
        cache = tmp_path / "judge_cache.jsonl"
        cache.write_text("{}\n")
        create_run_dir(str(tmp_path), _STARTED)
        assert cache.read_text() == "{}\n"
