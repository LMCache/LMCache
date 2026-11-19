# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for the verdict of the P2P write multiprocess workload.

``p2p_write_verify.py`` decides whether a GPU run of a main LMCache server and
a headless peer passes. These tests feed its ``verify`` function recorded rows
instead of live servers, so a broken run cannot pass unnoticed.

Run without loading the torch-dependent tests/conftest.py:
    uv run --no-sync --python 3.12 python -m pytest -q \
        --confcutdir=tests/ci tests/ci/test_p2p_write_verify.py
"""

# Standard
from pathlib import Path
from types import ModuleType
from typing import Any, Callable
import dataclasses
import importlib.util
import sys

# Third Party
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFY_SCRIPT = (
    REPO_ROOT
    / ".buildkite/k3_tests/multiprocess/scripts/workloads/vllm/p2p_write_verify.py"
)
USAGE_THRESHOLD = 0.5
MIN_HIT_RATIO = 0.85
MAIN_L1_BYTES = 100
OBJECTS_PER_PROMPT = 4
BYTES_PER_PROMPT = 20


def _load_verify_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("p2p_write_verify", VERIFY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


p2p_write_verify = _load_verify_script()


def _healthy_run() -> tuple[list[Any], list[Any]]:
    """Six prompts: the first two stay on the main server, the rest go to the
    peer, and the replay is served entirely from LMCache."""
    fill = []
    for prompt in range(6):
        on_main = min(prompt + 1, 2)
        on_peer = prompt + 1 - on_main
        fill.append(
            p2p_write_verify.Row(
                prompt=prompt,
                prompt_tokens=1000,
                cached_tokens=0,
                text="alpha",
                main_objects=on_main * OBJECTS_PER_PROMPT,
                main_used_bytes=on_main * BYTES_PER_PROMPT,
                main_total_bytes=MAIN_L1_BYTES,
                peer_objects=on_peer * OBJECTS_PER_PROMPT,
                peer_used_bytes=on_peer * BYTES_PER_PROMPT,
                main_objects_added=OBJECTS_PER_PROMPT if prompt < 2 else 0,
                peer_objects_added=0 if prompt < 2 else OBJECTS_PER_PROMPT,
                peer_used_bytes_added=0 if prompt < 2 else BYTES_PER_PROMPT,
            )
        )
    replay = [
        dataclasses.replace(
            fill[-1],
            prompt=prompt,
            cached_tokens=960,
            main_objects_added=0,
            peer_objects_added=0,
            peer_used_bytes_added=0,
        )
        for prompt in range(6)
    ]
    return fill, replay


def _spill_below_threshold(fill: list[Any], replay: list[Any]) -> None:
    fill[0] = dataclasses.replace(
        fill[0], peer_objects_added=1, peer_used_bytes_added=BYTES_PER_PROMPT // 4
    )


def _kept_local_above_threshold(fill: list[Any], replay: list[Any]) -> None:
    fill[2] = dataclasses.replace(
        fill[2],
        main_used_bytes=3 * BYTES_PER_PROMPT,
        main_objects_added=OBJECTS_PER_PROMPT,
        peer_objects_added=0,
        peer_used_bytes_added=0,
    )


def _fill_stops_early(fill: list[Any], replay: list[Any]) -> None:
    del fill[3:], replay[3:]


def _objects_evicted_during_fill(fill: list[Any], replay: list[Any]) -> None:
    fill[4] = dataclasses.replace(fill[4], main_objects_added=-OBJECTS_PER_PROMPT)


def _peer_prompt_misses_on_replay(fill: list[Any], replay: list[Any]) -> None:
    replay[5] = dataclasses.replace(replay[5], cached_tokens=0)


def _answer_changes_on_replay(fill: list[Any], replay: list[Any]) -> None:
    replay[5] = dataclasses.replace(replay[5], text="omega")


def _replay_keeps_peer_objects_locally(fill: list[Any], replay: list[Any]) -> None:
    replay[-1] = dataclasses.replace(
        replay[-1], main_objects=replay[-1].main_objects + OBJECTS_PER_PROMPT
    )


def test_healthy_run_passes() -> None:
    fill, replay = _healthy_run()

    assert p2p_write_verify.verify(fill, replay, USAGE_THRESHOLD, MIN_HIT_RATIO) == []


@pytest.mark.parametrize(
    "break_run",
    [
        _spill_below_threshold,
        _kept_local_above_threshold,
        _fill_stops_early,
        _objects_evicted_during_fill,
        _peer_prompt_misses_on_replay,
        _answer_changes_on_replay,
        _replay_keeps_peer_objects_locally,
    ],
)
def test_broken_run_is_rejected(
    break_run: Callable[[list[Any], list[Any]], None],
) -> None:
    fill, replay = _healthy_run()
    break_run(fill, replay)

    assert p2p_write_verify.verify(fill, replay, USAGE_THRESHOLD, MIN_HIT_RATIO)
