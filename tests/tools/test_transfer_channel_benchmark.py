# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the transfer channel benchmark's configuration and the
write-verification side-channel.

Written against the documented behavior of
``lmcache/tools/transfer_channel_benchmark``: the ``--op`` option, the write
pattern the client stores, and the server's answer to a verify request.
"""

# Standard
from types import SimpleNamespace
import argparse
import itertools
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.tools.transfer_channel_benchmark import benchmark
from lmcache.tools.transfer_channel_benchmark.config import (
    BenchmarkConfig,
    add_benchmark_arguments,
    build_config,
)

_control_ports = itertools.count(27610)


def _parse(argv: list[str]) -> BenchmarkConfig:
    parser = argparse.ArgumentParser()
    add_benchmark_arguments(parser)
    return build_config(parser.parse_args(argv))


# =========================================================
# Configuration
# =========================================================
def test_op_defaults_to_read():
    cfg = _parse(["--role", "client"])
    assert cfg.op == "read"


def test_op_write_is_accepted():
    cfg = _parse(["--role", "client", "--op", "write", "--verify"])
    assert cfg.op == "write"
    assert cfg.verify is True


def test_op_rejects_unknown_values():
    with pytest.raises(ValueError):
        BenchmarkConfig(role="client", op="copy")


def test_parser_rejects_unknown_op():
    parser = argparse.ArgumentParser()
    add_benchmark_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--role", "client", "--op", "copy"])


# =========================================================
# Write pattern and server-side verification
# =========================================================
def test_write_pattern_differs_from_initial_pattern_every_iteration():
    for index in (0, 7, 255, 1000):
        for iteration in range(10):
            value = benchmark._expected_write_pattern(index, iteration)
            assert 0 <= value <= 255
            assert value != index % 256


def _objects(values: list[int], size: int = 16) -> list:
    """Fake source objects: each exposes a ``tensor`` filled with one byte."""
    return [
        SimpleNamespace(tensor=torch.full((size,), value, dtype=torch.uint8))
        for value in values
    ]


def test_mismatched_objects_reports_only_differing_indices():
    objs = _objects([1, 2, 3])
    assert benchmark._mismatched_objects(objs, {0: 1, 1: 2, 2: 3}) == []
    assert benchmark._mismatched_objects(objs, {0: 1, 1: 9, 2: 3}) == [1]
    # Out-of-range indices are mismatches rather than errors.
    assert benchmark._mismatched_objects(objs, {5: 1}) == [5]


def test_mismatched_objects_requires_every_byte_to_match():
    objs = _objects([4])
    objs[0].tensor[3] = 5
    assert benchmark._mismatched_objects(objs, {0: 4}) == [0]


def _serve(cfg: BenchmarkConfig, objs: list) -> threading.Thread:
    thread = threading.Thread(
        target=benchmark._serve_catalog, args=(cfg, objs), daemon=True
    )
    thread.start()
    return thread


def test_server_answers_catalog_and_verify_requests():
    port = next(_control_ports)
    cfg = BenchmarkConfig(
        role="server",
        control_url=f"127.0.0.1:{port}",
        page_size=16,
        object_size=16,
        num_objects=1,
        num_source_objects=3,
        server_timeout=3.0,
    )
    objs = _objects([0, 1, 2])
    for obj, offset in zip(objs, (0, 16, 32), strict=True):
        obj.shm_offset = offset
        obj.shm_byte_length = 16
    thread = _serve(cfg, objs)

    client_cfg = BenchmarkConfig(
        role="client",
        control_url=f"127.0.0.1:{port}",
        page_size=16,
        object_size=16,
        num_objects=1,
        server_timeout=3.0,
    )
    catalog = benchmark._fetch_catalog(client_cfg)
    assert catalog["page_size"] == 16
    assert catalog["object_size"] == 16
    assert [tuple(entry) for entry in catalog["objects"]] == [
        (0, 16, 0),
        (16, 16, 1),
        (32, 16, 2),
    ]

    # Matching bytes verify; a differing byte names the object.
    benchmark._verify_remote(client_cfg, {0: 0, 2: 2})
    with pytest.raises(RuntimeError, match=r"\[1\]"):
        benchmark._verify_remote(client_cfg, {1: 7})

    thread.join(timeout=cfg.server_timeout + 5)
    assert not thread.is_alive()
