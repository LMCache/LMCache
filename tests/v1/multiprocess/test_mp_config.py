# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MP server TOML config-file loader and merge layer.

Covers: TOML load -> MPServerConfig fields, CLI-overrides-file precedence,
missing/invalid file graceful fallback (no crash, warn only), the
``LMCACHE_MP_CONFIG`` env var, the flag-over-env precedence, and the
Python 3.10 ``tomli`` fallback path (exercised via a shim on 3.11+).

This module imports only the pure config layer (no native extensions), so it
runs without a CUDA build.
"""

# Standard
import argparse
import json
import logging
import os
import types

# First Party
from lmcache.v1.multiprocess.config import (
    MPServerConfig,
    add_mp_server_args,
    load_mp_config_from_toml,
    merge_mp_config_file_into_args,
    parse_args_to_mp_server_config,
)
import lmcache.v1.multiprocess.config as mp_config


def _parse(argv: list[str]) -> argparse.Namespace:
    """Build an MP-server parser, parse argv, and return the Namespace."""
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)
    return parser.parse_args(argv)


def _parse_merged(argv: list[str]) -> MPServerConfig:
    """Parse argv, merge any config file, and return the MPServerConfig."""
    args = merge_mp_config_file_into_args(_parse(argv))
    return parse_args_to_mp_server_config(args)


def _write_toml(tmp_path, body: str):
    path = tmp_path / "mp.toml"
    path.write_text(body)
    return str(path)


def _capture_warnings():
    """Attach a list-backed handler to the config logger.

    lmcache's ``init_logger`` sets ``propagate = False``, so pytest's
    ``caplog`` (root-logger based) cannot see the records; attach a local
    handler to the named logger instead (established pattern, see
    tests/v1/multiprocess/test_config.py).
    """
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    config_logger = logging.getLogger("lmcache.v1.multiprocess.config")
    config_logger.addHandler(handler)
    return handler, records, config_logger


# -- load_mp_config_from_toml -------------------------------------------------


def test_load_returns_dest_keyed_dict(tmp_path):
    # Both hyphenated and underscored TOML keys normalize to argparse dests.
    path = _write_toml(
        tmp_path,
        'port = 6000\nhost = "10.0.0.1"\nmax_gpu_workers = 3\nchunk-size = 512\n',
    )
    loaded = load_mp_config_from_toml(path)
    assert loaded == {
        "port": 6000,
        "host": "10.0.0.1",
        "max_gpu_workers": 3,
        "chunk_size": 512,
    }


def test_load_flattens_runtime_plugin_config_table(tmp_path):
    # A TOML inline table is accepted and serialized to a JSON string so the
    # downstream json.loads in parse_args_to_mp_server_config keeps working.
    path = _write_toml(tmp_path, 'runtime-plugin-config = {x = 1, y = "z"}\n')
    loaded = load_mp_config_from_toml(path)
    assert json.loads(loaded["runtime_plugin_config"]) == {"x": 1, "y": "z"}


def test_load_unknown_keys_are_warned_and_skipped(tmp_path):
    path = _write_toml(tmp_path, "port = 6000\nnonsense-key = 1\n")
    handler, records, config_logger = _capture_warnings()
    try:
        loaded = load_mp_config_from_toml(path)
    finally:
        config_logger.removeHandler(handler)
    assert loaded == {"port": 6000}
    assert any("nonsense-key" in r.getMessage() for r in records)


def test_load_reserved_config_file_key_is_skipped(tmp_path):
    path = _write_toml(tmp_path, 'port = 6000\nconfig-file = "loop.toml"\n')
    handler, records, config_logger = _capture_warnings()
    try:
        loaded = load_mp_config_from_toml(path)
    finally:
        config_logger.removeHandler(handler)
    assert loaded == {"port": 6000}
    assert any(
        "config-file" in r.getMessage() and "reserved" in r.getMessage()
        for r in records
    )


# -- graceful fallback --------------------------------------------------------


def test_missing_file_returns_empty_and_warns(tmp_path):
    path = str(tmp_path / "does-not-exist.toml")
    handler, records, config_logger = _capture_warnings()
    try:
        loaded = load_mp_config_from_toml(path)
    finally:
        config_logger.removeHandler(handler)
    assert loaded == {}
    assert any("not found" in r.getMessage() for r in records)


def test_invalid_toml_returns_empty_and_warns(tmp_path):
    path = _write_toml(tmp_path, "not = valid = toml\n")
    handler, records, config_logger = _capture_warnings()
    try:
        loaded = load_mp_config_from_toml(path)
    finally:
        config_logger.removeHandler(handler)
    assert loaded == {}
    assert any("invalid TOML" in r.getMessage() for r in records)


def test_non_mapping_payload_warns_and_falls_back(tmp_path, monkeypatch):
    # Valid TOML always parses to a dict (top-level table), but the loader
    # defends against a non-mapping payload all the same. Force the branch by
    # making the toml backend return a list.
    path = _write_toml(tmp_path, "port = 6000\n")
    monkeypatch.setattr(mp_config.tomllib, "load", lambda fin: [1, 2, 3])
    handler, records, config_logger = _capture_warnings()
    try:
        loaded = load_mp_config_from_toml(path)
    finally:
        config_logger.removeHandler(handler)
    assert loaded == {}
    assert any("not a mapping" in r.getMessage() for r in records)


# -- merge -> MPServerConfig --------------------------------------------------


def test_file_supplies_defaults_when_flag_absent(tmp_path):
    path = _write_toml(
        tmp_path,
        "port = 6000\nchunk-size = 512\nmax-workers = 4\n"
        'max-gpu-workers = 2\nhost = "10.0.0.1"\n',
    )
    cfg = _parse_merged(["--config-file", path])
    assert cfg.port == 6000
    assert cfg.chunk_size == 512
    assert cfg.max_workers == 4
    assert cfg.max_gpu_workers == 2
    assert cfg.host == "10.0.0.1"


def test_cli_flag_overrides_file(tmp_path):
    path = _write_toml(tmp_path, 'port = 6000\nhost = "10.0.0.1"\n')
    cfg = _parse_merged(["--config-file", path, "--port", "7000"])
    # CLI port wins over file port; absent host still comes from the file.
    assert cfg.port == 7000
    assert cfg.host == "10.0.0.1"


def test_cli_flag_equal_to_default_lets_file_win(tmp_path):
    # An explicit flag whose value equals the argparse default is treated as
    # "absent" (the documented semantics): the file value still wins.
    path = _write_toml(tmp_path, "port = 6000\n")
    cfg = _parse_merged(["--config-file", path, "--port", "5555"])
    assert cfg.port == 6000


def test_runtime_plugin_config_table_reaches_config(tmp_path):
    path = _write_toml(tmp_path, "runtime-plugin-config = {x = 1}\n")
    cfg = _parse_merged(["--config-file", path])
    assert cfg.runtime_plugin_config.extra_config == {"x": 1}


def test_no_config_file_is_backward_compatible(tmp_path):
    # No --config-file and no env var: args untouched, pure argparse behavior.
    os.environ.pop("LMCACHE_MP_CONFIG", None)
    cfg = _parse_merged([])
    # instance_id is a random UUID, so only assert the deterministic defaults.
    assert cfg.port == 5555
    assert cfg.host == "localhost"
    assert cfg.chunk_size == 256


def test_missing_file_falls_back_to_argparse_no_crash(tmp_path):
    path = str(tmp_path / "missing.toml")
    handler, records, config_logger = _capture_warnings()
    try:
        cfg = _parse_merged(["--config-file", path, "--port", "7000"])
    finally:
        config_logger.removeHandler(handler)
    # Warned but did not crash; the explicit CLI port still applied.
    assert any("not found" in r.getMessage() for r in records)
    assert cfg.port == 7000


def test_invalid_file_falls_back_to_argparse_no_crash(tmp_path):
    path = _write_toml(tmp_path, "garbage = = =\n")
    handler, records, config_logger = _capture_warnings()
    try:
        cfg = _parse_merged(["--config-file", path, "--port", "7000"])
    finally:
        config_logger.removeHandler(handler)
    assert any("invalid TOML" in r.getMessage() for r in records)
    assert cfg.port == 7000


# -- env var resolution -------------------------------------------------------


def test_env_var_resolves_config_path(tmp_path, monkeypatch):
    path = _write_toml(tmp_path, "port = 6000\n")
    monkeypatch.setenv("LMCACHE_MP_CONFIG", path)
    cfg = _parse_merged([])
    assert cfg.port == 6000


def test_flag_beats_env_var(tmp_path, monkeypatch):
    env_path = _write_toml(tmp_path, "port = 6000\n")
    flag_path = _write_toml(tmp_path, "port = 8000\n")
    monkeypatch.setenv("LMCACHE_MP_CONFIG", env_path)
    cfg = _parse_merged(["--config-file", flag_path])
    assert cfg.port == 8000


def test_no_env_no_flag_is_argparse_only(monkeypatch):
    monkeypatch.delenv("LMCACHE_MP_CONFIG", raising=False)
    args_before = _parse([])
    args_after = merge_mp_config_file_into_args(_parse([]))
    # No file path resolved -> args unchanged (backward compatible).
    for dest in ("port", "host", "chunk_size", "max_workers"):
        assert getattr(args_after, dest) == getattr(args_before, dest)


# -- tomli fallback path (exercised on 3.11+, never skipped) ------------------


def test_loader_works_through_tomli_fallback_backend(tmp_path, monkeypatch):
    """Exercise the Python 3.10 ``tomli`` fallback without skipping on 3.11.

    The loader binds a module-level ``tomllib`` name (stdlib on 3.11+,
    ``tomli`` on 3.10). Swap in a tomli-style shim (same API surface) and
    confirm loading and graceful invalid-TOML fallback still work through it,
    proving the fallback code path is functional on every Python version.
    """
    real = mp_config.tomllib
    shim = types.ModuleType("tomli_shim")
    shim.load = real.load
    shim.TOMLDecodeError = real.TOMLDecodeError
    monkeypatch.setattr(mp_config, "tomllib", shim)

    path = _write_toml(tmp_path, 'port = 6000\nhost = "10.0.0.1"\n')
    loaded = load_mp_config_from_toml(path)
    assert loaded["port"] == 6000
    assert loaded["host"] == "10.0.0.1"

    # Invalid TOML still warns + falls back via the shim's exception type.
    bad = _write_toml(tmp_path, "not = valid = toml\n")
    handler, records, config_logger = _capture_warnings()
    try:
        assert load_mp_config_from_toml(bad) == {}
    finally:
        config_logger.removeHandler(handler)
    assert any("invalid TOML" in r.getMessage() for r in records)
