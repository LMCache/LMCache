# SPDX-License-Identifier: Apache-2.0
"""Tests for the configuration-dump helpers in
:mod:`lmcache.v1.multiprocess.http_server`.

Covers:
- Default-path derivation per HTTP port.
- Dataclass and plain-dict serialization to JSON-safe structures.
- File-write behavior (valid JSON, parent-directory creation,
  overwrite-on-rewrite, indented output).
"""

# Standard
from dataclasses import dataclass
from pathlib import Path
import json
import tempfile

# First Party
from lmcache.v1.multiprocess.http_server import (
    _resolve_config_dump_path,
    _serialize_configs,
    _write_config_dump,
)


@dataclass
class _FakeConfig:
    """Minimal dataclass stand-in for a real LMCache config object."""

    host: str = "0.0.0.0"
    port: int = 8080


class TestResolveConfigDumpPath:
    """Tests for ``_resolve_config_dump_path``."""

    def test_empty_string_uses_default_path_for_port(self) -> None:
        """Empty input resolves to /tmp/lmcache-config-<port>.json."""
        path = _resolve_config_dump_path("", 9999)
        assert path == Path("/tmp/lmcache-config-9999.json")

    def test_explicit_path_is_passed_through(self) -> None:
        """A user-supplied path is returned verbatim as a ``Path``."""
        path = _resolve_config_dump_path("/var/log/foo.json", 8080)
        assert path == Path("/var/log/foo.json")

    def test_different_ports_produce_distinct_defaults(self) -> None:
        """Two servers on the same host don't overwrite each other."""
        a = _resolve_config_dump_path("", 8080)
        b = _resolve_config_dump_path("", 8081)
        assert a != b


class TestSerializeConfigs:
    """Tests for ``_serialize_configs``."""

    def test_dataclass_values_are_serialized(self) -> None:
        """Dataclass instances become nested dicts of JSON-safe values."""
        result = _serialize_configs({"http": _FakeConfig()})
        assert result == {"http": {"host": "0.0.0.0", "port": 8080}}

    def test_plain_dict_passes_through_make_json_safe(self) -> None:
        """Non-dataclass values are still made JSON-safe (e.g. Path -> str)."""
        result = _serialize_configs({"extra": {"k": Path("/v")}})
        assert result == {"extra": {"k": "/v"}}

    def test_multiple_keys_are_all_serialized(self) -> None:
        """All configs in the mapping appear in the output."""
        result = _serialize_configs(
            {
                "a": _FakeConfig(host="1.1.1.1"),
                "b": _FakeConfig(port=9090),
            }
        )
        assert set(result.keys()) == {"a", "b"}
        assert result["a"]["host"] == "1.1.1.1"
        assert result["b"]["port"] == 9090

    def test_empty_mapping_returns_empty_dict(self) -> None:
        """An empty config map produces an empty result."""
        assert _serialize_configs({}) == {}


class TestWriteConfigDump:
    """Tests for ``_write_config_dump``."""

    def test_writes_valid_json_to_disk(self) -> None:
        """File contains the serialized payload as parseable JSON."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dump.json"
            _write_config_dump({"http": _FakeConfig()}, path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert data == {"http": {"host": "0.0.0.0", "port": 8080}}

    def test_creates_missing_parent_directories(self) -> None:
        """Nested target directories are created automatically."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "subdir" / "dump.json"
            _write_config_dump({"http": _FakeConfig()}, path)
            assert path.exists()

    def test_overwrites_existing_file(self) -> None:
        """A new dump replaces stale contents at the same path."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dump.json"
            path.write_text("stale contents")
            _write_config_dump({"http": _FakeConfig(port=9000)}, path)
            data = json.loads(path.read_text())
            assert data == {"http": {"host": "0.0.0.0", "port": 9000}}

    def test_output_is_indented(self) -> None:
        """The dumped file is human-readable (indented JSON)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dump.json"
            _write_config_dump({"http": _FakeConfig()}, path)
            text = path.read_text()
            assert "\n" in text
            assert '  "host"' in text

    def test_keys_are_sorted(self) -> None:
        """JSON output uses sorted keys for stable diffs across runs."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dump.json"
            _write_config_dump(
                {
                    "z_last": _FakeConfig(),
                    "a_first": _FakeConfig(),
                },
                path,
            )
            text = path.read_text()
            assert text.index('"a_first"') < text.index('"z_last"')
