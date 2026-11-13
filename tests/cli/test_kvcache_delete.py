# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache kvcache delete``.

The command maps onto the server's ``DELETE /cache/objects`` endpoint, which
already accepted L1 keys before there was a CLI for it. These tests pin the
request body the CLI builds and the guards around a destructive call.
"""

# Standard
from argparse import Namespace
import json

# Third Party
import pytest

# First Party
from lmcache.cli.commands import kvcache as kvcache_module
from lmcache.cli.commands.kvcache import KVCacheCommand

KEY = {"chunk_hash_hex": "ab" * 16, "model_name": "qwen", "kv_rank": 0}


def _args(**overrides) -> Namespace:
    base = {
        "action": "delete",
        "url": "http://localhost:8083",
        "keys": None,
        "keys_file": None,
        "tier": "l1",
        "adapter": None,
        "force": False,
        "yes": True,
        "quiet": True,
        "format": "json",
        "output": None,
    }
    base.update(overrides)
    return Namespace(**base)


@pytest.fixture
def captured(monkeypatch):
    """Capture the HTTP call the command would make."""
    calls: list[tuple[str, str, dict]] = []

    def fake_request(method, url, data=None):
        calls.append((method, url, data))
        return {"deleted": 1, "skipped": 0, "ok": True}

    monkeypatch.setattr(kvcache_module, "_http_request", fake_request)
    return calls


class TestRequestShape:
    def test_single_key_defaults_to_l1(self, captured):
        KVCacheCommand().execute(_args(keys=[json.dumps(KEY)]))

        assert len(captured) == 1
        method, url, body = captured[0]
        assert method == "DELETE"
        assert url == "http://localhost:8083/cache/objects"
        assert body == {"keys": [KEY], "tier": "l1", "force": False}

    def test_repeated_keys_are_all_sent(self, captured):
        other = {**KEY, "kv_rank": 1}
        KVCacheCommand().execute(
            _args(keys=[json.dumps(KEY), json.dumps(other)], tier="all", force=True)
        )

        _, _, body = captured[0]
        assert body["keys"] == [KEY, other]
        assert body["tier"] == "all"
        assert body["force"] is True

    def test_keys_file_with_array(self, captured, tmp_path):
        path = tmp_path / "keys.json"
        path.write_text(json.dumps([KEY]), encoding="utf-8")

        KVCacheCommand().execute(_args(keys_file=str(path)))

        assert captured[0][2]["keys"] == [KEY]

    def test_keys_file_with_keys_object(self, captured, tmp_path):
        path = tmp_path / "keys.json"
        path.write_text(json.dumps({"keys": [KEY]}), encoding="utf-8")

        KVCacheCommand().execute(_args(keys_file=str(path)))

        assert captured[0][2]["keys"] == [KEY]

    def test_adapter_is_forwarded_only_for_l2(self, captured):
        KVCacheCommand().execute(
            _args(keys=[json.dumps(KEY)], tier="l2", adapter="s3-main")
        )
        assert captured[0][2]["adapter"] == "s3-main"

    def test_adapter_is_dropped_for_l1(self, captured):
        KVCacheCommand().execute(_args(keys=[json.dumps(KEY)], adapter="s3-main"))
        assert "adapter" not in captured[0][2]


class TestGuards:
    def test_refuses_without_yes_when_not_a_tty(self, captured, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
        monkeypatch.setattr(kvcache_module.sys.stdin, "isatty", lambda: False)

        with pytest.raises(SystemExit) as excinfo:
            KVCacheCommand().execute(_args(keys=[json.dumps(KEY)], yes=False))

        assert excinfo.value.code == 2
        assert captured == []

    def test_interactive_no_aborts(self, captured, monkeypatch):
        monkeypatch.setattr(kvcache_module.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _prompt: "n")

        with pytest.raises(SystemExit) as excinfo:
            KVCacheCommand().execute(_args(keys=[json.dumps(KEY)], yes=False))

        assert excinfo.value.code == 1
        assert captured == []

    def test_interactive_yes_proceeds(self, captured, monkeypatch):
        monkeypatch.setattr(kvcache_module.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _prompt: "y")

        KVCacheCommand().execute(_args(keys=[json.dumps(KEY)], yes=False))

        assert len(captured) == 1

    def test_invalid_key_json_exits_2(self, captured):
        with pytest.raises(SystemExit) as excinfo:
            KVCacheCommand().execute(_args(keys=["not json"]))
        assert excinfo.value.code == 2
        assert captured == []

    def test_missing_keys_file_exits_2(self, captured, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            KVCacheCommand().execute(_args(keys_file=str(tmp_path / "missing.json")))
        assert excinfo.value.code == 2
        assert captured == []

    def test_empty_keys_file_exits_2(self, captured, tmp_path):
        path = tmp_path / "keys.json"
        path.write_text("[]", encoding="utf-8")

        with pytest.raises(SystemExit) as excinfo:
            KVCacheCommand().execute(_args(keys_file=str(path)))
        assert excinfo.value.code == 2
        assert captured == []


class TestServerSideFailure:
    """A 200 response with ``ok: false`` means the deletion failed."""

    @staticmethod
    def _failing_request(monkeypatch):
        calls: list[tuple[str, str, dict]] = []

        def fake_request(method, url, data=None):
            calls.append((method, url, data))
            return {"deleted": 0, "skipped": 2, "ok": False, "error": "adapter down"}

        monkeypatch.setattr(kvcache_module, "_http_request", fake_request)
        return calls

    def test_ok_false_exits_1(self, monkeypatch, capsys):
        calls = self._failing_request(monkeypatch)

        with pytest.raises(SystemExit) as excinfo:
            KVCacheCommand().execute(
                _args(keys=[json.dumps(KEY)], quiet=False, format="terminal")
            )

        assert excinfo.value.code == 1
        assert len(calls) == 1
        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "adapter down" in out

    def test_ok_false_exits_1_in_quiet_mode(self, monkeypatch, capsys):
        calls = self._failing_request(monkeypatch)

        with pytest.raises(SystemExit) as excinfo:
            KVCacheCommand().execute(_args(keys=[json.dumps(KEY)], quiet=True))

        assert excinfo.value.code == 1
        assert len(calls) == 1
        assert capsys.readouterr().out == ""

    def test_ok_true_does_not_exit(self, captured):
        KVCacheCommand().execute(_args(keys=[json.dumps(KEY)]))
        assert len(captured) == 1
