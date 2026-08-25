# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache query kvcache`` CLI command."""

# Standard
from unittest.mock import MagicMock, patch
import argparse
import json
import urllib.error

# Third Party
import pytest

# First Party
from lmcache.cli.commands.query._lookup import (
    CacheLookup,
    CoverageResult,
    summarize_coverage,
)
from lmcache.cli.commands.query.kvcache_command import KVCacheCommand


def _kvcache_parser() -> tuple[KVCacheCommand, argparse.ArgumentParser]:
    """Build a parser with ``KVCacheCommand`` registered as a subcommand."""
    cmd = KVCacheCommand()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    cmd.register(sub)
    return cmd, parser


def _fake_response(payload: dict) -> MagicMock:
    """Build a mock ``urlopen`` context manager returning *payload* as JSON."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


class TestSummarizeCoverage:
    def test_full_hit(self) -> None:
        layout = {"inst-0": ("cpu", 256)}
        result = summarize_coverage(layout, total_tokens=256, chunk_size=256)
        assert isinstance(result, CoverageResult)
        assert result.prompt_tokens == 256
        assert result.cached_tokens == 256
        assert result.cache_status == "HIT"
        assert result.cached_chunks == 1
        assert result.total_chunks == 1
        assert result.locations == [("inst-0", "cpu")]

    def test_partial_hit(self) -> None:
        layout = {"inst-0": ("cpu", 512)}
        result = summarize_coverage(layout, total_tokens=768, chunk_size=256)
        assert result.cache_status == "HIT (partial)"
        assert result.cached_tokens == 512
        assert result.cached_chunks == 2
        assert result.total_chunks == 3

    def test_miss_on_empty_layout(self) -> None:
        result = summarize_coverage({}, total_tokens=300, chunk_size=256)
        assert result.cache_status == "MISS"
        assert result.cached_tokens == 0
        assert result.cached_chunks == 0
        assert result.total_chunks == 2
        assert result.locations == []

    def test_multiple_instances_use_longest_prefix(self) -> None:
        # layout_info arrives from JSON as lists, not tuples.
        layout = {"inst-0": ["cpu", 256], "inst-1": ["disk", 512]}
        result = summarize_coverage(layout, total_tokens=512, chunk_size=256)
        assert result.cached_tokens == 512
        assert result.cache_status == "HIT"
        assert ("inst-1", "disk") in result.locations
        assert ("inst-0", "cpu") in result.locations


class TestRequestLookup:
    @patch("lmcache.cli.commands.query._lookup.urllib.request.urlopen")
    def test_posts_tokens_and_returns_layout_info(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = _fake_response(
            {"event_id": "x", "layout_info": {"inst-0": ["cpu", 256]}}
        )
        lookup = CacheLookup(url="http://host:5555", model="m")

        layout = lookup.request_lookup([1, 2, 3])

        assert layout == {"inst-0": ["cpu", 256]}
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://host:5555/lookup"
        assert req.get_method() == "POST"
        assert json.loads(req.data.decode()) == {"tokens": [1, 2, 3]}

    @patch("lmcache.cli.commands.query._lookup.urllib.request.urlopen")
    def test_bare_host_gets_http_scheme(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _fake_response({"layout_info": {}})
        lookup = CacheLookup(url="host:5555", model="m")

        lookup.request_lookup([1])

        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://host:5555/lookup"

    @patch("lmcache.cli.commands.query._lookup.urllib.request.urlopen")
    def test_connection_error_becomes_runtime_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        lookup = CacheLookup(url="http://host:5555", model="m")

        with pytest.raises(RuntimeError, match="lookup"):
            lookup.request_lookup([1])


class TestTokenize:
    @patch("transformers.AutoTokenizer")
    def test_encodes_prompt_with_model_tokenizer(self, mock_auto: MagicMock) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [10, 20, 30]
        mock_auto.from_pretrained.return_value = tokenizer
        lookup = CacheLookup(url="http://host", model="facebook/opt-125m")

        tokens = lookup.tokenize("hello")

        assert tokens == [10, 20, 30]
        mock_auto.from_pretrained.assert_called_once_with("facebook/opt-125m")
        tokenizer.encode.assert_called_once_with("hello")

    def test_missing_transformers_raises_runtime_error(self) -> None:
        lookup = CacheLookup(url="http://host", model="m")
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(RuntimeError, match="transformers"):
                lookup.tokenize("hello")

    @patch("transformers.AutoTokenizer")
    def test_gated_model_load_failure_hints_login(self, mock_auto: MagicMock) -> None:
        mock_auto.from_pretrained.side_effect = OSError("gated repo")
        lookup = CacheLookup(url="http://host", model="meta-llama/Llama-3.1-8B")

        with pytest.raises(RuntimeError, match="huggingface-cli login"):
            lookup.tokenize("hello")


class TestRun:
    def test_run_tokenizes_looks_up_and_summarizes(self) -> None:
        lookup = CacheLookup(url="http://host", model="m", chunk_size=256)
        with (
            patch.object(lookup, "tokenize", return_value=[0] * 512),
            patch.object(
                lookup, "request_lookup", return_value={"inst-0": ["cpu", 512]}
            ),
        ):
            result = lookup.run("prompt")

        assert isinstance(result, CoverageResult)
        assert result.prompt_tokens == 512
        assert result.cached_tokens == 512
        assert result.cache_status == "HIT"
        assert result.locations == [("inst-0", "cpu")]


class TestKVCacheCommandMetadata:
    def test_name(self) -> None:
        assert KVCacheCommand().name() == "kvcache"

    def test_help_is_not_placeholder(self) -> None:
        help_text = KVCacheCommand().help().lower()
        assert "not implemented" not in help_text
        assert "cache" in help_text


class TestKVCacheCommandArguments:
    def test_required_and_default_args(self) -> None:
        _, parser = _kvcache_parser()
        args = parser.parse_args(
            [
                "kvcache",
                "--url",
                "http://host:5555",
                "--prompt",
                "{ctx} question",
                "--model",
                "m",
                "--documents",
                "ctx=/tmp/x",
            ],
        )
        assert args.url == "http://host:5555"
        assert args.prompt == "{ctx} question"
        assert args.model == "m"
        assert args.documents == ["ctx=/tmp/x"]
        assert args.chunk_size == 256

    def test_chunk_size_override(self) -> None:
        _, parser = _kvcache_parser()
        args = parser.parse_args(
            [
                "kvcache",
                "--url",
                "http://host:5555",
                "--prompt",
                "hi",
                "--model",
                "m",
                "--chunk-size",
                "128",
            ],
        )
        assert args.chunk_size == 128


class TestKVCacheCommandExecute:
    @patch("lmcache.cli.commands.query.kvcache_command.CacheLookup")
    def test_execute_renders_coverage(
        self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_lookup = MagicMock()
        mock_lookup.run.return_value = CoverageResult(
            prompt_tokens=512,
            cached_tokens=256,
            cache_status="HIT (partial)",
            cached_chunks=1,
            total_chunks=2,
            locations=[("inst-0", "cpu")],
        )
        mock_cls.return_value = mock_lookup

        cmd, parser = _kvcache_parser()
        args = parser.parse_args(
            [
                "kvcache",
                "--url",
                "http://host:5555",
                "--prompt",
                "hello",
                "--model",
                "facebook/opt-125m",
            ],
        )
        cmd.execute(args)

        out = capsys.readouterr().out
        assert "Query KV Cache" in out
        assert "HIT (partial)" in out
        assert "256/512" in out
        assert "1/2" in out
        assert "cpu" in out
        mock_cls.assert_called_once_with(
            url="http://host:5555",
            model="facebook/opt-125m",
            chunk_size=256,
        )
        mock_lookup.run.assert_called_once_with("hello")

    @patch("lmcache.cli.commands.query.kvcache_command.CacheLookup")
    def test_execute_reports_miss(
        self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_lookup = MagicMock()
        mock_lookup.run.return_value = CoverageResult(
            prompt_tokens=300,
            cached_tokens=0,
            cache_status="MISS",
            cached_chunks=0,
            total_chunks=2,
            locations=[],
        )
        mock_cls.return_value = mock_lookup

        cmd, parser = _kvcache_parser()
        args = parser.parse_args(
            ["kvcache", "--url", "http://h", "--prompt", "hi", "--model", "m"],
        )
        cmd.execute(args)

        assert "MISS" in capsys.readouterr().out

    @patch("lmcache.cli.commands.query.kvcache_command.CacheLookup")
    def test_execute_error_exits_1_to_stderr(
        self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_lookup = MagicMock()
        mock_lookup.run.side_effect = RuntimeError("lookup boom")
        mock_cls.return_value = mock_lookup

        cmd, parser = _kvcache_parser()
        args = parser.parse_args(
            ["kvcache", "--url", "http://h", "--prompt", "hi", "--model", "m"],
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd.execute(args)

        assert exc_info.value.code == 1
        assert "lookup boom" in capsys.readouterr().err
