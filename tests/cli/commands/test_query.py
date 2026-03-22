# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache query`` CLI command."""

# Standard
from unittest.mock import MagicMock, patch
import argparse

# Third Party
import pytest

# First Party
from lmcache.cli.commands.query import QueryCommand


@pytest.fixture
def cmd() -> QueryCommand:
    return QueryCommand()


@pytest.fixture
def parser(cmd: QueryCommand) -> argparse.ArgumentParser:
    """An :class:`~argparse.ArgumentParser` with ``QueryCommand`` registered."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    cmd.register(sub)
    return p


class TestQueryCommandMetadata:
    def test_name(self, cmd: QueryCommand) -> None:
        assert cmd.name() == "query"

    def test_help(self, cmd: QueryCommand) -> None:
        assert "inference" in cmd.help().lower()


class TestQueryCommandArguments:
    def test_registers_subcommand(self, parser: argparse.ArgumentParser) -> None:
        """The ``query engine`` subcommand should be parseable."""
        args = parser.parse_args(
            [
                "query",
                "engine",
                "--url",
                "http://localhost:8000/v1",
                "--prompt",
                "hello",
            ],
        )
        assert hasattr(args, "func")

    def test_engine_args_registered(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(
            [
                "query",
                "engine",
                "--url",
                "http://host:9/v1",
                "--prompt",
                "{ffmpeg} test",
                "--model",
                "m",
                "--max-tokens",
                "64",
                "--timeout",
                "5",
                "--corpus",
                "a=/tmp/x",
                "--completions",
                "--chat-first",
                "--format",
                "json",
                "--output",
                "/tmp/out",
            ],
        )
        assert args.url == "http://host:9/v1"
        assert args.prompt == "{ffmpeg} test"
        assert args.model == "m"
        assert args.max_tokens == 64
        assert args.timeout == 5.0
        assert args.corpus == ["a=/tmp/x"]
        assert args.completions is True
        assert args.chat_first is True
        assert args.format == "json"
        assert args.output == "/tmp/out"

    def test_default_values(self, parser: argparse.ArgumentParser) -> None:
        """Required args only — everything else should get defaults."""
        args = parser.parse_args(
            [
                "query",
                "engine",
                "--url",
                "http://localhost:8000/v1",
                "--prompt",
                "hi",
            ],
        )
        assert args.model is None
        assert args.max_tokens == 128
        assert args.timeout == 30.0
        assert args.corpus == []
        assert args.completions is False
        assert args.chat_first is False
        assert args.format is None
        assert args.output is None


class TestQueryCommandExecute:
    def test_func_bound_to_execute(
        self, cmd: QueryCommand, parser: argparse.ArgumentParser
    ) -> None:
        """``parse_args`` should bind ``func`` to :meth:`QueryCommand.execute`."""
        args = parser.parse_args(
            [
                "query",
                "engine",
                "--url",
                "http://localhost:8000/v1",
                "--prompt",
                "hello",
                "--model",
                "m",
            ],
        )
        assert args.func == cmd.execute

    @patch("lmcache.cli.commands.query._query_with_fallback")
    def test_execute_calls_query_with_fallback(
        self,
        mock_qwf: MagicMock,
        cmd: QueryCommand,
        parser: argparse.ArgumentParser,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``execute()`` should call the streaming query helper with parsed args."""
        mock_qwf.return_value = {
            "prompt_tokens": 10,
            "output_tokens": 5,
            "ttft_ms": 1.0,
            "tpot_ms_per_token": 2.0,
            "total_latency_ms": 100.0,
            "throughput_tokens_per_s": 50.0,
        }
        args = parser.parse_args(
            [
                "query",
                "engine",
                "--url",
                "http://localhost:8000/v1",
                "--prompt",
                "hello",
                "--model",
                "facebook/opt-125m",
            ],
        )
        cmd.execute(args)

        mock_qwf.assert_called_once()
        call_kw = mock_qwf.call_args.kwargs
        assert call_kw["completions_only"] is False
        assert call_kw["chat_first"] is False

        out = capsys.readouterr().out
        assert "Query Engine Result" in out
        assert "facebook/opt-125m" in out

    @patch("lmcache.cli.commands.query._query_with_fallback")
    @patch("lmcache.cli.commands.query._first_model_id", return_value="listed-model")
    def test_execute_resolves_model_when_omitted(
        self,
        mock_first: MagicMock,
        mock_qwf: MagicMock,
        cmd: QueryCommand,
        parser: argparse.ArgumentParser,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_qwf.return_value = {
            "prompt_tokens": 1,
            "output_tokens": 1,
            "ttft_ms": 1.0,
            "tpot_ms_per_token": 1.0,
            "total_latency_ms": 10.0,
            "throughput_tokens_per_s": 10.0,
        }
        args = parser.parse_args(
            [
                "query",
                "engine",
                "--url",
                "http://localhost:8000/v1",
                "--prompt",
                "x",
            ],
        )
        cmd.execute(args)
        mock_first.assert_called_once()
        assert "listed-model" in capsys.readouterr().out

    @patch("lmcache.cli.commands.query._die")
    def test_execute_invalid_prompt_exits(
        self,
        mock_die: MagicMock,
        cmd: QueryCommand,
        parser: argparse.ArgumentParser,
    ) -> None:
        """Invalid ``--prompt`` placeholders call :func:`_die` (exits the process)."""
        mock_die.side_effect = SystemExit(1)
        args = parser.parse_args(
            [
                "query",
                "engine",
                "--url",
                "http://localhost:8000/v1",
                "--prompt",
                "{unknown_corpus}",
                "--model",
                "m",
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd.execute(args)
        assert exc_info.value.code == 1
        mock_die.assert_called_once()
        assert "Unknown corpus" in mock_die.call_args[0][0]
