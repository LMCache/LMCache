# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache query kvcache`` and its coverage summarizer."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch
import argparse
import io
import json
import urllib.error

# Third Party
import pytest

# First Party
from lmcache.cli.commands.query import QueryCommand
from lmcache.cli.commands.query import kvcache_command as kvcache_module
from lmcache.cli.commands.query._lookup import post_lookup, summarize_coverage
from lmcache.cli.commands.query.kvcache_command import KVCacheCommand

CHUNK_TOKENS = 4


def _key(chunk: int, rank: int) -> dict[str, Any]:
    """One encoded object key, shaped like the coordinator's reply."""
    return {
        "chunk_hash_hex": f"{chunk:064x}",
        "model_name": "m",
        "kv_rank": rank,
        "cache_salt": "",
    }


def _entry(
    chunk: int,
    rank: int,
    tiers: list[str],
    tokens: int = CHUNK_TOKENS,
) -> dict[str, Any]:
    """One ``DirectoryKeyPlacements`` entry; ``tiers`` empty means a miss."""
    return {
        "key": _key(chunk, rank),
        "placements": [
            {
                "instance_id": f"inst-{rank}",
                "incarnation": 1,
                "tier": tier,
                "backend": "cpu" if tier == "l1" else "disk",
                "size_bytes": 1024,
                "shared": False,
            }
            for tier in tiers
        ],
        "token_ids": list(range(tokens)),
        "access_count": 1,
    }


def _reply(chunks: int, cached_per_chunk: list[list[str]]) -> dict[str, Any]:
    """A tokens-form ``DirectoryLookupResponse`` with one rank per chunk.

    Args:
        chunks: Chunk count the coordinator resolved.
        cached_per_chunk: Per chunk, the tiers its single rank is placed on;
            an empty list marks the chunk as absent.
    """
    return {
        "chunks": chunks,
        "results": [
            _entry(chunk, 0, tiers) for chunk, tiers in enumerate(cached_per_chunk)
        ],
    }


class TestSummarizeCoverage:
    def test_full_hit(self) -> None:
        result = summarize_coverage(_reply(2, [["l1"], ["l1", "l2"]]), prompt_tokens=8)

        assert result.status == "HIT"
        assert (result.cached_chunks, result.total_chunks) == (2, 2)
        assert result.cached_tokens == 2 * CHUNK_TOKENS
        assert result.tier_chunks == (("l1", 2), ("l2", 1))

    def test_prefix_stops_at_first_gap(self) -> None:
        """A resident chunk behind a gap is not reusable, so it is not counted."""
        result = summarize_coverage(_reply(3, [["l1"], [], ["l1"]]), prompt_tokens=12)

        assert result.status == "HIT (partial)"
        assert (result.cached_chunks, result.total_chunks) == (1, 3)
        assert result.cached_tokens == CHUNK_TOKENS

    def test_miss(self) -> None:
        result = summarize_coverage(_reply(2, [[], []]), prompt_tokens=8)

        assert result.status == "MISS"
        assert result.cached_chunks == 0
        assert result.cached_tokens == 0
        assert result.tier_chunks == ()
        assert result.locations == ()

    def test_prompt_shorter_than_one_chunk(self) -> None:
        """No complete chunk means no key was resolved at all."""
        result = summarize_coverage({"chunks": 0, "results": []}, prompt_tokens=3)

        assert result.status == "MISS"
        assert result.total_chunks == 0
        assert result.prompt_tokens == 3

    def test_counts_a_chunk_once_per_rank_and_tier(self) -> None:
        """Two ranks on the same tier still count as one chunk for that tier."""
        body = {
            "chunks": 1,
            "results": [_entry(0, 0, ["l1"]), _entry(0, 1, ["l1"])],
        }

        result = summarize_coverage(body, prompt_tokens=CHUNK_TOKENS)

        assert result.status == "HIT"
        assert result.cached_chunks == 1
        assert result.tier_chunks == (("l1", 1),)

    def test_requires_every_rank_of_a_chunk(self) -> None:
        """A chunk missing one rank's placement cannot be reused."""
        body = {
            "chunks": 1,
            "results": [_entry(0, 0, ["l1"]), _entry(0, 1, [])],
        }

        result = summarize_coverage(body, prompt_tokens=CHUNK_TOKENS)

        assert result.status == "MISS"

    def test_locations_are_deduped_in_report_order(self) -> None:
        body = {
            "chunks": 3,
            "results": [
                _entry(0, 0, ["l1"]),
                _entry(1, 0, ["l1"]),
                _entry(2, 0, ["l2"]),
            ],
        }

        result = summarize_coverage(body, prompt_tokens=3 * CHUNK_TOKENS)

        assert result.locations == ("inst-0: l1/cpu", "inst-0: l2/disk")

    def test_shared_placement_has_no_instance(self) -> None:
        body = _reply(1, [["l2"]])
        body["results"][0]["placements"][0]["instance_id"] = ""

        result = summarize_coverage(body, prompt_tokens=CHUNK_TOKENS)

        assert result.locations == ("(shared): l2/disk",)


class _FakeResponse:
    """Context-manager stand-in for ``urlopen``'s reply."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class TestPostLookup:
    def test_posts_json_to_directory_lookup(self) -> None:
        seen: dict[str, Any] = {}

        def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
            seen["url"] = request.full_url
            seen["body"] = json.loads(request.data)
            seen["method"] = request.get_method()
            return _FakeResponse(json.dumps({"chunks": 0, "results": []}).encode())

        with patch("urllib.request.urlopen", fake_urlopen):
            body = post_lookup("coord:9000", {"token_ids": [1, 2], "model_name": "m"})

        assert seen["url"] == "http://coord:9000/directory/lookup"
        assert seen["method"] == "POST"
        assert seen["body"] == {"token_ids": [1, 2], "model_name": "m"}
        assert body == {"chunks": 0, "results": []}

    def test_strips_trailing_slash_from_url(self) -> None:
        seen: dict[str, Any] = {}

        def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
            seen["url"] = request.full_url
            return _FakeResponse(b"{}")

        with patch("urllib.request.urlopen", fake_urlopen):
            post_lookup("http://coord:9000/", {})

        assert seen["url"] == "http://coord:9000/directory/lookup"

    def test_http_error_detail_reaches_the_user(self) -> None:
        error = urllib.error.HTTPError(
            "http://coord:9000/directory/lookup",
            422,
            "Unprocessable Entity",
            None,  # type: ignore[arg-type]
            io.BytesIO(b'{"detail": "token sequence too long"}'),
        )

        with patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(RuntimeError, match="token sequence too long"):
                post_lookup("coord:9000", {})

    def test_unreachable_coordinator_is_a_runtime_error(self) -> None:
        error = urllib.error.URLError("connection refused")

        with patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(RuntimeError, match="is the coordinator running"):
                post_lookup("coord:9000", {})


class _FakeTokenizer:
    """Tokenizer stand-in that expands every prompt to ``tokens`` IDs."""

    def __init__(self, tokens: int) -> None:
        self.tokens = tokens
        self.seen: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.seen.append(text)
        return list(range(self.tokens))


def _namespace(**overrides: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "url": "http://coord:9000",
        "prompt": "hello",
        "model": "m",
        "world_size": 2,
        "cache_salt": "",
        "timeout": 2.0,
        "documents": [],
        "format": "terminal",
        "output": None,
        "quiet": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def tokenizer() -> Iterator[_FakeTokenizer]:
    tok = _FakeTokenizer(tokens=3 * CHUNK_TOKENS)
    with patch.object(kvcache_module, "try_load_tokenizer", return_value=tok):
        yield tok


class TestKVCacheCommand:
    def test_name_and_help(self) -> None:
        cmd = KVCacheCommand()
        assert cmd.name() == "kvcache"
        assert "cache" in cmd.help().lower()

    def test_registers_under_query(self) -> None:
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        QueryCommand().register(sub)

        args = parser.parse_args(
            [
                "query",
                "kvcache",
                "--url",
                "http://coord:9000",
                "--prompt",
                "hello",
                "--model",
                "m",
                "--world-size",
                "4",
                "--cache-salt",
                "tenant-a",
            ]
        )

        assert args.func is not None
        assert args.world_size == 4
        assert args.cache_salt == "tenant-a"

    def test_sends_token_ids_and_identity(
        self, capsys: pytest.CaptureFixture[str], tokenizer: _FakeTokenizer
    ) -> None:
        sent: dict[str, Any] = {}

        def fake_post(url: str, payload: Any, timeout: float = 10.0) -> Any:
            sent["url"] = url
            sent["payload"] = payload
            sent["timeout"] = timeout
            return _reply(3, [["l1"], ["l1"], ["l2"]])

        with patch.object(kvcache_module, "post_lookup", fake_post):
            KVCacheCommand().execute(_namespace())

        assert sent["url"] == "http://coord:9000"
        assert sent["payload"] == {
            "token_ids": list(range(12)),
            "model_name": "m",
            "world_size": 2,
            "cache_salt": "",
        }
        assert sent["timeout"] == 2.0
        assert tokenizer.seen == ["hello"]

    def test_reports_coverage_and_tiers(
        self, capsys: pytest.CaptureFixture[str], tokenizer: _FakeTokenizer
    ) -> None:
        with patch.object(
            kvcache_module,
            "post_lookup",
            return_value=_reply(3, [["l1"], ["l1", "l2"], []]),
        ):
            KVCacheCommand().execute(_namespace())

        out = capsys.readouterr().out
        assert "HIT (partial)" in out
        assert "Query KV Cache" in out
        assert "disk" in out

    def test_emits_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        tokenizer = _FakeTokenizer(tokens=CHUNK_TOKENS)
        with (
            patch.object(kvcache_module, "try_load_tokenizer", return_value=tokenizer),
            patch.object(
                kvcache_module,
                "post_lookup",
                return_value=_reply(1, [["l1"]]),
            ),
        ):
            KVCacheCommand().execute(_namespace(format="json"))

        report = json.loads(capsys.readouterr().out)
        assert report["metrics"]["cached_chunks_hit"] == 1
        assert report["metrics"]["cache_status"] == "HIT"

    def test_missing_tokenizer_fails_loudly(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Silently reporting a miss would be indistinguishable from an empty cache."""
        with patch.object(kvcache_module, "try_load_tokenizer", return_value=None):
            with pytest.raises(SystemExit) as exit_info:
                KVCacheCommand().execute(_namespace(model="unavailable"))

        assert exit_info.value.code == 1
        assert "No tokenizer available" in capsys.readouterr().err

    def test_coordinator_error_fails_loudly(
        self, capsys: pytest.CaptureFixture[str], tokenizer: _FakeTokenizer
    ) -> None:
        with patch.object(
            kvcache_module,
            "post_lookup",
            side_effect=RuntimeError("is the coordinator running?"),
        ):
            with pytest.raises(SystemExit) as exit_info:
                KVCacheCommand().execute(_namespace())

        assert exit_info.value.code == 1
        assert "is the coordinator running?" in capsys.readouterr().err
