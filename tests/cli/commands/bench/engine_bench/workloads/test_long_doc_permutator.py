# SPDX-License-Identifier: Apache-2.0
"""Tests for long-doc-permutator workload config and workload."""

# Standard
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import math
import time

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.stats import RequestResult
from lmcache.cli.commands.bench.engine_bench.tokenizers import build_single_token_pool
from lmcache.cli.commands.bench.engine_bench.workloads import long_doc_permutator as ldp
from lmcache.cli.commands.bench.engine_bench.workloads.long_doc_permutator import (
    LongDocPermutatorConfig,
    LongDocPermutatorWorkload,
    _enumerate_permutations,
    _generate_contexts,
    _generate_system_prompt,
)

# Local
from ..fake_tokenizer import FakeTokenizer, make_fake_tokenizer

# ---------------------------------------------------------------------------
# LongDocPermutatorConfig — direct construction
# ---------------------------------------------------------------------------


class TestLongDocPermutatorConfig:
    def test_defaults(self) -> None:
        cfg = LongDocPermutatorConfig()
        assert cfg.num_contexts == 5
        assert cfg.context_length == 5000
        assert cfg.system_prompt_length == 1000
        assert cfg.num_permutations == 10
        assert cfg.vocab_size == 8000
        assert cfg.num_inflight_requests == 1
        assert cfg.max_output_length == 128

    def test_custom_values(self) -> None:
        cfg = LongDocPermutatorConfig(
            num_contexts=3,
            context_length=200,
            system_prompt_length=50,
            num_permutations=4,
            vocab_size=500,
            num_inflight_requests=2,
            max_output_length=1,
        )
        assert cfg.num_contexts == 3
        assert cfg.context_length == 200
        assert cfg.system_prompt_length == 50
        assert cfg.num_permutations == 4
        assert cfg.vocab_size == 500
        assert cfg.num_inflight_requests == 2
        assert cfg.max_output_length == 1

    def test_invalid_num_contexts_zero(self) -> None:
        with pytest.raises(ValueError, match="num_contexts must be >= 1"):
            LongDocPermutatorConfig(num_contexts=0)

    def test_invalid_num_contexts_negative(self) -> None:
        with pytest.raises(ValueError, match="num_contexts must be >= 1"):
            LongDocPermutatorConfig(num_contexts=-1)

    def test_invalid_context_length_zero(self) -> None:
        with pytest.raises(ValueError, match="context_length must be positive"):
            LongDocPermutatorConfig(context_length=0)

    def test_invalid_context_length_negative(self) -> None:
        with pytest.raises(ValueError, match="context_length must be positive"):
            LongDocPermutatorConfig(context_length=-5)

    def test_invalid_num_permutations_zero(self) -> None:
        with pytest.raises(ValueError, match="num_permutations must be >= 1"):
            LongDocPermutatorConfig(num_permutations=0)

    def test_invalid_vocab_size_zero(self) -> None:
        with pytest.raises(ValueError, match="vocab_size must be >= 1"):
            LongDocPermutatorConfig(vocab_size=0)

    def test_invalid_num_inflight_requests_zero(self) -> None:
        with pytest.raises(ValueError, match="num_inflight_requests must be >= 1"):
            LongDocPermutatorConfig(num_inflight_requests=0)

    def test_invalid_max_output_length_zero(self) -> None:
        with pytest.raises(ValueError, match="max_output_length must be >= 1"):
            LongDocPermutatorConfig(max_output_length=0)

    def test_invalid_max_output_length_negative(self) -> None:
        with pytest.raises(ValueError, match="max_output_length must be >= 1"):
            LongDocPermutatorConfig(max_output_length=-1)


# ---------------------------------------------------------------------------
# LongDocPermutatorConfig.resolve
# ---------------------------------------------------------------------------


class TestLongDocPermutatorConfigResolve:
    def test_resolve_defaults(self) -> None:
        cfg = LongDocPermutatorConfig.resolve()
        assert cfg.num_contexts == 5
        assert cfg.context_length == 5000
        assert cfg.system_prompt_length == 1000
        assert cfg.num_permutations == 10
        assert cfg.vocab_size == 8000
        assert cfg.num_inflight_requests == 1
        assert cfg.max_output_length == 128

    def test_resolve_custom(self) -> None:
        cfg = LongDocPermutatorConfig.resolve(
            num_contexts=3,
            context_length=200,
            system_prompt_length=50,
            num_permutations=4,
            vocab_size=500,
            num_inflight_requests=2,
            max_output_length=1,
        )
        assert cfg.num_contexts == 3
        assert cfg.context_length == 200
        assert cfg.system_prompt_length == 50
        assert cfg.num_permutations == 4
        assert cfg.vocab_size == 500
        assert cfg.num_inflight_requests == 2
        assert cfg.max_output_length == 1


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestGenerateSystemPrompt:
    def test_empty_for_zero_length(self, fake_tokenizer: FakeTokenizer) -> None:
        pool = build_single_token_pool(fake_tokenizer, 100)
        assert _generate_system_prompt(0, pool) == ""

    def test_exact_token_length(self, fake_tokenizer: FakeTokenizer) -> None:
        pool = build_single_token_pool(fake_tokenizer, 100)
        prompt = _generate_system_prompt(50, pool)
        assert len(fake_tokenizer.encode(prompt)) == 50

    def test_deterministic(self, fake_tokenizer: FakeTokenizer) -> None:
        pool = build_single_token_pool(fake_tokenizer, 100)
        assert _generate_system_prompt(30, pool, seed=42) == (
            _generate_system_prompt(30, pool, seed=42)
        )

    def test_different_seeds_differ(self, fake_tokenizer: FakeTokenizer) -> None:
        pool = build_single_token_pool(fake_tokenizer, 100)
        assert _generate_system_prompt(30, pool, seed=1) != (
            _generate_system_prompt(30, pool, seed=2)
        )


class TestGenerateContexts:
    def test_correct_count(self, fake_tokenizer: FakeTokenizer) -> None:
        pool = build_single_token_pool(fake_tokenizer, 200)
        contexts = _generate_contexts(4, 50, pool)
        assert len(contexts) == 4

    def test_exact_token_length(self, fake_tokenizer: FakeTokenizer) -> None:
        pool = build_single_token_pool(fake_tokenizer, 200)
        for ctx in _generate_contexts(3, 100, pool):
            assert len(fake_tokenizer.encode(ctx)) == 100

    def test_contexts_are_unique(self, fake_tokenizer: FakeTokenizer) -> None:
        pool = build_single_token_pool(fake_tokenizer, 250)
        contexts = _generate_contexts(5, 200, pool)
        assert len(set(contexts)) == 5

    def test_deterministic(self, fake_tokenizer: FakeTokenizer) -> None:
        pool = build_single_token_pool(fake_tokenizer, 200)
        assert _generate_contexts(3, 50, pool, seed=10) == (
            _generate_contexts(3, 50, pool, seed=10)
        )


class TestEnumeratePermutations:
    def test_all_permutations_when_small(self) -> None:
        # 3! = 6 <= num_permutations
        perms = _enumerate_permutations(3, 100)
        assert len(perms) == math.factorial(3)

    def test_capped_at_requested(self) -> None:
        perms = _enumerate_permutations(5, 4)
        assert len(perms) == 4

    def test_all_are_valid_permutations(self) -> None:
        perms = _enumerate_permutations(4, 10)
        for perm in perms:
            assert sorted(perm) == list(range(4))

    def test_no_duplicates(self) -> None:
        perms = _enumerate_permutations(4, 20)
        assert len(set(perms)) == len(perms)

    def test_deterministic(self) -> None:
        p1 = _enumerate_permutations(5, 8, seed=0)
        p2 = _enumerate_permutations(5, 8, seed=0)
        assert p1 == p2

    def test_single_context_single_permutation(self) -> None:
        perms = _enumerate_permutations(1, 5)
        assert perms == [(0,)]


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> LongDocPermutatorConfig:
    defaults = dict(
        num_contexts=3,
        context_length=20,
        system_prompt_length=10,
        num_permutations=4,
        vocab_size=100,
        num_inflight_requests=1,
    )
    defaults.update(overrides)
    return LongDocPermutatorConfig(**defaults)  # type: ignore[arg-type]


def _make_mock_result(request_id: str = "req_0") -> RequestResult:
    now = time.time()
    return RequestResult(
        request_id=request_id,
        successful=True,
        ttft=0.1,
        request_latency=0.5,
        num_input_tokens=100,
        num_output_tokens=5,
        decode_speed=20.0,
        submit_time=now,
        first_token_time=now + 0.1,
        finish_time=now + 0.5,
        error="",
    )


def _make_mock_sender() -> MagicMock:
    sender = MagicMock()
    sender.send_request = AsyncMock(return_value=_make_mock_result())
    sender.send_warmup_request = AsyncMock(return_value=_make_mock_result())
    sender.close = AsyncMock(return_value=None)
    return sender


def _make_workload(
    config: LongDocPermutatorConfig | None = None,
    seed: int = 42,
    tokenizer: FakeTokenizer | None = None,
) -> tuple[LongDocPermutatorWorkload, MagicMock, MagicMock, MagicMock]:
    if config is None:
        config = _make_config()
    if tokenizer is None:
        tokenizer = make_fake_tokenizer()
    sender = _make_mock_sender()
    collector = MagicMock()
    monitor = MagicMock()
    with patch.object(ldp, "try_load_tokenizer", return_value=tokenizer):
        workload = LongDocPermutatorWorkload(
            config,
            sender,
            collector,
            monitor,
            seed=seed,
            model_name="fake-model",
        )
    return workload, sender, collector, monitor


# ---------------------------------------------------------------------------
# LongDocPermutatorWorkload — construction
# ---------------------------------------------------------------------------


class TestLongDocPermutatorWorkloadInit:
    def test_system_prompt_built(self) -> None:
        cfg = _make_config(system_prompt_length=20)
        tokenizer = make_fake_tokenizer()
        w, *_ = _make_workload(cfg, tokenizer=tokenizer)
        assert len(tokenizer.encode(w._system_prompt)) == 20

    def test_contexts_have_exact_token_length(self) -> None:
        cfg = _make_config(num_contexts=3, context_length=25)
        tokenizer = make_fake_tokenizer()
        w, *_ = _make_workload(cfg, tokenizer=tokenizer)
        for ctx in w._contexts:
            assert len(tokenizer.encode(ctx)) == 25

    def test_raises_without_tokenizer(self) -> None:
        """A tokenizer is required: the length flags are token counts."""
        with patch.object(ldp, "try_load_tokenizer", return_value=None):
            with pytest.raises(ValueError, match="needs a tokenizer"):
                LongDocPermutatorWorkload(
                    _make_config(),
                    _make_mock_sender(),
                    MagicMock(),
                    MagicMock(),
                    model_name=None,
                )

    def test_system_prompt_empty_when_zero(self) -> None:
        cfg = _make_config(system_prompt_length=0)
        w, *_ = _make_workload(cfg)
        assert w._system_prompt == ""

    def test_contexts_count(self) -> None:
        cfg = _make_config(num_contexts=4)
        w, *_ = _make_workload(cfg)
        assert len(w._contexts) == 4

    def test_permutations_capped(self) -> None:
        # 3! = 6; requesting 4 should yield 4
        cfg = _make_config(num_contexts=3, num_permutations=4)
        w, *_ = _make_workload(cfg)
        assert len(w._permutations) == 4

    def test_permutations_all_when_small(self) -> None:
        cfg = _make_config(num_contexts=2, num_permutations=100)
        w, *_ = _make_workload(cfg)
        assert len(w._permutations) == math.factorial(2)

    def test_request_list_length_matches_permutations(self) -> None:
        cfg = _make_config(num_contexts=3, num_permutations=4)
        w, *_ = _make_workload(cfg)
        assert len(w._request_list) == len(w._permutations)

    def test_request_index_starts_at_zero(self) -> None:
        w, *_ = _make_workload()
        assert w._request_index == 0


# ---------------------------------------------------------------------------
# LongDocPermutatorWorkload — request list building
# ---------------------------------------------------------------------------


class TestBuildRequestList:
    def test_each_entry_is_tuple_messages_and_perm_idx(self) -> None:
        cfg = _make_config(num_contexts=2, num_permutations=2)
        w, *_ = _make_workload(cfg)
        for messages, perm_idx in w._request_list:
            assert isinstance(messages, list)
            assert isinstance(perm_idx, int)

    def test_system_prompt_prepended_when_present(self) -> None:
        cfg = _make_config(system_prompt_length=10)
        w, *_ = _make_workload(cfg)
        messages, _ = w._request_list[0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_no_system_message_when_empty(self) -> None:
        cfg = _make_config(system_prompt_length=0)
        w, *_ = _make_workload(cfg)
        messages, _ = w._request_list[0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_user_message_contains_all_contexts(self) -> None:
        cfg = _make_config(num_contexts=3, num_permutations=1)
        w, *_ = _make_workload(cfg)
        messages, perm_idx = w._request_list[0]
        user_content = messages[-1]["content"]
        perm = w._permutations[perm_idx]
        for idx in perm:
            assert w._contexts[idx] in user_content

    def test_perm_idx_matches_permutations_list(self) -> None:
        cfg = _make_config(num_contexts=3, num_permutations=4)
        w, *_ = _make_workload(cfg)
        for _, perm_idx in w._request_list:
            assert 0 <= perm_idx < len(w._permutations)


# ---------------------------------------------------------------------------
# LongDocPermutatorWorkload — warmup (async)
# ---------------------------------------------------------------------------


class TestLongDocPermutatorWarmup:
    def test_warmup_sends_exactly_one_request(self) -> None:
        w, sender, _, _ = _make_workload()
        asyncio.run(w.warmup())
        assert sender.send_warmup_request.call_count == 1

    def test_warmup_request_id_is_warmup_0(self) -> None:
        w, sender, _, _ = _make_workload()
        asyncio.run(w.warmup())
        call_args = sender.send_warmup_request.call_args[0]
        assert call_args[0] == "warmup_0"

    def test_warmup_sends_chat_messages(self) -> None:
        w, sender, _, _ = _make_workload()
        asyncio.run(w.warmup())
        messages = sender.send_warmup_request.call_args[0][1]
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles

    def test_warmup_ignores_max_output_length(self) -> None:
        """Warmup keeps the sender's 1-token default whatever the config asks."""
        w, sender, _, _ = _make_workload(_make_config(max_output_length=64))
        asyncio.run(w.warmup())
        assert "max_tokens" not in sender.send_warmup_request.await_args.kwargs

    def test_warmup_logs_progress(self) -> None:
        w, _, _, monitor = _make_workload()
        asyncio.run(w.warmup())
        assert monitor.on_request_sent.call_count == 1
        assert monitor.log_message.call_count >= 2  # start + complete

    def test_warmup_failed_result_logs_message(self) -> None:
        now = time.time()
        failed = RequestResult(
            request_id="warmup_0",
            successful=False,
            ttft=0.0,
            request_latency=0.0,
            num_input_tokens=0,
            num_output_tokens=0,
            decode_speed=0.0,
            submit_time=now,
            first_token_time=now,
            finish_time=now,
            error="connection refused",
        )
        w, sender, _, monitor = _make_workload()
        sender.send_warmup_request = AsyncMock(return_value=failed)
        asyncio.run(w.warmup())
        messages = [call[0][0] for call in monitor.log_message.call_args_list]
        assert any("failed" in m.lower() for m in messages)


# ---------------------------------------------------------------------------
# LongDocPermutatorWorkload — step (async)
# ---------------------------------------------------------------------------


class TestLongDocPermutatorStep:
    def test_first_step_dispatches_one_and_returns_zero(self) -> None:
        async def _run() -> None:
            cfg = _make_config(num_contexts=2, num_permutations=3)
            w, _, _, _ = _make_workload(cfg)
            result = await w.step(0.0)
            assert result == 0.0
            assert w._request_index == 1

        asyncio.run(_run())

    def test_step_returns_negative_when_all_done(self) -> None:
        async def _run() -> None:
            cfg = _make_config(num_contexts=2, num_permutations=2)
            w, _, _, _ = _make_workload(cfg)

            while True:
                r = await w.step(0.0)
                if w._pending_tasks:
                    await asyncio.gather(*w._pending_tasks)
                if r == -1.0:
                    break

            assert w._request_index == len(w._request_list)

        asyncio.run(_run())

    def test_step_calls_send_request(self) -> None:
        async def _run() -> None:
            cfg = _make_config(num_contexts=2, num_permutations=1)
            w, sender, _, _ = _make_workload(cfg)

            await w.step(0.0)
            if w._pending_tasks:
                await asyncio.gather(*w._pending_tasks)

            assert sender.send_request.call_count == 1

        asyncio.run(_run())

    def test_step_passes_configured_max_output_length(self) -> None:
        """The configured output length reaches the sender, not its default."""

        async def _run() -> None:
            cfg = _make_config(
                num_contexts=2,
                num_permutations=1,
                max_output_length=1,
            )
            w, sender, _, _ = _make_workload(cfg)

            await w.step(0.0)
            if w._pending_tasks:
                await asyncio.gather(*w._pending_tasks)

            assert sender.send_request.await_args.kwargs["max_tokens"] == 1

        asyncio.run(_run())

    def test_step_waits_when_pending_and_no_new_requests(self) -> None:
        async def _run() -> None:
            cfg = _make_config(
                num_contexts=2,
                num_permutations=2,
                num_inflight_requests=1,
            )
            w, _, _, _ = _make_workload(cfg)

            await w.step(0.0)
            await asyncio.gather(*list(w._pending_tasks))
            await w.step(0.0)
            await asyncio.gather(*list(w._pending_tasks))
            result = await w.step(0.0)
            assert result == -1.0

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# LongDocPermutatorWorkload — on_request_finished
# ---------------------------------------------------------------------------


class TestLongDocPermutatorOnRequestFinished:
    def test_noop(self) -> None:
        w, *_ = _make_workload()
        w.on_request_finished("perm0_req1", "some output")  # must not raise


# ---------------------------------------------------------------------------
# LongDocPermutatorWorkload — full run
# ---------------------------------------------------------------------------


class TestLongDocPermutatorFullRun:
    def test_full_run_sends_all_permutations(self) -> None:
        cfg = _make_config(num_contexts=2, num_permutations=2)
        w, sender, collector, _ = _make_workload(cfg)

        w.run()

        assert sender.send_warmup_request.call_count == 1
        assert sender.send_request.call_count == len(w._permutations)
        collector.reset.assert_called_once()

    def test_full_run_with_all_permutations(self) -> None:
        # 3 contexts → 3! = 6 permutations
        cfg = _make_config(num_contexts=3, num_permutations=100)
        w, sender, collector, _ = _make_workload(cfg)

        w.run()

        expected = math.factorial(3)
        assert sender.send_request.call_count == expected

    def test_full_run_reproducible(self) -> None:
        cfg = _make_config(num_contexts=2, num_permutations=2)
        w1, s1, _, _ = _make_workload(cfg, seed=7)
        w2, s2, _, _ = _make_workload(cfg, seed=7)

        w1.run()
        w2.run()

        calls1 = [c[0][1] for c in s1.send_request.call_args_list]
        calls2 = [c[0][1] for c in s2.send_request.call_args_list]
        assert calls1 == calls2
