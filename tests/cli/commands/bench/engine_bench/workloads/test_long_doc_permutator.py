# SPDX-License-Identifier: Apache-2.0
"""Tests for long-doc-permutator workload config and workload."""

# Standard
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import hashlib
import json
import math
import time

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.config import WarmupPolicy
from lmcache.cli.commands.bench.engine_bench.stats import RequestResult
from lmcache.cli.commands.bench.engine_bench.workloads import kv_tier_pressure as ktp
from lmcache.cli.commands.bench.engine_bench.workloads.long_doc_permutator import (
    LongDocPermutatorConfig,
    LongDocPermutatorWorkload,
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
# Shared test fixtures
# ---------------------------------------------------------------------------


def _make_config(**overrides: int) -> LongDocPermutatorConfig:
    defaults = dict(
        num_contexts=3,
        context_length=20,
        system_prompt_length=10,
        num_permutations=4,
        vocab_size=100,
        num_inflight_requests=1,
    )
    defaults.update(overrides)
    return LongDocPermutatorConfig(**defaults)


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
    with patch.object(ktp, "try_load_tokenizer", return_value=tokenizer):
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
    def test_raises_without_tokenizer(self) -> None:
        with patch.object(ktp, "try_load_tokenizer", return_value=None):
            with pytest.raises(ValueError, match="needs a tokenizer"):
                LongDocPermutatorWorkload(
                    _make_config(), _make_mock_sender(), MagicMock(), MagicMock()
                )

    @pytest.mark.parametrize(
        "contexts,requested,actual",
        [
            (1, 100, 1),
            (2, 100, 2),
            (3, 4, 4),
            (3, 100, 6),
            (5, 100, 100),
            (5, 8, 8),
            (16, 20, 20),
        ],
    )
    def test_distinct_requests_capped_at_factorial(
        self, contexts: int, requested: int, actual: int
    ) -> None:
        workload, sender, _, _ = _make_workload(
            _make_config(num_contexts=contexts, num_permutations=requested)
        )
        assert workload.num_requests == actual
        workload.run()
        bodies = [
            call.args[1][-1]["content"] for call in sender.send_request.call_args_list
        ]
        assert len(bodies) == actual
        assert len(set(bodies)) == actual

    @pytest.mark.parametrize("system_length", [0, 10])
    def test_prompts_have_exact_token_lengths_and_all_documents(
        self, system_length: int
    ) -> None:
        tokenizer = make_fake_tokenizer()
        workload, sender, _, _ = _make_workload(
            _make_config(
                num_contexts=3, context_length=25, system_prompt_length=system_length
            ),
            tokenizer=tokenizer,
        )
        workload.run()
        prompts = [call.args[1] for call in sender.send_request.call_args_list]
        corpus = sorted(prompts[0][-1]["content"].split("\n\n"))
        assert len(corpus) == 3
        for messages in prompts:
            assert [m["role"] for m in messages] == (
                ["system", "user"] if system_length else ["user"]
            )
            assert sorted(messages[-1]["content"].split("\n\n")) == corpus
            assert all(len(tokenizer.encode(doc)) == 25 for doc in corpus)
            if system_length:
                assert len(tokenizer.encode(messages[0]["content"])) == system_length
            assert (
                sum(len(tokenizer.encode(m["content"])) for m in messages)
                == workload.tokens_per_request
            )

    def test_different_seeds_produce_different_prompts(self) -> None:
        first, first_sender, _, _ = _make_workload(seed=7)
        second, second_sender, _, _ = _make_workload(seed=8)
        first.run()
        second.run()
        assert (
            first_sender.send_request.call_args_list
            != second_sender.send_request.call_args_list
        )

    def test_legacy_report_has_no_tier_pressure_section(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        workload, _, _, _ = _make_workload(
            _make_config(num_contexts=2, num_permutations=100)
        )
        workload.log_config()
        output = capsys.readouterr().out
        assert "long-doc-permutator" in output
        assert workload.num_requests == 2
        assert workload.extra_metric_sections() == []


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


async def _drive_steps(workload: LongDocPermutatorWorkload) -> None:
    """Drive the public step interface until all requests finish."""
    while await workload.step(0.0) >= 0:
        await asyncio.sleep(0)


class TestLongDocPermutatorStep:
    def test_first_step_dispatches_one_and_returns_zero(self) -> None:
        async def run() -> None:
            workload, sender, _, _ = _make_workload(_make_config(num_permutations=1))
            assert await workload.step(0.0) == 0.0
            await asyncio.wait_for(_drive_steps(workload), timeout=2.0)
            assert sender.send_request.await_count == 1
            assert sender.send_request.await_args.args[0] == "perm0_req0"

        asyncio.run(run())

    def test_step_passes_configured_max_output_length(self) -> None:
        workload, sender, _, _ = _make_workload(_make_config(max_output_length=1))
        asyncio.run(_drive_steps(workload))
        assert all(
            call.kwargs["max_tokens"] == 1
            for call in sender.send_request.await_args_list
        )

    @pytest.mark.parametrize(
        "failure", [None, RuntimeError("connection lost"), asyncio.CancelledError()]
    )
    def test_concurrency_and_completion_after_failed_or_cancelled_request(
        self, failure: BaseException | None
    ) -> None:
        async def run() -> None:
            workload, sender, _, monitor = _make_workload(
                _make_config(
                    num_contexts=3, num_permutations=6, num_inflight_requests=2
                )
            )
            active = 0
            peak = 0
            completed = 0

            async def send(
                request_id: str, messages: list[dict[str, str]], max_tokens: int
            ) -> RequestResult:
                nonlocal active, peak, completed
                active += 1
                peak = max(peak, active)
                try:
                    await asyncio.sleep(0.001)
                    if request_id == "perm0_req0" and failure is not None:
                        raise failure
                    return _make_mock_result(request_id)
                finally:
                    active -= 1
                    completed += 1

            sender.send_request.side_effect = send
            await asyncio.wait_for(_drive_steps(workload), timeout=2.0)
            assert sender.send_request.await_count == completed == 6
            assert active == 0
            assert peak == 2
            if isinstance(failure, RuntimeError):
                assert any(
                    "Dispatch task failed" in call.args[0]
                    for call in monitor.log_message.call_args_list
                )

        asyncio.run(run())


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
    def test_no_warmup_keeps_measured_requests_and_closes_sender(self) -> None:
        workload, sender, collector, _ = _make_workload()
        workload.run(WarmupPolicy.SKIP)
        sender.send_warmup_request.assert_not_awaited()
        assert sender.send_request.await_count == workload.num_requests
        sender.close.assert_awaited_once()
        collector.reset.assert_called_once()

    @pytest.mark.parametrize(
        "contexts,requests,system_length,seed,expected",
        [
            (
                5,
                100,
                10,
                42,
                "31a7001f23803f9efa4cad4cd894f1e8200446e639049cea86a475670345929a",
            ),
            (
                3,
                100,
                0,
                7,
                "c774aa724684f29bd179bc0bf48bc7b6fb358befa382445661de5411b42695e1",
            ),
            (
                1,
                5,
                10,
                42,
                "33ada7c4a0b98d71f2691002ea3f02e1a7579f5e0eb84994a83fd0c44f7b8eb1",
            ),
            (
                5,
                8,
                10,
                7,
                "a2dfebef3975d887b0f4a538b199129351bc56bef911067ce62c2fe9e72131c1",
            ),
            (
                16,
                20,
                10,
                42,
                "5209e0b0479bd68ab282a8c43d916bd564d9a9533c367862d1db50d9e6e24fcf",
            ),
        ],
    )
    def test_preserves_seeded_request_stream(
        self, contexts: int, requests: int, system_length: int, seed: int, expected: str
    ) -> None:
        """Pin IDs, payloads, order and output limits before consolidation."""
        cfg = _make_config(
            num_contexts=contexts,
            num_permutations=requests,
            system_prompt_length=system_length,
        )
        workload, sender, _, _ = _make_workload(cfg, seed=seed)
        workload.run()
        records = [
            (phase, call.args, call.kwargs)
            for phase, calls in [
                ("warmup", sender.send_warmup_request.call_args_list),
                ("measured", sender.send_request.call_args_list),
            ]
            for call in calls
        ]
        assert (
            hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
            == expected
        )

    def test_full_run_sends_all_permutations(self) -> None:
        cfg = _make_config(num_contexts=2, num_permutations=2)
        w, sender, collector, _ = _make_workload(cfg)

        w.run()

        assert sender.send_warmup_request.call_count == 1
        assert sender.send_request.call_count == w.num_requests
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
