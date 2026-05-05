# SPDX-License-Identifier: Apache-2.0
"""Tests for prefix-suffix-tuner workload config and workload generator."""

# Standard
from unittest.mock import AsyncMock, MagicMock
import time

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.stats import RequestResult
from lmcache.cli.commands.bench.engine_bench.workloads.prefix_suffix_tuner import (
    PrefixSuffixTunerConfig,
    PrefixSuffixTunerWorkload,
)

# ---------------------------------------------------------------------------
# PrefixSuffixTunerConfig — direct construction
# ---------------------------------------------------------------------------


class TestPrefixSuffixTunerConfig:
    def test_defaults(self) -> None:
        cfg = PrefixSuffixTunerConfig()
        assert cfg.context_length == 8000
        assert cfg.prefix_ratio == 0.8
        assert cfg.thrash == 1.05
        assert cfg.num_prefixes == 1
        assert cfg.prefix_tokens == 1
        assert cfg.suffix_tokens == 1
        assert cfg.breaker_tokens == 32

    def test_custom_values(self) -> None:
        cfg = PrefixSuffixTunerConfig(
            context_length=4000,
            prefix_ratio=0.5,
            thrash=2.0,
            num_prefixes=10,
            prefix_tokens=2000,
            suffix_tokens=1968,
            breaker_tokens=32,
        )
        assert cfg.context_length == 4000
        assert cfg.prefix_ratio == 0.5
        assert cfg.thrash == 2.0
        assert cfg.num_prefixes == 10

    def test_invalid_context_length(self) -> None:
        with pytest.raises(ValueError, match="context_length must be positive"):
            PrefixSuffixTunerConfig(context_length=0)

    def test_invalid_prefix_ratio_zero(self) -> None:
        with pytest.raises(ValueError, match=r"prefix_ratio must be in \(0.0, 1.0\)"):
            PrefixSuffixTunerConfig(prefix_ratio=0.0)

    def test_invalid_prefix_ratio_one(self) -> None:
        with pytest.raises(ValueError, match=r"prefix_ratio must be in \(0.0, 1.0\)"):
            PrefixSuffixTunerConfig(prefix_ratio=1.0)

    def test_invalid_prefix_ratio_negative(self) -> None:
        with pytest.raises(ValueError, match=r"prefix_ratio must be in \(0.0, 1.0\)"):
            PrefixSuffixTunerConfig(prefix_ratio=-0.5)

    def test_invalid_thrash(self) -> None:
        with pytest.raises(ValueError, match="thrash must be >= 1.0"):
            PrefixSuffixTunerConfig(thrash=0.9)

    def test_thrash_at_one_is_valid(self) -> None:
        cfg = PrefixSuffixTunerConfig(thrash=1.0)
        assert cfg.thrash == 1.0

    def test_invalid_num_prefixes(self) -> None:
        with pytest.raises(ValueError, match="num_prefixes must be >= 1"):
            PrefixSuffixTunerConfig(num_prefixes=0)

    def test_invalid_breaker_tokens(self) -> None:
        with pytest.raises(ValueError, match="breaker_tokens must be >= 1"):
            PrefixSuffixTunerConfig(breaker_tokens=0)


# ---------------------------------------------------------------------------
# PrefixSuffixTunerConfig.resolve
# ---------------------------------------------------------------------------


class TestPrefixSuffixTunerConfigResolve:
    def test_resolve_basic(self) -> None:
        cfg = PrefixSuffixTunerConfig.resolve(
            kv_cache_volume_gb=10.0,
            tokens_per_gb_kvcache=10000,
            context_length=4000,
            prefix_ratio=0.5,
            thrash=1.05,
        )
        # prefix_tokens = round(4000 * 0.5) = 2000
        assert cfg.prefix_tokens == 2000
        # suffix_tokens = 4000 - 2000 - 32 = 1968
        assert cfg.suffix_tokens == 1968
        # target = 10 * 1.05 = 10.5 GB; 10.5 * 10000 / 2000 = 52.5 → 52
        assert cfg.num_prefixes == 52

    def test_resolve_thrash_just_above_one(self) -> None:
        cfg = PrefixSuffixTunerConfig.resolve(
            kv_cache_volume_gb=100.0,
            tokens_per_gb_kvcache=50000,
            context_length=8000,
            prefix_ratio=0.8,
            thrash=1.05,
        )
        # prefix_tokens = round(8000 * 0.8) = 6400
        assert cfg.prefix_tokens == 6400
        # 100 * 1.05 * 50000 / 6400 = 820.31... → 820
        assert cfg.num_prefixes == 820

    def test_resolve_minimum_one_prefix(self) -> None:
        cfg = PrefixSuffixTunerConfig.resolve(
            kv_cache_volume_gb=0.0001,
            tokens_per_gb_kvcache=1,
            context_length=4000,
            prefix_ratio=0.5,
            thrash=1.0,
        )
        assert cfg.num_prefixes == 1

    def test_resolve_suffix_too_small(self) -> None:
        # context=200, prefix_ratio=0.95 → prefix=190, breaker=32, suffix=-22
        with pytest.raises(ValueError, match="suffix_tokens=.* below minimum 100"):
            PrefixSuffixTunerConfig.resolve(
                kv_cache_volume_gb=10.0,
                tokens_per_gb_kvcache=10000,
                context_length=200,
                prefix_ratio=0.95,
                thrash=1.05,
            )

    def test_resolve_suffix_at_minimum(self) -> None:
        # context = 200, prefix=68, breaker=32, suffix=100 (exactly minimum)
        cfg = PrefixSuffixTunerConfig.resolve(
            kv_cache_volume_gb=1.0,
            tokens_per_gb_kvcache=1000,
            context_length=200,
            prefix_ratio=0.34,
            thrash=1.0,
        )
        assert cfg.suffix_tokens == 100

    def test_resolve_thrash_scales_pool(self) -> None:
        cfg_small = PrefixSuffixTunerConfig.resolve(
            kv_cache_volume_gb=10.0,
            tokens_per_gb_kvcache=10000,
            context_length=4000,
            prefix_ratio=0.5,
            thrash=1.0,
        )
        cfg_big = PrefixSuffixTunerConfig.resolve(
            kv_cache_volume_gb=10.0,
            tokens_per_gb_kvcache=10000,
            context_length=4000,
            prefix_ratio=0.5,
            thrash=2.0,
        )
        # Doubling thrash should roughly double num_prefixes
        assert cfg_big.num_prefixes == 2 * cfg_small.num_prefixes


# ---------------------------------------------------------------------------
# PrefixSuffixTunerWorkload — helpers
# ---------------------------------------------------------------------------


def _make_workload_config(**overrides) -> PrefixSuffixTunerConfig:
    defaults = dict(
        context_length=200,
        prefix_ratio=0.5,
        thrash=1.05,
        num_prefixes=4,
        prefix_tokens=100,
        suffix_tokens=68,
        breaker_tokens=32,
    )
    defaults.update(overrides)
    return PrefixSuffixTunerConfig(**defaults)  # type: ignore[arg-type]


def _make_mock_result(request_id: str = "req_0") -> RequestResult:
    now = time.time()
    return RequestResult(
        request_id=request_id,
        successful=True,
        ttft=0.1,
        request_latency=0.5,
        num_input_tokens=100,
        num_output_tokens=1,
        decode_speed=10.0,
        submit_time=now,
        first_token_time=now + 0.1,
        finish_time=now + 0.5,
        error="",
    )


def _make_mock_sender() -> MagicMock:
    sender = MagicMock()
    sender.send_request = AsyncMock(return_value=_make_mock_result())
    sender.send_warmup_request = AsyncMock(return_value=_make_mock_result())
    return sender


def _make_workload(
    config: PrefixSuffixTunerConfig | None = None,
    seed: int = 42,
) -> tuple[PrefixSuffixTunerWorkload, MagicMock, MagicMock, MagicMock]:
    if config is None:
        config = _make_workload_config()
    sender = _make_mock_sender()
    collector = MagicMock()
    monitor = MagicMock()
    workload = PrefixSuffixTunerWorkload(
        config,
        sender,
        collector,
        monitor,
        seed=seed,
    )
    return workload, sender, collector, monitor


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


class TestPrefixSuffixTunerData:
    def test_correct_prefix_count(self) -> None:
        w, *_ = _make_workload(_make_workload_config(num_prefixes=7))
        assert len(w._prefixes) == 7

    def test_prefixes_have_unique_id_at_start(self) -> None:
        w, *_ = _make_workload(_make_workload_config(num_prefixes=3))
        assert w._prefixes[0].startswith("PREFIX_00000000")
        assert w._prefixes[1].startswith("PREFIX_00000001")
        assert w._prefixes[2].startswith("PREFIX_00000002")

    def test_prefixes_are_distinct(self) -> None:
        w, *_ = _make_workload(_make_workload_config(num_prefixes=5))
        assert len(set(w._prefixes)) == 5

    def test_single_shared_suffix(self) -> None:
        w, *_ = _make_workload()
        assert isinstance(w._suffix, str)
        assert w._suffix.startswith("SUFFIX")

    def test_data_is_deterministic_with_seed(self) -> None:
        cfg = _make_workload_config(num_prefixes=5)
        w1, *_ = _make_workload(cfg, seed=42)
        w2, *_ = _make_workload(cfg, seed=42)
        assert w1._prefixes == w2._prefixes
        assert w1._suffix == w2._suffix

    def test_data_differs_with_different_seed(self) -> None:
        cfg = _make_workload_config(num_prefixes=5)
        w1, *_ = _make_workload(cfg, seed=42)
        w2, *_ = _make_workload(cfg, seed=99)
        assert w1._prefixes != w2._prefixes
        assert w1._suffix != w2._suffix


# ---------------------------------------------------------------------------
# Message construction — request structure
# ---------------------------------------------------------------------------


class TestPrefixSuffixTunerMessages:
    def test_message_has_user_role(self) -> None:
        w, *_ = _make_workload()
        msgs = w._build_messages(0)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_message_contains_prefix(self) -> None:
        w, *_ = _make_workload()
        msgs = w._build_messages(2)
        assert "PREFIX_00000002" in msgs[0]["content"]

    def test_message_contains_shared_suffix(self) -> None:
        w, *_ = _make_workload()
        msgs0 = w._build_messages(0)
        msgs1 = w._build_messages(1)
        # Same suffix appears in both
        assert w._suffix in msgs0[0]["content"]
        assert w._suffix in msgs1[0]["content"]

    def test_breaker_differs_per_request(self) -> None:
        w, *_ = _make_workload()
        msgs_a = w._build_messages(0)
        msgs_b = w._build_messages(0)  # same prefix, different breaker
        # The two prompts must differ even for the same prefix index
        assert msgs_a[0]["content"] != msgs_b[0]["content"]

    def test_request_layout_prefix_breaker_suffix(self) -> None:
        w, *_ = _make_workload(_make_workload_config(num_prefixes=1))
        content = w._build_messages(0)[0]["content"]
        # Prefix appears before suffix
        prefix_pos = content.find("PREFIX_")
        suffix_pos = content.find("SUFFIX")
        assert prefix_pos == 0
        assert suffix_pos > prefix_pos

    def test_on_request_finished_noop(self) -> None:
        w, *_ = _make_workload()
        w.on_request_finished("some_id", "some_text")  # should not raise


# ---------------------------------------------------------------------------
# Pass 1 — warmup (async)
# ---------------------------------------------------------------------------


class TestPrefixSuffixTunerWarmup:
    @pytest.mark.asyncio
    async def test_warmup_sends_each_prefix_once(self) -> None:
        cfg = _make_workload_config(num_prefixes=4)
        w, sender, _, _ = _make_workload(cfg)

        await w.warmup()

        assert sender.send_warmup_request.call_count == 4
        for i, call in enumerate(sender.send_warmup_request.call_args_list):
            request_id = call[0][0]
            assert request_id == f"pass1_p{i}"

    @pytest.mark.asyncio
    async def test_warmup_sends_in_order(self) -> None:
        cfg = _make_workload_config(num_prefixes=3)
        w, sender, _, _ = _make_workload(cfg)

        await w.warmup()

        ids = [call[0][0] for call in sender.send_warmup_request.call_args_list]
        assert ids == ["pass1_p0", "pass1_p1", "pass1_p2"]

    @pytest.mark.asyncio
    async def test_warmup_uses_real_request_structure(self) -> None:
        cfg = _make_workload_config(num_prefixes=2)
        w, sender, _, _ = _make_workload(cfg)
        await w.warmup()

        # Pass 1 should send full prefix+breaker+suffix prompts (not a stub)
        first_call_messages = sender.send_warmup_request.call_args_list[0][0][1]
        content = first_call_messages[0]["content"]
        assert "PREFIX_00000000" in content
        assert w._suffix in content


# ---------------------------------------------------------------------------
# Pass 2 — step (async)
# ---------------------------------------------------------------------------


class TestPrefixSuffixTunerStep:
    @pytest.mark.asyncio
    async def test_step_sends_one_request_per_call(self) -> None:
        cfg = _make_workload_config(num_prefixes=2)
        w, sender, _, _ = _make_workload(cfg)

        result = await w.step(0.0)
        assert result == 0.0
        assert sender.send_request.call_count == 1

    @pytest.mark.asyncio
    async def test_step_terminates_after_pool_exhausted(self) -> None:
        cfg = _make_workload_config(num_prefixes=2)
        w, _, _, _ = _make_workload(cfg)

        await w.step(0.0)
        await w.step(0.0)
        assert (await w.step(0.0)) == -1.0

    @pytest.mark.asyncio
    async def test_step_dispatches_in_pool_order(self) -> None:
        cfg = _make_workload_config(num_prefixes=3)
        w, sender, _, _ = _make_workload(cfg)

        await w.step(0.0)
        await w.step(0.0)
        await w.step(0.0)

        ids = [call[0][0] for call in sender.send_request.call_args_list]
        assert ids == ["pass2_p0", "pass2_p1", "pass2_p2"]

    @pytest.mark.asyncio
    async def test_step_uses_max_tokens_one(self) -> None:
        cfg = _make_workload_config(num_prefixes=1)
        w, sender, _, _ = _make_workload(cfg)
        await w.step(0.0)
        # send_request called with max_tokens=1
        assert sender.send_request.call_args.kwargs["max_tokens"] == 1


# ---------------------------------------------------------------------------
# Pass ordering — pass 1 and pass 2 use same prefix order, different breakers
# ---------------------------------------------------------------------------


class TestPrefixSuffixTunerTwoPassOrdering:
    @pytest.mark.asyncio
    async def test_pass1_and_pass2_share_prefix_order(self) -> None:
        cfg = _make_workload_config(num_prefixes=4)
        w, sender, _, _ = _make_workload(cfg)

        await w.warmup()
        # Run pass 2 to exhaustion
        while True:
            r = await w.step(0.0)
            if r < 0:
                break

        warmup_prefixes = [
            c[0][1][0]["content"].split()[0]
            for c in sender.send_warmup_request.call_args_list
        ]
        bench_prefixes = [
            c[0][1][0]["content"].split()[0] for c in sender.send_request.call_args_list
        ]
        # Same prefix sequence in both passes
        assert warmup_prefixes == bench_prefixes
        assert warmup_prefixes == [f"PREFIX_{i:08x}" for i in range(4)]

    @pytest.mark.asyncio
    async def test_pass1_and_pass2_use_different_breakers(self) -> None:
        cfg = _make_workload_config(num_prefixes=2)
        w, sender, _, _ = _make_workload(cfg)

        await w.warmup()
        while True:
            r = await w.step(0.0)
            if r < 0:
                break

        # For prefix 0, the pass-1 and pass-2 prompts must differ
        # (same prefix and suffix, but different random breaker).
        pass1_prefix0 = sender.send_warmup_request.call_args_list[0][0][1][0]["content"]
        pass2_prefix0 = sender.send_request.call_args_list[0][0][1][0]["content"]
        assert pass1_prefix0 != pass2_prefix0


# ---------------------------------------------------------------------------
# Full run end-to-end
# ---------------------------------------------------------------------------


class TestPrefixSuffixTunerFullRun:
    def test_full_run(self) -> None:
        cfg = _make_workload_config(num_prefixes=3)
        w, sender, collector, _ = _make_workload(cfg)

        w.run()

        # Pass 1: 3 warmup requests
        assert sender.send_warmup_request.call_count == 3
        # Pass 2: 3 measured requests
        assert sender.send_request.call_count == 3
        # Stats reset between passes (warmup discarded)
        collector.reset.assert_called_once()
