# SPDX-License-Identifier: Apache-2.0
"""Tests for kv-tier-pressure workload config and workload."""

# Standard
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import math
import time

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.stats import RequestResult
from lmcache.cli.commands.bench.engine_bench.workloads import (
    long_doc_permutator as ldp_mod,
)
from lmcache.cli.commands.bench.engine_bench.workloads import kv_tier_pressure as ktp
from lmcache.cli.commands.bench.engine_bench.workloads.kv_tier_pressure import (
    KVTierPressureConfig,
    KVTierPressureWorkload,
    sample_requests,
    scrape_lookup_tokens,
    warmup_sweep,
)

# Local
from ..fake_tokenizer import FakeTokenizer, make_fake_tokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestKVTierPressureConfig:
    def test_defaults_require_explicit_pool(self) -> None:
        """pool_size defaults to 0, which only resolve() may supply."""
        with pytest.raises(ValueError, match="pool_size must be >= 1"):
            KVTierPressureConfig()

    def test_direct_construction(self) -> None:
        cfg = KVTierPressureConfig(pool_size=100, docs_per_request=4)
        assert cfg.pool_size == 100
        assert cfg.docs_per_request == 4
        assert cfg.context_length == 2560
        assert cfg.system_prompt_length == 256
        assert cfg.num_requests == 200
        assert cfg.access_skew == 0.0
        assert cfg.num_inflight_requests == 8
        assert cfg.max_output_length == 1

    def test_docs_per_request_may_not_exceed_pool(self) -> None:
        with pytest.raises(ValueError, match="cannot carry more"):
            KVTierPressureConfig(pool_size=4, docs_per_request=5)

    @pytest.mark.parametrize(
        "field,value,match",
        [
            ("pool_size", 0, "pool_size must be >= 1"),
            ("docs_per_request", 0, "docs_per_request must be >= 1"),
            ("context_length", 0, "context_length must be positive"),
            ("system_prompt_length", -1, "system_prompt_length must be >= 0"),
            ("num_requests", 0, "num_requests must be >= 1"),
            ("access_skew", -0.5, "access_skew must be >= 0"),
            ("vocab_size", 0, "vocab_size must be >= 1"),
            ("num_inflight_requests", 0, "num_inflight_requests must be >= 1"),
            ("max_output_length", 0, "max_output_length must be >= 1"),
        ],
    )
    def test_validation(self, field: str, value: float, match: str) -> None:
        kwargs: dict[str, float] = {"pool_size": 50, "docs_per_request": 4}
        kwargs[field] = value
        with pytest.raises(ValueError, match=match):
            KVTierPressureConfig(**kwargs)  # type: ignore[arg-type]


class TestResolve:
    def test_explicit_pool_size_wins(self) -> None:
        cfg = KVTierPressureConfig.resolve(
            pool_size=80, l1_capacity_gb=999.0, tokens_per_gb_kvcache=9999
        )
        assert cfg.pool_size == 80

    def test_derives_from_queried_l1_capacity(self) -> None:
        """2.0 x 100 GB x 1000 tok/GB / 2500 tok = 80 documents."""
        cfg = KVTierPressureConfig.resolve(
            l1_capacity_gb=100.0,
            tokens_per_gb_kvcache=1000,
            context_length=2500,
            overflow_factor=2.0,
        )
        assert cfg.pool_size == 80

    def test_overflow_factor_scales_the_pool(self) -> None:
        def pool_for(factor: float) -> int:
            return KVTierPressureConfig.resolve(
                l1_capacity_gb=100.0,
                tokens_per_gb_kvcache=1000,
                context_length=2500,
                overflow_factor=factor,
            ).pool_size

        assert pool_for(1.0) == 40
        assert pool_for(2.0) == 80
        assert pool_for(4.0) == 160

    def test_pool_is_rounded_up(self) -> None:
        cfg = KVTierPressureConfig.resolve(
            l1_capacity_gb=1.0,
            tokens_per_gb_kvcache=1000,
            context_length=300,
            overflow_factor=1.0,
            docs_per_request=1,
        )
        # ceil(1000 / 300) = 4, never 3: rounding down would under-fill L1
        assert cfg.pool_size == 4

    def test_errors_when_neither_pool_nor_capacity_available(self) -> None:
        """Silently running a pool of 1 would do no storage I/O at all."""
        with pytest.raises(ValueError, match="cannot be derived"):
            KVTierPressureConfig.resolve()

    def test_rejects_non_positive_overflow_factor(self) -> None:
        with pytest.raises(ValueError, match="overflow_factor must be positive"):
            KVTierPressureConfig.resolve(
                pool_size=10, docs_per_request=1, overflow_factor=0.0
            )


# ---------------------------------------------------------------------------
# Warm-up sweep
# ---------------------------------------------------------------------------


class TestWarmupSweep:
    def test_covers_every_document_exactly_once(self) -> None:
        groups = warmup_sweep(pool_size=20, docs_per_request=4)
        seen = [i for g in groups for i in g]
        assert sorted(seen) == list(range(20))

    def test_group_count_is_ceil(self) -> None:
        assert len(warmup_sweep(20, 4)) == 5
        assert len(warmup_sweep(21, 4)) == math.ceil(21 / 4)

    def test_backfills_final_group_to_keep_shape(self) -> None:
        """Every request keeps the same token shape, including the last."""
        groups = warmup_sweep(pool_size=10, docs_per_request=4)
        assert all(len(g) == 4 for g in groups)
        # 10 docs in groups of 4 leaves 2; the tail borrows from the front.
        assert groups[-1] == (8, 9, 0, 1)

    def test_still_covers_everything_when_backfilled(self) -> None:
        groups = warmup_sweep(pool_size=10, docs_per_request=4)
        assert set(i for g in groups for i in g) == set(range(10))

    def test_deterministic(self) -> None:
        assert warmup_sweep(37, 5) == warmup_sweep(37, 5)

    @pytest.mark.parametrize(
        "pool,docs,match",
        [
            (0, 4, "pool_size must be >= 1"),
            (10, 0, "docs_per_request must be >= 1"),
            (3, 4, "must be <="),
        ],
    )
    def test_validation(self, pool: int, docs: int, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            warmup_sweep(pool, docs)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestSampleRequests:
    def test_shape(self) -> None:
        reqs = sample_requests(100, 8, 25, 0.0, seed=42)
        assert len(reqs) == 25
        assert all(len(r) == 8 for r in reqs)

    def test_documents_within_a_request_are_distinct(self) -> None:
        for req in sample_requests(50, 10, 40, 0.0, seed=1):
            assert len(set(req)) == 10

    def test_indices_in_range(self) -> None:
        for req in sample_requests(30, 5, 50, 0.0, seed=7):
            assert all(0 <= i < 30 for i in req)

    def test_deterministic(self) -> None:
        assert sample_requests(60, 6, 30, 0.0, 42) == sample_requests(
            60, 6, 30, 0.0, 42
        )

    def test_different_seeds_differ(self) -> None:
        assert sample_requests(60, 6, 30, 0.0, 1) != sample_requests(60, 6, 30, 0.0, 2)

    def test_uniform_sampling_spreads_over_pool(self) -> None:
        """Enough uniform draws should touch essentially the whole pool."""
        reqs = sample_requests(50, 10, 200, 0.0, seed=3)
        assert len(set(i for r in reqs for i in r)) == 50

    def test_skew_concentrates_on_a_hot_subset(self) -> None:
        """Skewed sampling must favour low indices over uniform."""
        pool, docs, n = 200, 5, 300
        uniform = sample_requests(pool, docs, n, 0.0, seed=5)
        skewed = sample_requests(pool, docs, n, 1.5, seed=5)
        hot = set(range(20))
        uniform_hits = sum(1 for r in uniform for i in r if i in hot)
        skewed_hits = sum(1 for r in skewed for i in r if i in hot)
        assert skewed_hits > uniform_hits

    def test_skewed_requests_still_distinct_and_in_range(self) -> None:
        for req in sample_requests(40, 8, 30, 2.0, seed=9):
            assert len(set(req)) == 8
            assert all(0 <= i < 40 for i in req)

    @pytest.mark.parametrize(
        "pool,docs,n,skew,match",
        [
            (0, 4, 10, 0.0, "pool_size must be >= 1"),
            (10, 0, 10, 0.0, "docs_per_request must be >= 1"),
            (3, 4, 10, 0.0, "must be <="),
            (10, 4, 0, 0.0, "num_requests must be >= 1"),
            (10, 4, 10, -1.0, "access_skew must be >= 0"),
        ],
    )
    def test_validation(
        self, pool: int, docs: int, n: int, skew: float, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            sample_requests(pool, docs, n, skew, seed=0)


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------


def _make_config(**overrides: float) -> KVTierPressureConfig:
    kwargs: dict[str, float] = {
        "pool_size": 12,
        "docs_per_request": 3,
        "context_length": 20,
        "system_prompt_length": 10,
        "num_requests": 6,
        "vocab_size": 200,
        "num_inflight_requests": 2,
        "max_output_length": 1,
    }
    kwargs.update(overrides)
    return KVTierPressureConfig(**kwargs)  # type: ignore[arg-type]


def _make_mock_result(request_id: str = "req_0") -> RequestResult:
    now = time.time()
    return RequestResult(
        request_id=request_id,
        successful=True,
        ttft=0.1,
        request_latency=0.5,
        num_input_tokens=100,
        num_output_tokens=1,
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
    config: KVTierPressureConfig | None = None,
    seed: int = 42,
    tokenizer: FakeTokenizer | None = None,
) -> tuple[KVTierPressureWorkload, MagicMock, MagicMock, MagicMock]:
    if config is None:
        config = _make_config()
    if tokenizer is None:
        tokenizer = make_fake_tokenizer()
    sender = _make_mock_sender()
    collector = MagicMock()
    monitor = MagicMock()
    with patch.object(ktp, "try_load_tokenizer", return_value=tokenizer):
        workload = KVTierPressureWorkload(
            config,
            sender,
            collector,
            monitor,
            seed=seed,
            model_name="fake-model",
        )
    return workload, sender, collector, monitor


class TestWorkloadConstruction:
    def test_generates_whole_pool(self) -> None:
        w, *_ = _make_workload(_make_config(pool_size=9, docs_per_request=3))
        assert len(w._documents) == 9

    def test_documents_have_exact_token_length(self) -> None:
        tokenizer = make_fake_tokenizer()
        w, *_ = _make_workload(_make_config(context_length=25), tokenizer=tokenizer)
        for doc in w._documents:
            assert len(tokenizer.encode(doc)) == 25

    def test_system_prompt_has_exact_token_length(self) -> None:
        tokenizer = make_fake_tokenizer()
        w, *_ = _make_workload(
            _make_config(system_prompt_length=20), tokenizer=tokenizer
        )
        assert len(tokenizer.encode(w._system_prompt)) == 20

    def test_no_system_prompt_when_zero(self) -> None:
        w, *_ = _make_workload(_make_config(system_prompt_length=0))
        assert w._system_prompt == ""
        assert w._build_messages((0, 1))[0]["role"] == "user"

    def test_measured_stream_length(self) -> None:
        w, *_ = _make_workload(_make_config(num_requests=11))
        assert len(w._measured_groups) == 11

    def test_sweep_sized_from_pool(self) -> None:
        w, *_ = _make_workload(_make_config(pool_size=12, docs_per_request=3))
        assert len(w._sweep_groups) == 4

    def test_working_set_tokens(self) -> None:
        w, *_ = _make_workload(_make_config(pool_size=10, context_length=20))
        assert w.working_set_tokens == 200

    def test_raises_without_tokenizer(self) -> None:
        """A tokenizer is required: the length flags are token counts."""
        with patch.object(ktp, "try_load_tokenizer", return_value=None):
            with pytest.raises(ValueError, match="needs a tokenizer"):
                KVTierPressureWorkload(
                    _make_config(),
                    _make_mock_sender(),
                    MagicMock(),
                    MagicMock(),
                    model_name="fake-model",
                )


class TestWorkloadBehaviour:
    def test_warmup_sends_one_request_per_sweep_group(self) -> None:
        cfg = _make_config(pool_size=12, docs_per_request=3)
        w, sender, _, _ = _make_workload(cfg)
        asyncio.run(w.warmup())
        assert sender.send_warmup_request.await_count == 4

    def test_warmup_covers_every_document(self) -> None:
        cfg = _make_config(pool_size=12, docs_per_request=3)
        w, *_ = _make_workload(cfg)
        assert set(i for g in w._sweep_groups for i in g) == set(range(12))

    def test_step_dispatches_all_measured_requests(self) -> None:
        cfg = _make_config(num_requests=5)
        w, sender, _, _ = _make_workload(cfg)

        async def drive() -> None:
            while await w.step(0.0) >= 0:
                pass

        asyncio.run(drive())
        assert sender.send_request.await_count == 5

    def test_step_honours_max_output_length(self) -> None:
        cfg = _make_config(num_requests=2, max_output_length=7)
        w, sender, _, _ = _make_workload(cfg)

        async def drive() -> None:
            while await w.step(0.0) >= 0:
                pass

        asyncio.run(drive())
        assert sender.send_request.await_args.kwargs["max_tokens"] == 7

    def test_requests_carry_docs_per_request_documents(self) -> None:
        cfg = _make_config(docs_per_request=3, system_prompt_length=0)
        w, *_ = _make_workload(cfg)
        messages = w._build_messages(w._measured_groups[0])
        assert messages[-1]["content"].count("\n\n") == 2

    def test_on_request_finished_is_a_noop(self) -> None:
        """Stateless workload: recording a completion must not raise."""
        w, *_ = _make_workload()
        w.on_request_finished("sample0", "text")

    def test_log_config_runs(self) -> None:
        w, *_ = _make_workload()
        w.log_config()

    def test_extra_metric_section(self) -> None:
        cfg = _make_config(pool_size=12, docs_per_request=3)
        w, *_ = _make_workload(cfg)
        sections = w.extra_metric_sections()
        assert len(sections) == 1
        entries = dict((k, v) for k, _, v in sections[0].entries)
        assert entries["pool_size"] == 12
        assert entries["docs_per_request"] == 3
        assert "predicted_l2_read_share_pct" not in entries
        assert "overflow_factor" not in entries
        assert entries["warmup_sweep_requests"] == 4


class TestDecoupling:
    """The property the permutator cannot have: corpus size != prompt size."""

    def test_pool_grows_while_prompt_stays_fixed(self) -> None:
        small, *_ = _make_workload(_make_config(pool_size=6, docs_per_request=3))
        large, *_ = _make_workload(_make_config(pool_size=24, docs_per_request=3))
        assert large.working_set_tokens == 4 * small.working_set_tokens
        assert len(large._build_messages(large._measured_groups[0])) == len(
            small._build_messages(small._measured_groups[0])
        )
        assert len(large._measured_groups[0]) == len(small._measured_groups[0])


class TestPermutatorEquivalence:
    """At pool_size == docs_per_request the sampler reproduces the permutator.

    The permutator is the degenerate case of this workload: when every
    document in the pool goes into every request, sampling K of D with K == D
    yields a permutation of the whole corpus.  These tests pin that down so
    the sampler can stand in for the older workload.
    """

    N = 16
    CTX = 40
    SYS = 10
    REQ = 20

    def _pair(
        self,
    ) -> tuple[KVTierPressureWorkload, ldp_mod.LongDocPermutatorWorkload]:
        tokenizer = make_fake_tokenizer()
        sampler, *_ = _make_workload(
            KVTierPressureConfig(
                pool_size=self.N,
                docs_per_request=self.N,
                context_length=self.CTX,
                system_prompt_length=self.SYS,
                num_requests=self.REQ,
                vocab_size=200,
                num_inflight_requests=8,
                max_output_length=1,
            ),
            tokenizer=tokenizer,
        )
        with patch.object(ldp_mod, "try_load_tokenizer", return_value=tokenizer):
            permutator = ldp_mod.LongDocPermutatorWorkload(
                ldp_mod.LongDocPermutatorConfig(
                    num_contexts=self.N,
                    context_length=self.CTX,
                    system_prompt_length=self.SYS,
                    num_permutations=self.REQ,
                    vocab_size=200,
                    num_inflight_requests=8,
                    max_output_length=1,
                ),
                _make_mock_sender(),
                MagicMock(),
                MagicMock(),
                seed=42,
                model_name="fake-model",
            )
        return sampler, permutator

    def test_same_corpus_size(self) -> None:
        sampler, permutator = self._pair()
        assert len(sampler._documents) == len(permutator._contexts) == self.N

    def test_same_request_count(self) -> None:
        sampler, permutator = self._pair()
        assert len(sampler._measured_groups) == len(permutator._request_list)

    def test_same_tokens_per_request(self) -> None:
        """The operating point must match, or TTFT is not comparable."""
        sampler, permutator = self._pair()
        assert sampler._tokens_per_request == permutator._tokens_per_request

    def test_same_message_shape(self) -> None:
        sampler, permutator = self._pair()
        s_msgs = sampler._build_messages(sampler._measured_groups[0])
        p_msgs = permutator._request_list[0][0]
        assert [m["role"] for m in s_msgs] == [m["role"] for m in p_msgs]
        assert s_msgs[-1]["content"].count("\n\n") == p_msgs[-1]["content"].count(
            "\n\n"
        )

    def test_same_working_set(self) -> None:
        sampler, _ = self._pair()
        assert sampler.working_set_tokens == self.N * self.CTX

    def test_every_request_is_a_full_permutation(self) -> None:
        groups = sample_requests(self.N, self.N, self.REQ, 0.0, seed=42)
        assert all(sorted(g) == list(range(self.N)) for g in groups)

    def test_permutations_are_distinct_at_this_scale(self) -> None:
        """16! is astronomically larger than the request count, so repeats
        do not occur in practice -- unlike a small pool, where they would."""
        groups = sample_requests(self.N, self.N, self.REQ, 0.0, seed=42)
        assert len(set(groups)) == self.REQ

    def test_warmup_primes_the_whole_corpus_in_one_request(self) -> None:
        sampler, _ = self._pair()
        assert len(sampler._sweep_groups) == 1
        assert sorted(sampler._sweep_groups[0]) == list(range(self.N))


class TestScrapeLookupTokens:
    """Parsing the counters that reveal whether the cache is being reused."""

    BODY = (
        "# HELP lmcache_mp_lookup_hit_tokens_total hits\n"
        "# TYPE lmcache_mp_lookup_hit_tokens_total counter\n"
        "lmcache_mp_lookup_hit_tokens_total 1234.0\n"
        'lmcache_mp_lookup_requested_tokens_total{instance="a"} 4000.0\n'
        'lmcache_mp_lookup_requested_tokens_total{instance="b"} 1000.0\n'
        "lmcache_mp_l1_read_chunks_total 99.0\n"
    )

    def _scrape(self, body: str) -> dict[str, float]:
        payload = MagicMock()
        payload.read.return_value = body.encode()
        payload.__enter__ = lambda self_: payload
        payload.__exit__ = lambda self_, *a: False
        with patch.object(ktp.urllib.request, "urlopen", return_value=payload):
            return scrape_lookup_tokens("http://localhost:8080")

    def test_reads_both_counters(self) -> None:
        totals = self._scrape(self.BODY)
        assert totals["hit"] == 1234.0

    def test_sums_labelled_series(self) -> None:
        """Counters arrive per-instance; the total is the sum."""
        assert self._scrape(self.BODY)["requested"] == 5000.0

    def test_ignores_unrelated_counters(self) -> None:
        totals = self._scrape(self.BODY)
        assert set(totals) == {"hit", "requested"}

    def test_unreachable_server_degrades_quietly(self) -> None:
        """A missing metrics endpoint must not fail the benchmark."""
        with patch.object(
            ktp.urllib.request, "urlopen", side_effect=OSError("refused")
        ):
            assert scrape_lookup_tokens("http://localhost:9999") == {
                "hit": 0.0,
                "requested": 0.0,
            }

    def test_absent_counters_give_zero(self) -> None:
        assert self._scrape("lmcache_mp_l1_read_chunks_total 7.0\n") == {
            "hit": 0.0,
            "requested": 0.0,
        }


class TestHitTokenReporting:
    """The metric that distinguishes a reusing cache from a broken one."""

    def test_absent_without_an_lmcache_url(self) -> None:
        w, *_ = _make_workload()
        keys = {k for k, _, _ in w.extra_metric_sections()[0].entries}
        assert "hit_tokens_per_request" not in keys

    def test_reported_as_a_measured_phase_delta(self) -> None:
        """Warm-up stores must not count toward the measured hit rate."""
        cfg = _make_config(pool_size=12, docs_per_request=3, num_requests=10)
        tokenizer = make_fake_tokenizer()
        sender = _make_mock_sender()
        with patch.object(ktp, "try_load_tokenizer", return_value=tokenizer):
            w = KVTierPressureWorkload(
                cfg,
                sender,
                MagicMock(),
                MagicMock(),
                seed=42,
                model_name="fake-model",
                lmcache_url="http://localhost:8080",
            )
        # boundary snapshot, then end-of-run totals
        with patch.object(
            ktp,
            "scrape_lookup_tokens",
            side_effect=[
                {"hit": 1000.0, "requested": 2000.0},
                {"hit": 1800.0, "requested": 4000.0},
            ],
        ):
            asyncio.run(w.warmup())
            entries = dict((k, v) for k, _, v in w.extra_metric_sections()[0].entries)
        assert entries["hit_tokens_total"] == 800
        assert entries["requested_tokens_total"] == 2000
        assert entries["hit_tokens_per_request"] == pytest.approx(80.0)
        assert entries["hit_token_rate_pct"] == pytest.approx(40.0)
