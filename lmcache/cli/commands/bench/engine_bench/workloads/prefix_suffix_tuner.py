# SPDX-License-Identifier: Apache-2.0
"""Prefix-suffix tuner workload for ``lmcache bench engine``.

Exercises the tiered KV-cache hierarchy (L0 HBM / L1 DRAM / L2 disk) with a
single sequential workload that can be run unchanged across three baselines:

  Baseline 1 — vanilla vLLM (L0 only).  ``--kv-cache-volume`` set to L0 size,
      ``--psf-thrash`` slightly > 1 forces every pass-2 request to miss L0.

  Baseline 2 — vLLM + LMCache L1 + L2.  ``--kv-cache-volume`` set to L1 size,
      so pass-2 requests miss L1 and hit L2 (prefix only — vanilla prefix
      caching cannot reuse the suffix because it sits behind a random
      breaker).

  Baseline 3 — vLLM + LMCache L1 + L2 + CacheBlend.  Same sizing as Baseline
      2; CacheBlend additionally allows the shared suffix's KV chunks to be
      reused, so both the prefix and the suffix hit cache.

Each request is::

    [prefix_i][random breaker][shared suffix]

with::

  - ``num_prefixes`` distinct prefixes, each starting with a unique ID so
    the prefix hash differs even if the random body collides.
  - A fresh random breaker per request, defeating ordinary prefix caching
    past the prefix boundary and preventing non-CacheBlend reuse of the
    suffix.
  - One shared suffix used by every request.

Two passes run sequentially, one request at a time, in identical order:

  - Pass 1 (warmup): populates the cache.  Stats discarded.
  - Pass 2 (measured): repeats the same prefix order.  Because LRU evicts
    the next-needed prefix on each pass-2 access, even a 1.05× overflow of
    the targeted tier is enough to ensure every pass-2 request misses that
    tier and falls through to the next one.
"""

# Standard
from dataclasses import dataclass
import random

# First Party
from lmcache.cli.commands.bench.engine_bench.progress import ProgressMonitor
from lmcache.cli.commands.bench.engine_bench.request_sender import RequestSender
from lmcache.cli.commands.bench.engine_bench.stats import StatsCollector
from lmcache.cli.commands.bench.engine_bench.workloads.base import BaseWorkload
from lmcache.logging import init_logger

logger = init_logger(__name__)

_BREAKER_TOKENS = 32
_MIN_SUFFIX_TOKENS = 100
_UNIQUE_ID_TOKENS = 4  # rough token count for ``PREFIX_<8-hex-digits>``
_MAX_OUTPUT_TOKENS = 1

# Internal multiplier on ``thrash`` GB to size the prefix pool.  Set to 1.05
# (a 5% overflow) because — with sequential pass-1/pass-2 dispatch and LRU
# eviction — even a 5% overflow is sufficient to evict the next-needed
# prefix on every pass-2 access, ensuring all pass-2 requests miss the
# targeted tier.  See ``docs/design/cli/commands/bench/engine_bench/
# bench-engine.md`` §4.5 for the analysis.
_OVERFLOW_FACTOR = 1.05


@dataclass
class PrefixSuffixTunerConfig:
    """Workload-specific config for the prefix-suffix-tuner workload.

    Attributes:
        context_length: Total tokens per request (prefix + breaker + suffix).
        prefix_ratio: Fraction of ``context_length`` allocated to the prefix.
        thrash: Size in GB of the KV-cache tier the workload should overflow
            (L0 / vLLM HBM for Baseline 1; L1 / LMCache DRAM for Baselines
            2 and 3).  The prefix pool is sized to ``thrash *
            _OVERFLOW_FACTOR`` GB internally — i.e., just barely larger than
            the targeted tier — which is sufficient under sequential
            dispatch and LRU to ensure every pass-2 request misses that
            tier.
        num_prefixes: Number of distinct prefixes generated (computed by
            :meth:`resolve` from the cache budget).
        prefix_tokens: Token length of each prefix (computed).
        suffix_tokens: Token length of the shared suffix (computed).
        breaker_tokens: Token length of the random breaker between prefix
            and suffix.
    """

    context_length: int = 8000
    prefix_ratio: float = 0.8
    thrash: float = 20.0
    num_prefixes: int = 1
    prefix_tokens: int = 1
    suffix_tokens: int = 1
    breaker_tokens: int = _BREAKER_TOKENS

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError(
                f"context_length must be positive, got {self.context_length}"
            )
        if not 0.0 < self.prefix_ratio < 1.0:
            raise ValueError(
                f"prefix_ratio must be in (0.0, 1.0), got {self.prefix_ratio}"
            )
        if self.thrash <= 0.0:
            raise ValueError(f"thrash (GB) must be positive, got {self.thrash}")
        if self.num_prefixes < 1:
            raise ValueError(f"num_prefixes must be >= 1, got {self.num_prefixes}")
        if self.prefix_tokens < 1:
            raise ValueError(f"prefix_tokens must be >= 1, got {self.prefix_tokens}")
        if self.suffix_tokens < 1:
            raise ValueError(f"suffix_tokens must be >= 1, got {self.suffix_tokens}")
        if self.breaker_tokens < 1:
            raise ValueError(f"breaker_tokens must be >= 1, got {self.breaker_tokens}")

    @classmethod
    def resolve(
        cls,
        tokens_per_gb_kvcache: int,
        context_length: int = 8000,
        prefix_ratio: float = 0.8,
        thrash: float = 20.0,
        breaker_tokens: int = _BREAKER_TOKENS,
    ) -> "PrefixSuffixTunerConfig":
        """Compute ``num_prefixes`` and token splits from the target tier size.

        ``num_prefixes`` is sized so that the total prefix-pool footprint
        in tokens equals
        ``thrash * _OVERFLOW_FACTOR * tokens_per_gb_kvcache``.  ``thrash``
        is the **size of the targeted KV-cache tier in GB**; the internal
        :data:`_OVERFLOW_FACTOR` (1.05) provides the small overflow needed
        for the LRU invariant to drive every pass-2 access to a miss of
        that tier.

        Args:
            tokens_per_gb_kvcache: Tokens fitting in 1 GB of KV cache for
                the served model (auto-detected from the engine in
                ``parse_args_to_config``; user need not supply directly).
            context_length: Total tokens per request.
            prefix_ratio: Fraction of ``context_length`` allocated to the
                prefix.  Must be strictly between 0 and 1.
            thrash: Size of the targeted KV-cache tier in GB.  Defaults
                to 20.0 GB (typical L0 size for a single H100 with low
                ``--gpu-memory-utilization``).
            breaker_tokens: Token length of the random breaker between
                prefix and suffix.  Defaults to 32.

        Returns:
            A fully-resolved PrefixSuffixTunerConfig.

        Raises:
            ValueError: If ``prefix_ratio`` leaves fewer than
                :data:`_MIN_SUFFIX_TOKENS` for the suffix, or if any field
                fails validation.
        """
        prefix_tokens = max(round(context_length * prefix_ratio), 1)
        suffix_tokens = context_length - prefix_tokens - breaker_tokens
        if suffix_tokens < _MIN_SUFFIX_TOKENS:
            raise ValueError(
                f"suffix_tokens={suffix_tokens} is below minimum "
                f"{_MIN_SUFFIX_TOKENS}; reduce prefix_ratio or "
                f"increase context_length"
            )

        pool_gb = thrash * _OVERFLOW_FACTOR
        num_prefixes = max(
            int(pool_gb * tokens_per_gb_kvcache / prefix_tokens),
            1,
        )
        logger.debug(
            "Computed num_prefixes=%d from thrash=%.2f GB (target tier), "
            "_OVERFLOW_FACTOR=%.3f -> pool=%.2f GB, "
            "tokens_per_gb_kvcache=%d, prefix_tokens=%d",
            num_prefixes,
            thrash,
            _OVERFLOW_FACTOR,
            pool_gb,
            tokens_per_gb_kvcache,
            prefix_tokens,
        )
        return cls(
            context_length=context_length,
            prefix_ratio=prefix_ratio,
            thrash=thrash,
            num_prefixes=num_prefixes,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            breaker_tokens=breaker_tokens,
        )


# ---------------------------------------------------------------------------
# Synthetic data generation (module-level helpers)
# ---------------------------------------------------------------------------


def _random_word_stream(rng: random.Random, count: int) -> str:
    """Return ``count`` space-separated random tokens drawn from *rng*.

    Uses 6-digit zero-padded random integers so each token tokenizes to
    roughly one BPE token across most tokenizers, keeping the generated
    text close to the requested token count.

    Args:
        rng: Seeded random source.
        count: Number of words to emit.

    Returns:
        A single space-joined string.
    """
    return " ".join(f"{rng.randrange(1_000_000):06d}" for _ in range(count))


def _generate_prefix(index: int, num_tokens: int, rng: random.Random) -> str:
    """Generate a prefix that begins with a unique ID.

    Args:
        index: Zero-based prefix index; encoded into the unique ID so the
            prefix's tokenized hash differs from every other prefix even
            if random bodies collide.
        num_tokens: Approximate target token length of the full prefix.
        rng: Seeded random source for the body.

    Returns:
        Prefix text starting with ``"PREFIX_<8-hex-digits>"``.
    """
    unique_id = f"PREFIX_{index:08x}"
    body_words = max(num_tokens - _UNIQUE_ID_TOKENS, 1)
    body = _random_word_stream(rng, body_words)
    return f"{unique_id} {body}"


def _generate_suffix(num_tokens: int, rng: random.Random) -> str:
    """Generate the single shared suffix used by every request.

    Args:
        num_tokens: Approximate target token length.
        rng: Seeded random source.

    Returns:
        Suffix text starting with ``"SUFFIX"``.
    """
    body = _random_word_stream(rng, max(num_tokens - 1, 1))
    return f"SUFFIX {body}"


def _generate_breaker(num_tokens: int, rng: random.Random) -> str:
    """Generate a fresh random breaker for one request.

    The breaker sits between prefix and suffix; its randomness defeats
    ordinary prefix caching past the prefix boundary and prevents
    non-CacheBlend reuse of the suffix.

    Args:
        num_tokens: Approximate target token length.
        rng: Seeded random source.  Each call advances state, so successive
            requests get different breakers.

    Returns:
        A space-joined random token string.
    """
    return _random_word_stream(rng, num_tokens)


# ---------------------------------------------------------------------------
# Workload class
# ---------------------------------------------------------------------------


class PrefixSuffixTunerWorkload(BaseWorkload):
    """Sequential two-pass workload demonstrating tiered KV-cache reuse.

    Pass 1 (executed in :meth:`warmup`) populates the cache by sending each
    prefix once.  Stats from pass 1 are discarded.  Pass 2 (executed via
    :meth:`step`) repeats the same prefix order one request at a time;
    these are the measured requests.
    """

    def __init__(
        self,
        config: PrefixSuffixTunerConfig,
        request_sender: RequestSender,
        stats_collector: StatsCollector,
        progress_monitor: ProgressMonitor,
        seed: int = 42,
    ) -> None:
        super().__init__(request_sender, stats_collector, progress_monitor)
        self._config = config
        self._seed = seed

        prefix_rng = random.Random(seed)
        self._prefixes: list[str] = [
            _generate_prefix(i, config.prefix_tokens, prefix_rng)
            for i in range(config.num_prefixes)
        ]
        suffix_rng = random.Random(seed + 1)
        self._suffix: str = _generate_suffix(config.suffix_tokens, suffix_rng)
        self._breaker_rng = random.Random(seed + 2)

        self._pass2_index = 0

    def log_config(self) -> None:
        """Log key workload config before the benchmark starts."""
        c = self._config
        B = "\033[1m"
        C = "\033[96m"
        Y = "\033[93m"
        R = "\033[0m"
        pool_tokens_millions = c.num_prefixes * c.prefix_tokens / 1_000_000
        print(
            f"{B}{'═' * 50}{R}\n"
            f"{B} Workload: {C}prefix-suffix-tuner{R}\n"
            f"{B}{'─' * 50}{R}\n"
            f"  Context length:    {Y}{c.context_length}{R} tokens\n"
            f"  Prefix tokens:     {Y}{c.prefix_tokens}{R} (ratio={c.prefix_ratio})\n"
            f"  Breaker tokens:    {Y}{c.breaker_tokens}{R} (random per request)\n"
            f"  Suffix tokens:     {Y}{c.suffix_tokens}{R} (shared, deterministic)\n"
            f"  Target tier:       {Y}{c.thrash:.2f} GB{R}"
            f" (overflow x{_OVERFLOW_FACTOR:.2f} = "
            f"{c.thrash * _OVERFLOW_FACTOR:.2f} GB)\n"
            f"  Prefix pool size:  {Y}{c.num_prefixes}{R}\n"
            f"  Pool tokens:       {Y}{pool_tokens_millions:.2f}M{R}\n"
            f"  Total measured:    {Y}{c.num_prefixes}{R} requests "
            f"(pass 2 of 2)\n"
            f"{B}{'═' * 50}{R}"
        )

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_messages(self, prefix_index: int) -> list[dict[str, str]]:
        """Build chat messages for one request.

        The breaker is freshly randomized on every call, so two requests
        for the same prefix produce different prompts past the prefix
        boundary.  Pass 1 and pass 2 therefore use different breakers
        per prefix even though the prefix order is identical.

        Args:
            prefix_index: Index into the generated prefix pool.

        Returns:
            A single-message chat list.
        """
        prefix = self._prefixes[prefix_index]
        breaker = _generate_breaker(self._config.breaker_tokens, self._breaker_rng)
        content = f"{prefix} {breaker} {self._suffix}"
        return [{"role": "user", "content": content}]

    # ------------------------------------------------------------------
    # Pass 1 — warmup
    # ------------------------------------------------------------------

    async def warmup(self) -> None:
        """Run pass 1: send each prefix once sequentially to populate cache."""
        n = self._config.num_prefixes
        self._progress_monitor.log_message(f"Pass 1 (warmup): {n} requests")
        for i in range(n):
            request_id = f"pass1_p{i}"
            messages = self._build_messages(i)
            self._progress_monitor.on_request_sent(request_id)
            self._progress_monitor.log_message(f"Pass 1 dispatched {i + 1}/{n}")
            result = await self._request_sender.send_warmup_request(
                request_id,
                messages,
            )
            if not result.successful:
                self._progress_monitor.log_message(
                    f"Pass 1 {request_id} failed: {result.error}"
                )
        self._progress_monitor.log_message(
            f"Pass 1 complete: {n} prefixes populated",
        )

    # ------------------------------------------------------------------
    # Pass 2 — measured benchmark
    # ------------------------------------------------------------------

    async def step(self, time_offset: float) -> float:
        """Send one pass-2 request inline, sequentially.

        Awaiting the request inside ``step`` enforces strict
        one-request-at-a-time dispatch.  Returns ``0.0`` for an immediate
        re-call until the prefix list is exhausted, then ``-1.0``.

        Args:
            time_offset: Seconds since pass 2 started (unused).

        Returns:
            ``0.0`` if more requests remain, ``-1.0`` when pass 2 is
            complete.
        """
        if self._pass2_index >= self._config.num_prefixes:
            return -1.0

        i = self._pass2_index
        self._pass2_index += 1
        request_id = f"pass2_p{i}"
        messages = self._build_messages(i)
        self._progress_monitor.on_request_sent(request_id)
        self._progress_monitor.log_message(
            f"Pass 2 {i + 1}/{self._config.num_prefixes}"
        )
        await self._request_sender.send_request(
            request_id,
            messages,
            max_tokens=_MAX_OUTPUT_TOKENS,
        )
        return 0.0

    def on_request_finished(self, request_id: str, output: str) -> None:
        """No-op — this workload is stateless."""
