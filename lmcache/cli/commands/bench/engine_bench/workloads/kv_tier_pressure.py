# SPDX-License-Identifier: Apache-2.0
"""KV tier-pressure workload for ``lmcache bench engine``.

Drives the KV working set past L1 so the L2 storage tier is genuinely read.

Why this exists
---------------
``long-doc-permutator`` puts EVERY document in EVERY request, so the corpus
size and the prompt size are the same number.  The distinct working set is
therefore fixed at ``num_contexts x context_length`` tokens no matter how
many permutations are sent -- ``num_permutations`` adds requests, never new
bytes.  Since L1 must hold at least one request per in-flight slot just to
run, the working set is always ``1 / num_inflight_requests`` of the smallest
usable L1, and **no setting of any flag can make that workload spill to L2**.
That is an algebraic property of the permutation, not a tuning problem.

This workload decouples the two.  A pool of ``pool_size`` documents is
generated once; each request carries ``docs_per_request`` of them, sampled
per request.  The prompt keeps the same shape as the permutator's -- system
prompt followed by concatenated documents -- so TTFT and prefill stay
comparable, while the working set scales with ``pool_size`` independently of
how large any single prompt is.

Sizing
------
``pool_size`` is the knob that decides whether the storage tier is reached at
all: the distinct working set is ``pool_size x context_length`` tokens, and
only what exceeds L1 can be evicted and read back.  To overflow a cache of
``V`` GB by a factor of ``F``::

    pool_size = ceil(F x V x tokens_per_gb_kvcache / context_length)

``tokens_per_gb_kvcache`` comes from the engine's own KV layout -- ``lmcache
bench engine`` resolves it from ``--lmcache-url``, and it is
``(1024 ** 3) // (cache_size_per_token x world_size)``.  An ``F`` of 2 means
half the working set cannot be resident, so roughly half of all cache reads
must come from L2.

Treat that only as a dial.  The idealised share ``1 - 1 / F`` assumes pure
LRU over uniformly-drawn whole documents and ignores prefetch re-admission,
chunk-level sharing and intra-request re-reads: 45% was measured at ``F = 1``
on one stack and 36% at ``F = 2`` on another.  Size the pool with it, then
read the share that actually occurred from the cache's tier counters.

Requires blending
-----------------
This workload assumes **CacheBlend** (``enable_blending``).  Each request
concatenates a random subset of the pool in random order, so a document sits
at a different offset in almost every prompt it appears in.  Without
blending, cache keys are prefix-chained and a connector can only load a
contiguous hit prefix, so a document is reused only when everything before it
in the prompt also matches -- which these prompts almost never satisfy.  The
result is a run that writes a great deal to the storage tier and reads very
little back, while the *share* of reads served by that tier still looks
healthy.

That share cannot detect the problem: it is a ratio between two tiers and
reads much the same whether the cache is serving most of each prompt or
almost none of it.  Watch **cache-hit tokens per request** instead, which the
workload reports whenever ``--lmcache-url`` is given.

Without blending, use ``docs_per_request=1``.  Each prompt is then
``[system prompt][document]``, which is prefix-stable, so repeat draws of the
same document hit cache normally.

Warm-up
-------
Warm-up is a DETERMINISTIC SWEEP, not a dummy request, for two reasons:

  1. Until L1 fills there is no eviction, no L2 write and therefore no L2
     read.  A read share measured across a cold start is a function of run
     length, not a property of the system.
  2. Random sampling covers the pool far too slowly.  The coupon-collector
     cost to touch every document once is ``(D / K) x ln(D)``, which at
     D=1,820 and K=16 is ~855 requests -- a 200-request run would never
     finish warming up.

The sweep partitions the pool into ``ceil(pool_size / docs_per_request)``
non-overlapping groups and sends one request per group, so every document is
stored exactly once before measurement starts.
"""

# Standard
from dataclasses import dataclass
import asyncio
import math
import random
import urllib.error
import urllib.request

# First Party
from lmcache.cli.commands.bench.engine_bench.progress import ProgressMonitor
from lmcache.cli.commands.bench.engine_bench.request_sender import RequestSender
from lmcache.cli.commands.bench.engine_bench.stats import StatsCollector
from lmcache.cli.commands.bench.engine_bench.tokenizers import (
    TokenPool,
    build_single_token_pool,
    try_load_tokenizer,
)
from lmcache.cli.commands.bench.engine_bench.workloads.base import (
    BaseWorkload,
    MetricSection,
)
from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass
class KVTierPressureConfig:
    """Workload-specific config for the kv-tier-pressure workload."""

    pool_size: int = 0
    docs_per_request: int = 16
    context_length: int = 2560
    system_prompt_length: int = 256
    num_requests: int = 200
    overflow_factor: float = 2.0
    access_skew: float = 0.0
    vocab_size: int = 8000
    num_inflight_requests: int = 8
    max_output_length: int = 1

    def __post_init__(self) -> None:
        """Validate the resolved configuration.

        Raises:
            ValueError: If any field is outside its permitted range, or if
                ``docs_per_request`` exceeds ``pool_size``.
        """
        if self.pool_size < 1:
            raise ValueError(
                f"pool_size must be >= 1, got {self.pool_size}. Pass "
                f"--ktp-pool-size: it sets the working set "
                f"(pool_size x context_length tokens) and therefore whether "
                f"the storage tier is reached at all."
            )
        if self.docs_per_request < 1:
            raise ValueError(
                f"docs_per_request must be >= 1, got {self.docs_per_request}"
            )
        if self.docs_per_request > self.pool_size:
            raise ValueError(
                f"docs_per_request ({self.docs_per_request}) must be <= "
                f"pool_size ({self.pool_size}): a request cannot carry more "
                f"distinct documents than the pool holds"
            )
        if self.context_length <= 0:
            raise ValueError(
                f"context_length must be positive, got {self.context_length}"
            )
        if self.system_prompt_length < 0:
            raise ValueError(
                f"system_prompt_length must be >= 0, got {self.system_prompt_length}"
            )
        if self.num_requests < 1:
            raise ValueError(f"num_requests must be >= 1, got {self.num_requests}")
        if self.overflow_factor <= 0:
            raise ValueError(
                f"overflow_factor must be positive, got {self.overflow_factor}"
            )
        if self.access_skew < 0:
            raise ValueError(f"access_skew must be >= 0, got {self.access_skew}")
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {self.vocab_size}")
        if self.num_inflight_requests < 1:
            raise ValueError(
                f"num_inflight_requests must be >= 1, got {self.num_inflight_requests}"
            )
        if self.max_output_length < 1:
            raise ValueError(
                f"max_output_length must be >= 1, got {self.max_output_length}"
            )

    @classmethod
    def resolve(
        cls,
        pool_size: int = 0,
        l1_capacity_gb: float = 0.0,
        tokens_per_gb_kvcache: int = 0,
        overflow_factor: float = 2.0,
        docs_per_request: int = 16,
        context_length: int = 2560,
        system_prompt_length: int = 256,
        num_requests: int = 200,
        access_skew: float = 0.0,
        vocab_size: int = 8000,
        num_inflight_requests: int = 8,
        max_output_length: int = 1,
    ) -> "KVTierPressureConfig":
        """Create a config, deriving ``pool_size`` when it is not given.

        A pool size of 0 means "size the corpus against the cache": the
        working set is set to ``overflow_factor`` times the server's actual
        L1 capacity, which is what makes the storage tier reachable without
        the caller converting GB to tokens to documents by hand.

        Args:
            pool_size: Total documents in the corpus.  0 derives it from
                ``l1_capacity_gb`` and ``overflow_factor``.
            l1_capacity_gb: Host-memory capacity reported by the LMCache
                server.  Only consulted when ``pool_size`` is 0.
            tokens_per_gb_kvcache: Tokens fitting in 1 GB of KV cache.  Only
                consulted when ``pool_size`` is 0.
            overflow_factor: Working set as a multiple of L1.  Values above
                1.0 force eviction to the storage tier.
            docs_per_request: Documents sampled into each request.  Bounded
                above by the engine's context limit, not by ``pool_size``.
            context_length: Exact token length of each document.
            system_prompt_length: Exact token length of the shared system
                prompt.  Use 0 for no system prompt.
            num_requests: Number of measured requests to send.
            access_skew: Zipf exponent for document popularity.  0.0 samples
                uniformly; larger values concentrate reads on a hot subset
                and reduce the L2 read share.
            vocab_size: Number of distinct single-token words to sample
                document content from.
            num_inflight_requests: Max concurrent in-flight requests.
            max_output_length: Max tokens to generate per request.  The
                default of 1 isolates prefill, which is what the cache tier
                affects; larger values add decode time that dilutes the
                measurement.

        Returns:
            A fully-resolved KVTierPressureConfig.

        Raises:
            ValueError: If any value fails validation.
        """
        resolved_pool_size = pool_size
        if resolved_pool_size < 1:
            if l1_capacity_gb <= 0 or tokens_per_gb_kvcache <= 0:
                raise ValueError(
                    "pool_size was not given and cannot be derived: no L1 "
                    "capacity was available from the LMCache server. Pass "
                    "--ktp-pool-size explicitly, or point --lmcache-url at a "
                    "running server."
                )
            if context_length <= 0:
                raise ValueError(
                    f"context_length must be positive to derive pool_size, "
                    f"got {context_length}"
                )
            budget_tokens = overflow_factor * l1_capacity_gb * tokens_per_gb_kvcache
            resolved_pool_size = max(math.ceil(budget_tokens / context_length), 1)
            logger.debug(
                "Derived pool_size=%d from l1_capacity_gb=%.2f, "
                "tokens_per_gb_kvcache=%d, context_length=%d, "
                "overflow_factor=%.2f",
                resolved_pool_size,
                l1_capacity_gb,
                tokens_per_gb_kvcache,
                context_length,
                overflow_factor,
            )
        return cls(
            pool_size=resolved_pool_size,
            overflow_factor=overflow_factor,
            docs_per_request=docs_per_request,
            context_length=context_length,
            system_prompt_length=system_prompt_length,
            num_requests=num_requests,
            access_skew=access_skew,
            vocab_size=vocab_size,
            num_inflight_requests=num_inflight_requests,
            max_output_length=max_output_length,
        )


# ---------------------------------------------------------------------------
# Prompt generation helpers (module-level, before classes)
# ---------------------------------------------------------------------------


def _generate_text(length: int, pool: TokenPool, seed: int) -> str:
    """Generate text of exactly ``length`` tokens from ``pool``.

    Args:
        length: Exact number of tokens the text must encode to.
        pool: Single-token word pool and its joining convention.
        seed: Random seed; the same seed yields the same text.

    Returns:
        Text encoding to exactly ``length`` tokens, or "" when length is 0.
    """
    if length == 0:
        return ""
    rng = random.Random(seed)
    return pool.join([rng.choice(pool.words) for _ in range(length)])


def _generate_pool(
    pool_size: int,
    length: int,
    pool: TokenPool,
    seed: int,
) -> list[str]:
    """Generate ``pool_size`` documents of exactly ``length`` tokens each.

    Each document gets its own seed so the corpus is reproducible and the
    documents share no content beyond what the vocabulary forces.

    Args:
        pool_size: Number of documents to generate.
        length: Exact token length of each document.
        pool: Single-token word pool and its joining convention.
        seed: Base random seed; document ``i`` uses ``seed + i``.

    Returns:
        List of ``pool_size`` document strings.
    """
    return [_generate_text(length, pool, seed + i) for i in range(pool_size)]


def warmup_sweep(pool_size: int, docs_per_request: int) -> list[tuple[int, ...]]:
    """Partition the pool so every document is stored exactly once.

    Deterministic by construction.  The final group is back-filled from the
    front of the pool when ``pool_size`` is not a multiple of
    ``docs_per_request``, so every warm-up request keeps the same token shape
    as a measured one; those few repeats are harmless because the sweep is
    excluded from the measurement.

    Args:
        pool_size: Total documents in the corpus.
        docs_per_request: Documents carried by each request.

    Returns:
        List of document-index tuples, one per warm-up request.

    Raises:
        ValueError: If either argument is less than 1, or if
            ``docs_per_request`` exceeds ``pool_size``.
    """
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size}")
    if docs_per_request < 1:
        raise ValueError(f"docs_per_request must be >= 1, got {docs_per_request}")
    if docs_per_request > pool_size:
        raise ValueError(
            f"docs_per_request ({docs_per_request}) must be <= pool_size ({pool_size})"
        )
    indices = list(range(pool_size))
    groups: list[tuple[int, ...]] = []
    for start in range(0, pool_size, docs_per_request):
        group = indices[start : start + docs_per_request]
        if len(group) < docs_per_request:
            group = group + indices[: docs_per_request - len(group)]
        groups.append(tuple(group))
    return groups


def sample_requests(
    pool_size: int,
    docs_per_request: int,
    num_requests: int,
    access_skew: float,
    seed: int,
) -> list[tuple[int, ...]]:
    """Draw the measured request stream from the document pool.

    Each request samples ``docs_per_request`` distinct documents.  With
    ``access_skew`` at 0.0 every document is equally likely; larger values
    apply Zipf weights ``1 / (rank + 1) ** access_skew`` so a hot subset is
    drawn more often, which raises the L1 hit rate and lowers the L2 share.

    Args:
        pool_size: Total documents in the corpus.
        docs_per_request: Distinct documents per request.
        num_requests: Number of requests to generate.
        access_skew: Zipf exponent; 0.0 is uniform.
        seed: Random seed; the same seed yields the same stream.

    Returns:
        List of ``num_requests`` document-index tuples.

    Raises:
        ValueError: If any argument is outside its permitted range.
    """
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size}")
    if docs_per_request < 1:
        raise ValueError(f"docs_per_request must be >= 1, got {docs_per_request}")
    if docs_per_request > pool_size:
        raise ValueError(
            f"docs_per_request ({docs_per_request}) must be <= pool_size ({pool_size})"
        )
    if num_requests < 1:
        raise ValueError(f"num_requests must be >= 1, got {num_requests}")
    if access_skew < 0:
        raise ValueError(f"access_skew must be >= 0, got {access_skew}")

    rng = random.Random(seed)
    indices = list(range(pool_size))
    if access_skew == 0.0:
        return [
            tuple(rng.sample(indices, docs_per_request)) for _ in range(num_requests)
        ]

    weights = [1.0 / ((rank + 1) ** access_skew) for rank in range(pool_size)]
    requests: list[tuple[int, ...]] = []
    for _ in range(num_requests):
        remaining = list(indices)
        remaining_weights = list(weights)
        picked: list[int] = []
        for _ in range(docs_per_request):
            choice = rng.choices(range(len(remaining)), weights=remaining_weights)[0]
            picked.append(remaining.pop(choice))
            remaining_weights.pop(choice)
        requests.append(tuple(picked))
    return requests


# Counters that reveal whether the cache is actually being reused.  The L2
# share of reads cannot do this on its own: it is a ratio between two tiers,
# so it reads the same whether the cache is serving most of each prompt or
# almost none of it.  Hit tokens per request is the figure that separates
# those two cases.
_HIT_TOKENS = "lmcache_mp_lookup_hit_tokens_total"
_REQUESTED_TOKENS = "lmcache_mp_lookup_requested_tokens_total"


def scrape_lookup_tokens(metrics_url: str) -> dict[str, float]:
    """Read the lookup hit/requested token counters from an LMCache server.

    Args:
        metrics_url: Base URL of the LMCache HTTP server, or its ``/metrics``
            endpoint directly.

    Returns:
        Mapping with ``hit`` and ``requested`` totals.  Both are 0.0 when the
        server is unreachable or does not expose the counters, so a missing
        metrics endpoint degrades the report rather than failing the run.
    """
    url = metrics_url.rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    if not url.endswith("/metrics"):
        url = f"{url}/metrics"
    totals = {"hit": 0.0, "requested": 0.0}
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = resp.read().decode()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.debug("Could not scrape %s: %s", url, exc)
        return totals
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        base = name.split("{", 1)[0]
        key = (
            "hit"
            if base == _HIT_TOKENS
            else ("requested" if base == _REQUESTED_TOKENS else "")
        )
        if not key:
            continue
        try:
            totals[key] += float(value)
        except ValueError:
            continue
    return totals


# ---------------------------------------------------------------------------
# Workload class
# ---------------------------------------------------------------------------


class KVTierPressureWorkload(BaseWorkload):
    """Workload that samples documents from a pool larger than the cache.

    Generates a document pool sized to overflow L1, sweeps it once so every
    document is resident before measurement, then sends requests that each
    carry a sampled subset.  The overflow is what makes L2 reads happen, and
    the sweep is what makes the measured share a steady-state property rather
    than an artefact of run length.
    """

    def __init__(
        self,
        config: KVTierPressureConfig,
        request_sender: RequestSender,
        stats_collector: StatsCollector,
        progress_monitor: ProgressMonitor,
        seed: int = 42,
        model_name: str | None = None,
        lmcache_url: str = "",
    ) -> None:
        """Build the document pool, the warm-up sweep and the request stream.

        Args:
            config: Fully-resolved workload config.
            request_sender: Shared request sender instance.
            stats_collector: Shared stats collector instance.
            progress_monitor: Shared progress monitor instance.
            seed: Random seed for corpus generation and sampling.
            model_name: Model whose tokenizer sizes the documents.  Omitted
                means auto-detected from the engine.
            lmcache_url: LMCache HTTP server, used to read cache-hit counters
                across the measured phase.  Empty disables that reporting.

        Raises:
            ValueError: If no tokenizer can be loaded for ``model_name``.
        """
        super().__init__(request_sender, stats_collector, progress_monitor)
        self._config = config
        self._seed = seed

        # The configured lengths are exact token counts, which is only
        # meaningful against the tokenizer the engine actually uses.  Without
        # one this workload would silently emit prompts several times larger
        # than requested, so refuse to run rather than produce numbers that
        # describe a different operating point than the flags claim.
        tokenizer = try_load_tokenizer(model_name)
        if tokenizer is None:
            raise ValueError(
                "kv-tier-pressure needs a tokenizer to size its documents "
                f"in tokens, but none could be loaded for model {model_name!r} "
                "(auto-detected from the engine when --model is omitted). "
                "Pass --model with a HuggingFace repo ID or a local path, "
                "e.g. --model openai/gpt-oss-20b, and make sure transformers "
                "is installed."
            )

        pool = build_single_token_pool(tokenizer, config.vocab_size, seed=seed)
        self._system_prompt = _generate_text(
            config.system_prompt_length, pool, seed + 2
        )
        self._documents = _generate_pool(
            config.pool_size, config.context_length, pool, seed + 1
        )
        self._sweep_groups = warmup_sweep(config.pool_size, config.docs_per_request)
        self._measured_groups = sample_requests(
            config.pool_size,
            config.docs_per_request,
            config.num_requests,
            config.access_skew,
            seed,
        )

        # Measured rather than derived: the per-message totals are exact by
        # construction, but joining documents adds separator tokens whose
        # count is tokenizer-specific.
        self._tokens_per_request = sum(
            len(tokenizer.encode(m["content"], add_special_tokens=False))
            for m in self._build_messages(self._measured_groups[0])
        )

        self._semaphore = asyncio.Semaphore(config.num_inflight_requests)
        self._pending_tasks: set[asyncio.Task] = set()
        self._request_index = 0
        self._lmcache_url = lmcache_url
        self._lookup_at_boundary: dict[str, float] = {"hit": 0.0, "requested": 0.0}

        logger.debug(
            "KVTierPressure: pool=%d docs, %d per request, %d sweep + "
            "%d measured requests",
            config.pool_size,
            config.docs_per_request,
            len(self._sweep_groups),
            len(self._measured_groups),
        )

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def working_set_tokens(self) -> int:
        """Total distinct tokens in the document pool, excluding the prompt."""
        return self._config.pool_size * self._config.context_length

    def log_config(self) -> None:
        """Log key workload config before the benchmark starts."""
        c = self._config
        bold = "\033[1m"
        cyan = "\033[96m"
        yellow = "\033[93m"
        reset = "\033[0m"
        passes = c.num_requests * c.docs_per_request / c.pool_size
        print(
            f"{bold}{'═' * 50}{reset}\n"
            f"{bold} Workload: {cyan}kv-tier-pressure{reset}\n"
            f"{bold}{'─' * 50}{reset}\n"
            f"  Document pool:       {yellow}{c.pool_size}{reset} documents\n"
            f"  Document length:     {yellow}{c.context_length}{reset} tokens\n"
            f"  Docs per request:    {yellow}{c.docs_per_request}{reset}\n"
            f"  System prompt:       {yellow}{c.system_prompt_length}{reset} tokens\n"
            f"  Working set:         {yellow}{self.working_set_tokens:,}{reset}"
            f" tokens\n"
            f"  Access skew:         {yellow}{c.access_skew:.2f}{reset} "
            f"(0 = uniform)\n"
            f"  Warm-up sweep:       {yellow}{len(self._sweep_groups)}{reset} "
            f"requests (every document once)\n"
            f"  Measured requests:   {yellow}{c.num_requests}{reset} "
            f"({passes:.1f} passes over the pool)\n"
            f"  Tokens per request:  {yellow}{self._tokens_per_request}{reset} "
            f"(measured)\n"
            f"  Vocab size:          {yellow}{c.vocab_size}{reset}\n"
            f"  Max inflight:        {yellow}{c.num_inflight_requests}{reset}\n"
            f"  Max output length:   {yellow}{c.max_output_length}{reset} tokens\n"
            f"{bold}{'═' * 50}{reset}"
        )

    def extra_metric_sections(self) -> list[MetricSection]:
        """Report the pool geometry that drove the run.

        Deliberately reports the inputs only.  A derived "predicted L2 read
        share" used to sit here, but ``1 - 1 / F`` assumes pure
        LRU over uniformly-drawn whole documents and was measured 14 points
        low on one stack and 45 points high on another -- printing it beside
        real counters invited it being quoted as a result.  Read the share
        that actually occurred from the cache's tier counters.
        """
        c = self._config
        entries: list[tuple[str, str, str | int | float]] = [
            ("pool_size", "Document pool", c.pool_size),
            ("docs_per_request", "Docs per request", c.docs_per_request),
            ("working_set_tokens", "Working set (tokens)", self.working_set_tokens),
            ("access_skew", "Access skew", round(c.access_skew, 3)),
            (
                "warmup_sweep_requests",
                "Warm-up sweep requests",
                len(self._sweep_groups),
            ),
            ("tokens_per_request", "Tokens per request", self._tokens_per_request),
        ]
        entries.extend(self._hit_token_entries())
        return [
            MetricSection(
                key="kv_tier_pressure",
                label="Document pool",
                entries=entries,
            )
        ]

    def _hit_token_entries(self) -> list[tuple[str, str, str | int | float]]:
        """Cache-hit tokens over the measured phase, as counter deltas.

        Reported alongside the pool geometry because the share of reads
        served by the storage tier cannot distinguish a cache that is
        serving most of each prompt from one serving almost none of it --
        it is a ratio between tiers, and reads much the same either way.
        Hit tokens per request is the figure that separates them.

        Returns:
            Entries to append to the metric section.  Empty when no LMCache
            URL was supplied or the counters were unavailable.
        """
        if not self._lmcache_url:
            return []
        after = scrape_lookup_tokens(self._lmcache_url)
        hit = after["hit"] - self._lookup_at_boundary["hit"]
        requested = after["requested"] - self._lookup_at_boundary["requested"]
        if requested <= 0:
            return []
        per_request = hit / max(len(self._measured_groups), 1)
        return [
            ("hit_tokens_total", "Cache-hit tokens", int(hit)),
            ("requested_tokens_total", "Tokens looked up", int(requested)),
            (
                "hit_tokens_per_request",
                "Cache-hit tokens per request",
                round(per_request, 1),
            ),
            (
                "hit_token_rate_pct",
                "Prompt served from cache (%)",
                round(100.0 * hit / requested, 2),
            ),
        ]

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def _build_messages(self, group: tuple[int, ...]) -> list[dict[str, str]]:
        """Build the chat messages for one group of document indices.

        Args:
            group: Document indices to concatenate, in order.

        Returns:
            Chat messages: the shared system prompt when non-empty, followed
            by one user message holding the concatenated documents.
        """
        concatenated = "\n\n".join(self._documents[i] for i in group)
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": concatenated})
        return messages

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------

    async def warmup(self) -> None:
        """Sweep the pool so every document is cached exactly once.

        Runs ``ceil(pool_size / docs_per_request)`` requests at the
        configured concurrency.  Without this the measured L2 read share
        would describe a cold start rather than steady state.
        """
        total = len(self._sweep_groups)
        self._progress_monitor.log_message(
            f"Warm-up sweep: {total} requests covering every document once"
        )
        semaphore = asyncio.Semaphore(self._config.num_inflight_requests)

        async def send(index: int, group: tuple[int, ...]) -> None:
            async with semaphore:
                request_id = f"warm{index}"
                self._progress_monitor.on_request_sent(request_id)
                result = await self._request_sender.send_warmup_request(
                    request_id, self._build_messages(group)
                )
                if not result.successful:
                    self._progress_monitor.log_message(
                        f"Warm-up request {index} failed: {result.error}"
                    )

        await asyncio.gather(*[send(i, g) for i, g in enumerate(self._sweep_groups)])
        self._progress_monitor.log_message("Warm-up sweep complete")
        # Snapshot here, not at construction: everything the sweep stored is
        # warm-up, and counting it would inflate the measured hit rate.
        if self._lmcache_url:
            self._lookup_at_boundary = scrape_lookup_tokens(self._lmcache_url)

    # ------------------------------------------------------------------
    # Benchmark dispatch
    # ------------------------------------------------------------------

    async def step(self, time_offset: float) -> float:
        """Dispatch the next sampled request if the semaphore allows.

        Args:
            time_offset: Seconds since benchmark start (unused).

        Returns:
            0.0 to request an immediate re-call, or -1.0 when all done.
        """
        if self._request_index < len(self._measured_groups):
            await self._semaphore.acquire()
            req_idx = self._request_index
            group = self._measured_groups[req_idx]
            self._request_index += 1

            task = asyncio.create_task(self._dispatch(group, req_idx))
            self._pending_tasks.add(task)
            task.add_done_callback(self._on_task_done)
            return 0.0

        if self._pending_tasks:
            await asyncio.wait(
                self._pending_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            return 0.0

        return -1.0

    async def _dispatch(self, group: tuple[int, ...], req_idx: int) -> None:
        """Send a single benchmark request, then release the semaphore.

        Args:
            group: Document indices carried by this request.
            req_idx: Request index captured before incrementing the counter.
        """
        request_id = f"sample{req_idx}"
        self._progress_monitor.on_request_sent(request_id)
        try:
            await self._request_sender.send_request(
                request_id,
                self._build_messages(group),
                max_tokens=self._config.max_output_length,
            )
        finally:
            self._semaphore.release()

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Clean up completed tasks and log unexpected errors.

        Args:
            task: The completed asyncio Task.
        """
        self._pending_tasks.discard(task)
        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                self._progress_monitor.log_message(f"Dispatch task failed: {exc}")

    def on_request_finished(self, request_id: str, output: str) -> None:
        """No-op -- this workload is stateless.

        Args:
            request_id: Identifier of the completed request.
            output: Text the engine generated.
        """
