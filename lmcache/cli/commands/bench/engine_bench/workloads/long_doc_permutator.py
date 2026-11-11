# SPDX-License-Identifier: Apache-2.0
"""Long-document permutator workload for ``lmcache bench engine``.

Stress-tests blended KV cache reuse by sending permutations of a set of
context documents.  Each request is:

    [System Prompt] + [Doc_i1] + [Doc_i2] + ... + [Doc_iN]

where (i1, ..., iN) is one permutation of the N contexts.

Stress axes (controlled by config):
    1. Blended Context Boundaries  -> num_contexts
    2. Eviction                    -> num_permutations
    3. Chunk Homogeneity           -> vocab_size
    4. Prefix Domination           -> system_prompt_length
    5. Concurrency                 -> num_inflight_requests

``max_output_length`` sets how much decode follows each prefill: 1 makes the
run a pure prefill/TTFT measurement, while larger values also exercise the
decode phase (combine with ``--ignore-eos`` to make its length reproducible).

Context and system-prompt lengths are exact token counts, not estimates:
all synthetic text is built from words that encode to exactly one token
under the model's own tokenizer (see
:mod:`lmcache.cli.commands.bench.engine_bench.tokenizers`).  This workload
therefore requires a loadable tokenizer.
"""

# Standard
from dataclasses import dataclass
import math

# First Party
from lmcache.cli.commands.bench.engine_bench.progress import ProgressMonitor
from lmcache.cli.commands.bench.engine_bench.request_sender import RequestSender
from lmcache.cli.commands.bench.engine_bench.stats import StatsCollector
from lmcache.cli.commands.bench.engine_bench.workloads.base import MetricSection
from lmcache.cli.commands.bench.engine_bench.workloads.kv_tier_pressure import (
    DocumentWorkloadMode,
    KVTierPressureConfig,
    KVTierPressureWorkload,
)


@dataclass
class LongDocPermutatorConfig:
    """Workload-specific config for the long-doc-permutator workload."""

    num_contexts: int = 5
    context_length: int = 5000
    system_prompt_length: int = 1000
    num_permutations: int = 10
    vocab_size: int = 8000
    num_inflight_requests: int = 1
    max_output_length: int = 128

    def __post_init__(self) -> None:
        if self.num_contexts < 1:
            raise ValueError(f"num_contexts must be >= 1, got {self.num_contexts}")
        if self.context_length <= 0:
            raise ValueError(
                f"context_length must be positive, got {self.context_length}"
            )
        if self.num_permutations < 1:
            raise ValueError(
                f"num_permutations must be >= 1, got {self.num_permutations}"
            )
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
        num_contexts: int = 5,
        context_length: int = 5000,
        system_prompt_length: int = 1000,
        num_permutations: int = 10,
        vocab_size: int = 8000,
        num_inflight_requests: int = 1,
        max_output_length: int = 128,
    ) -> "LongDocPermutatorConfig":
        """Create a config directly from the provided parameters.

        Args:
            num_contexts: Number of unique context documents.
            context_length: Exact token length of each context.
            system_prompt_length: Exact token length of the shared system
                prompt.  Use 0 for no system prompt.
            num_permutations: Number of distinct permutations to send.
                Capped at N! where N = num_contexts.
            vocab_size: Number of distinct single-token words to sample
                context content from.  Smaller values increase chunk hash
                collision risk.
            num_inflight_requests: Max concurrent in-flight requests.
            max_output_length: Max tokens to generate per benchmark request.
                Use 1 to measure prefill alone; combine larger values with
                ``--ignore-eos`` for a reproducible decode phase.

        Returns:
            A fully-resolved LongDocPermutatorConfig.
        """
        return cls(
            num_contexts=num_contexts,
            context_length=context_length,
            system_prompt_length=system_prompt_length,
            num_permutations=num_permutations,
            vocab_size=vocab_size,
            num_inflight_requests=num_inflight_requests,
            max_output_length=max_output_length,
        )


class LongDocPermutatorWorkload(KVTierPressureWorkload):
    """Compatibility wrapper for distinct, full-pool document permutations.

    Uses the tier-pressure implementation for text generation, message
    construction and dispatch, preserving the historical RNG sequence,
    permutation cap, request IDs, and single dummy warm-up.
    """

    def __init__(
        self,
        config: LongDocPermutatorConfig,
        request_sender: RequestSender,
        stats_collector: StatsCollector,
        progress_monitor: ProgressMonitor,
        seed: int = 42,
        model_name: str | None = None,
    ) -> None:
        """Adapt legacy flags to a full-pool permutation workload.

        Args:
            config: Legacy workload configuration, including uncapped count.
            request_sender: Sender shared with the benchmark orchestrator.
            stats_collector: Collector for measured request results.
            progress_monitor: Monitor for dispatch and warm-up progress.
            seed: Seed for the historical text and permutation stream.
            model_name: Model whose tokenizer sizes the prompts.

        Raises:
            ValueError: If no tokenizer can be loaded or the config is invalid.
        """
        self._permutator_config = config
        super().__init__(
            KVTierPressureConfig(
                pool_size=config.num_contexts,
                docs_per_request=config.num_contexts,
                context_length=config.context_length,
                system_prompt_length=config.system_prompt_length,
                num_requests=config.num_permutations,
                vocab_size=config.vocab_size,
                num_inflight_requests=config.num_inflight_requests,
                max_output_length=config.max_output_length,
            ),
            request_sender,
            stats_collector,
            progress_monitor,
            seed=seed,
            model_name=model_name,
            mode=DocumentWorkloadMode.PERMUTATIONS,
        )

    def log_config(self) -> None:
        """Log key workload config before the benchmark starts."""
        c = self._permutator_config
        B = "\033[1m"
        C = "\033[96m"
        Y = "\033[93m"
        R = "\033[0m"
        total = self.num_requests
        actual_perms = self.num_requests
        print(
            f"{B}{'═' * 50}{R}\n"
            f"{B} Workload: {C}long-doc-permutator{R}\n"
            f"{B}{'─' * 50}{R}\n"
            f"  Contexts:            {Y}{c.num_contexts}{R}\n"
            f"  Context length:      {Y}{c.context_length}{R} tokens\n"
            f"  System prompt:       {Y}{c.system_prompt_length}{R} tokens\n"
            f"  Permutations:        {Y}{actual_perms}{R} "
            f"(of {math.factorial(c.num_contexts)} possible)\n"
            f"  Total requests:      {Y}{total}{R}\n"
            f"  Tokens per request:  {Y}{self.tokens_per_request}{R} (measured)\n"
            f"  Vocab size:          {Y}{c.vocab_size}{R}\n"
            f"  Max inflight:        {Y}{c.num_inflight_requests}{R}\n"
            f"  Max output length:   {Y}{c.max_output_length}{R} tokens\n"
            f"{B}{'═' * 50}{R}"
        )

    def extra_metric_sections(self) -> list[MetricSection]:
        """Keep the legacy report free of tier-pressure-specific metrics."""
        return []
