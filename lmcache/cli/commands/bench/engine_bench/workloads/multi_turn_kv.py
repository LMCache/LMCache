# SPDX-License-Identifier: Apache-2.0
"""Finite mixed-length conversations for MP KV offload qualification.

The example runner in ``examples/benchmarks/multi_turn_kv`` supplies the
serving configuration and collects MP counters. This workload only generates
traffic; it does not change cache admission, eviction or transfer lifetimes.
"""

# Standard
from dataclasses import dataclass
import asyncio
import math
import random
import time

# First Party
from lmcache.cli.commands.bench.engine_bench.progress import ProgressMonitor
from lmcache.cli.commands.bench.engine_bench.request_sender import RequestSender
from lmcache.cli.commands.bench.engine_bench.stats import RequestResult, StatsCollector
from lmcache.cli.commands.bench.engine_bench.tokenizers import (
    build_single_token_pool,
    try_load_tokenizer,
)
from lmcache.cli.commands.bench.engine_bench.workloads.base import BaseWorkload


@dataclass
class MultiTurnKVConfig:
    """Explicit token and concurrency budgets for a finite experiment.

    Sessions alternate short/long initial prefixes. ``max_context_tokens``
    includes the final turn's output reservation and must match the server's
    context window. These are traffic limits, not GPU or host memory limits.
    """

    num_sessions: int = 8
    max_inflight: int = 4
    turns: int = 4
    short_context: int = 1024
    long_context: int = 8192
    turn_tokens: int = 128
    output_tokens: int = 128
    think_seconds: float = 0.0
    max_context_tokens: int = 16384
    vocab_size: int = 256
    seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "num_sessions",
            "max_inflight",
            "turns",
            "short_context",
            "long_context",
            "turn_tokens",
            "output_tokens",
            "max_context_tokens",
            "vocab_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.num_sessions < 2 or self.short_context >= self.long_context:
            raise ValueError(
                "Use at least two sessions and short_context < long_context"
            )
        if not math.isfinite(self.think_seconds) or self.think_seconds < 0:
            raise ValueError("think_seconds must be finite and nonnegative")
        required = self.long_context + self.turns * (
            self.turn_tokens + self.output_tokens
        )
        if required > self.max_context_tokens:
            raise ValueError(
                f"Worst-case conversation needs {required} tokens, "
                f"exceeding max_context_tokens={self.max_context_tokens}"
            )


@dataclass
class ConversationTurn:
    """Per-turn measurements; queueing starts when the session becomes ready."""

    session_id: int
    cohort: str
    turn: int
    phase: str
    queue_seconds: float
    prefix_overlap_tokens: int
    result: RequestResult


@dataclass
class _Session:
    session_id: int
    cohort: str
    tokens: list[int]
    additions: list[list[int]]
    completed: int = 0
    ready_at: float = 0.0
    in_flight: bool = False
    failed: bool = False


class MultiTurnKVWorkload(BaseWorkload):
    """Append exact generated IDs while interleaving ready conversations.

    Each session has one request in flight at most. Round-robin selection
    skips busy/thinking sessions, avoiding head-of-line blocking. A failed
    turn terminates its session instead of inventing a response or retrying.
    """

    def __init__(
        self,
        config: MultiTurnKVConfig,
        request_sender: RequestSender,
        stats_collector: StatsCollector,
        progress_monitor: ProgressMonitor,
        model_name: str,
    ) -> None:
        """Build reproducible token prefixes and per-turn additions.

        Args:
            config: Validated traffic limits and seed.
            request_sender: vLLM token-capable benchmark sender.
            stats_collector: Shared request collector.
            progress_monitor: Shared progress display.
            model_name: Server tokenizer's HuggingFace ID or local path.

        Raises:
            ValueError: If a tokenizer or single-token word pool is unavailable.
        """
        super().__init__(request_sender, stats_collector, progress_monitor)
        tokenizer = try_load_tokenizer(model_name)
        if tokenizer is None:
            raise ValueError("multi-turn-kv requires the serving model's tokenizer")
        pool = build_single_token_pool(tokenizer, config.vocab_size, config.seed)
        self._config = config
        self._sessions: list[_Session] = []
        for index in range(config.num_sessions):
            rng = random.Random(config.seed + index)
            length = config.short_context if index % 2 == 0 else config.long_context
            prefix = tokenizer.encode(
                pool.join(rng.choices(pool.words, k=length)), add_special_tokens=False
            )
            additions = [
                tokenizer.encode(
                    pool.join(rng.choices(pool.words, k=config.turn_tokens)),
                    add_special_tokens=False,
                )
                for _ in range(config.turns)
            ]
            self._sessions.append(
                _Session(
                    index, "short" if index % 2 == 0 else "long", prefix, additions
                )
            )
        self._pending: set[asyncio.Task[None]] = set()
        self._records: list[ConversationTurn] = []
        self._cursor = 0
        self._origin: float | None = None

    @property
    def initial_working_set_tokens(self) -> int:
        """Sum of initial session-prefix lengths, excluding per-turn growth."""
        return sum(
            self._config.short_context if i % 2 == 0 else self._config.long_context
            for i in range(self._config.num_sessions)
        )

    def get_turns(self) -> list[ConversationTurn]:
        """Return completed turns in submission order, including failures.

        Call after ``run`` or from the workload event loop; this snapshot is
        not intended for concurrent access from the progress thread.
        """
        return sorted(self._records, key=lambda record: record.result.submit_time)

    def log_config(self) -> None:
        """Describe the traffic limits without implying a memory allocation."""
        print(
            f"Multi-turn KV: {self._config.num_sessions} sessions, "
            f"{self._config.turns} turns, max inflight={self._config.max_inflight}; "
            f"initial working set={self.initial_working_set_tokens} tokens"
        )

    async def warmup(self) -> None:
        """Leave session prefixes untouched; initial turns are measured."""

    async def step(self, time_offset: float) -> float:
        """Dispatch one ready session or wait for completion/think time.

        Args:
            time_offset: Seconds since the benchmark's measured phase began.

        Returns:
            Absolute next wakeup offset, or -1 when every session has finished.

        Raises:
            Exception: Unexpected sender/callback errors, after draining peers.
        """
        if self._origin is None:
            self._origin = time.monotonic() - time_offset
        done = {task for task in self._pending if task.done()}
        self._pending.difference_update(done)
        for task in done:
            if task.cancelled() or task.exception() is not None:
                # A local error must not close the client under peer requests.
                await asyncio.gather(*done, *self._pending, return_exceptions=True)
                self._pending.clear()
                task.result()
        available = [
            s
            for s in self._sessions
            if not s.failed and s.completed < self._config.turns and not s.in_flight
        ]
        if len(self._pending) < self._config.max_inflight:
            for offset in range(len(self._sessions)):
                index = (self._cursor + offset) % len(self._sessions)
                session = self._sessions[index]
                if session in available and session.ready_at <= time_offset:
                    session.in_flight = True
                    self._cursor = (index + 1) % len(self._sessions)
                    self._pending.add(
                        asyncio.create_task(
                            self._dispatch(
                                session, time_offset - session.ready_at, self._origin
                            )
                        )
                    )
                    return 0.0
        if self._pending:
            timeout = None
            if available and len(self._pending) < self._config.max_inflight:
                timeout = max(0.0, min(s.ready_at for s in available) - time_offset)
            await asyncio.wait(
                self._pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            return 0.0
        return min(s.ready_at for s in available) if available else -1.0

    def on_request_finished(self, request_id: str, output: str) -> None:
        """Ignore text callbacks; only validated server token IDs extend sessions."""

    async def _dispatch(self, session: _Session, queued: float, origin: float) -> None:
        """Keep session state until the request returns; never retry a failed turn."""
        turn = session.completed
        prompt = session.tokens + session.additions[turn]
        request_id = f"{session.cohort}_s{session.session_id}_t{turn}"
        self._progress_monitor.on_request_sent(request_id)
        completion = await self._request_sender.send_token_request(
            request_id,
            prompt,
            self._config.output_tokens,
            seed=self._config.seed + session.session_id * self._config.turns + turn,
        )
        self._records.append(
            ConversationTurn(
                session.session_id,
                session.cohort,
                turn,
                "initial" if turn == 0 else "continuation",
                queued,
                len(session.tokens) if turn else 0,
                completion.result,
            )
        )
        if completion.result.successful:
            session.tokens = prompt + completion.token_ids
            session.completed += 1
        else:
            session.failed = True
        session.in_flight = False
        session.ready_at = time.monotonic() - origin + self._config.think_seconds
