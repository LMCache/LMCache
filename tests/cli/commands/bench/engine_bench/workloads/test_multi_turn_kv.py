# SPDX-License-Identifier: Apache-2.0
"""Public-interface tests for exact-prefix mixed-length conversations."""

# Standard
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import time

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.config import WarmupPolicy
from lmcache.cli.commands.bench.engine_bench.request_sender import TokenCompletion
from lmcache.cli.commands.bench.engine_bench.stats import RequestResult, StatsCollector
from lmcache.cli.commands.bench.engine_bench.workloads import multi_turn_kv as mt

# Local
from ..fake_tokenizer import make_fake_tokenizer


def result(request_id: str, successful: bool = True) -> RequestResult:
    """Return a measured response suitable for deterministic sender doubles."""
    now = time.time()
    return RequestResult(
        request_id,
        successful,
        0.01,
        0.02,
        10,
        2,
        100,
        now,
        now + 0.01,
        now + 0.02,
        "" if successful else "failed",
    )


def make_workload(
    *,
    num_sessions: int = 4,
    max_inflight: int = 2,
    turns: int = 3,
    think_seconds: float = 0.0,
    seed: int = 42,
) -> tuple[mt.MultiTurnKVWorkload, MagicMock]:
    """Build a small workload with a real tokenizer pool and mocked transport."""
    config = mt.MultiTurnKVConfig(
        num_sessions=num_sessions,
        max_inflight=max_inflight,
        turns=turns,
        short_context=8,
        long_context=20,
        turn_tokens=3,
        output_tokens=2,
        max_context_tokens=64,
        vocab_size=64,
        think_seconds=think_seconds,
        seed=seed,
    )
    sender = MagicMock()
    sender.close = AsyncMock()

    async def send(
        request_id: str, prompt: list[int], max_tokens: int, seed: int
    ) -> TokenCompletion:
        await asyncio.sleep(0)
        return TokenCompletion(result(request_id), [1001, 1002])

    sender.send_token_request = AsyncMock(side_effect=send)
    with patch.object(mt, "try_load_tokenizer", return_value=make_fake_tokenizer()):
        workload = mt.MultiTurnKVWorkload(
            config,
            sender,
            StatsCollector(),
            MagicMock(),
            "fake",
        )
    return workload, sender


class TestMultiTurnKV:
    @pytest.mark.parametrize(
        "failure", [RuntimeError("callback failed"), asyncio.CancelledError()]
    )
    def test_unexpected_failure_drains_peers_before_client_close(
        self, failure: BaseException
    ) -> None:
        workload, sender = make_workload(num_sessions=2, turns=1)
        peer_finished = False

        async def send(
            request_id: str, prompt: list[int], max_tokens: int, seed: int
        ) -> TokenCompletion:
            nonlocal peer_finished
            await asyncio.sleep(0)
            if request_id.startswith("short"):
                raise failure
            await asyncio.sleep(0.005)
            peer_finished = True
            return TokenCompletion(result(request_id), [1001, 1002])

        async def close() -> None:
            assert peer_finished

        sender.send_token_request.side_effect = send
        sender.close.side_effect = close
        with pytest.raises(type(failure)):
            workload.run()
        assert peer_finished
        sender.close.assert_awaited_once()

    def test_exact_prefix_includes_generated_ids_without_reencoding(self) -> None:
        workload, sender = make_workload()
        workload.run()
        calls = sender.send_token_request.await_args_list
        assert len(calls) == 12
        for session in range(4):
            ordered = [call for call in calls if f"_s{session}_" in call.args[0]]
            assert [call.args[0].split("_t")[1] for call in ordered] == ["0", "1", "2"]
            assert len(ordered[0].args[1]) == (8 if session % 2 == 0 else 20) + 3
            for previous, following in zip(ordered, ordered[1:], strict=False):
                prefix = previous.args[1] + [1001, 1002]
                assert following.args[1][: len(prefix)] == prefix
                assert len(following.args[1]) == len(prefix) + 3
        assert workload.initial_working_set_tokens == 56
        records = workload.get_turns()
        assert len(records) == 12
        assert sum(record.phase == "initial" for record in records) == 4
        assert all(
            record.prefix_overlap_tokens > 0 for record in records if record.turn
        )
        sender.close.assert_awaited_once()

    def test_seeded_inputs_and_generation_seeds_are_reproducible(self) -> None:
        first, a = make_workload(seed=7)
        second, b = make_workload(seed=7)
        third, c = make_workload(seed=8)
        first.run()
        second.run()
        third.run()

        def normalize(sender: MagicMock) -> list[tuple[str, list[int], int]]:
            return sorted(
                (call.args[0], call.args[1], call.kwargs["seed"])
                for call in sender.send_token_request.await_args_list
            )

        assert normalize(a) == normalize(b)
        assert normalize(a) != normalize(c)

    def test_slow_session_does_not_block_other_ready_sessions(self) -> None:
        workload, sender = make_workload(num_sessions=2)
        active = 0
        peak = 0
        long_finished = 0

        async def send(
            request_id: str, prompt: list[int], max_tokens: int, seed: int
        ) -> TokenCompletion:
            nonlocal active, peak, long_finished
            active += 1
            peak = max(peak, active)
            if request_id == "short_s0_t0":
                while long_finished < 3:
                    await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(0)
            if request_id.startswith("long"):
                long_finished += 1
            active -= 1
            return TokenCompletion(result(request_id), [1001, 1002])

        sender.send_token_request.side_effect = send

        async def drive() -> None:
            start = time.monotonic()
            while await workload.step(time.monotonic() - start) >= 0:
                await asyncio.sleep(0)

        asyncio.run(asyncio.wait_for(drive(), timeout=2))
        assert peak == 2 and active == 0
        assert long_finished == 3

    def test_failure_stops_only_its_session_and_does_not_retry(self) -> None:
        workload, sender = make_workload(num_sessions=2)

        async def send(
            request_id: str, prompt: list[int], max_tokens: int, seed: int
        ) -> TokenCompletion:
            successful = not request_id.startswith("short")
            return TokenCompletion(
                result(request_id, successful), [1001, 1002] if successful else []
            )

        sender.send_token_request.side_effect = send
        workload.run(WarmupPolicy.SKIP)
        assert sender.send_token_request.await_count == 4
        assert sum(not record.result.successful for record in workload.get_turns()) == 1
        sender.close.assert_awaited_once()

    def test_think_time_begins_at_completion_and_global_limit_holds(self) -> None:
        workload, sender = make_workload(max_inflight=1, turns=2, think_seconds=0.01)
        completions: dict[str, float] = {}

        async def send(
            request_id: str, prompt: list[int], max_tokens: int, seed: int
        ) -> TokenCompletion:
            session = request_id.split("_t")[0]
            if session in completions:
                assert time.monotonic() - completions[session] >= 0.01
            await asyncio.sleep(0.001)
            completions[session] = time.monotonic()
            return TokenCompletion(result(request_id), [1001, 1002])

        sender.send_token_request.side_effect = send
        workload.run()
        assert len(workload.get_turns()) == 8
        assert all(turn.queue_seconds >= 0 for turn in workload.get_turns())

    @pytest.mark.parametrize(
        "construct",
        [
            lambda: mt.MultiTurnKVConfig(num_sessions=1),
            lambda: mt.MultiTurnKVConfig(max_inflight=0),
            lambda: mt.MultiTurnKVConfig(turns=0),
            lambda: mt.MultiTurnKVConfig(short_context=20, long_context=10),
            lambda: mt.MultiTurnKVConfig(think_seconds=-1),
            lambda: mt.MultiTurnKVConfig(think_seconds=float("nan")),
            lambda: mt.MultiTurnKVConfig(max_context_tokens=1),
        ],
    )
    def test_invalid_budgets_rejected(
        self, construct: Callable[[], mt.MultiTurnKVConfig]
    ) -> None:
        with pytest.raises(ValueError):
            construct()
