# SPDX-License-Identifier: Apache-2.0
"""Test doubles for ``bench quality``: a scripted sender and counter source."""

# Standard
from collections.abc import Callable

# First Party
from lmcache.cli.commands.bench.engine_bench.quality.dataset import Sample
from lmcache.cli.commands.bench.engine_bench.stats import RequestResult
from lmcache.cli.commands.bench.quality_bench.core.blend_counters import (
    BlendCounters,
    BlendCounterSource,
)
from lmcache.cli.commands.bench.quality_bench.core.judge import JudgeClient

# (request_id, messages) -> response text
Responder = Callable[[str, list[dict[str, str]]], str]


def result(request_id: str, successful: bool = True) -> RequestResult:
    """Build a completed (or failed) request result."""
    return RequestResult(
        request_id=request_id,
        successful=successful,
        ttft=0.1 if successful else -1.0,
        request_latency=0.5,
        num_input_tokens=100,
        num_output_tokens=8,
        decode_speed=16.0,
        submit_time=0.0,
        first_token_time=0.1,
        finish_time=0.5,
        error="" if successful else "boom",
    )


def answering(answers: dict[str, str]) -> Responder:
    """Respond to each measured question with a fixed answer.

    Args:
        answers: Question text -> answer. Questions not listed get no
            answer region (an unparsed response).
    """

    def respond(request_id: str, messages: list[dict[str, str]]) -> str:
        content = messages[0]["content"]
        for question, answer in answers.items():
            if f"Question: {question}" in content:
                return f"thinking... <final_answer>{answer}</final_answer>"
        return "thinking..."

    return respond


class FakeSender:
    """Stands in for ``RequestSender``: records requests + replies by script."""

    def __init__(self, respond: Responder, failing: frozenset[str] = frozenset()):
        self._respond = respond
        self._failing = failing
        self._callbacks: list[Callable[[RequestResult, str], None]] = []
        self.sent: list[tuple[str, list[dict[str, str]], int]] = []
        self.closed = False

    def add_on_finished_callback(
        self, callback: Callable[[RequestResult, str], None]
    ) -> None:
        self._callbacks.append(callback)

    async def send_request(
        self, request_id: str, messages: list[dict[str, str]], max_tokens: int = 128
    ) -> RequestResult:
        self.sent.append((request_id, messages, max_tokens))
        ok = request_id not in self._failing
        text = self._respond(request_id, messages) if ok else ""
        outcome = result(request_id, ok)
        for callback in self._callbacks:
            callback(outcome, text)
        return outcome

    async def send_warmup_request(
        self, request_id: str, messages: list[dict[str, str]], max_tokens: int = 1
    ) -> RequestResult:
        return await self.send_request(request_id, messages, max_tokens)

    async def close(self) -> None:
        self.closed = True


class SteppingCounters(BlendCounterSource):
    """Counters that advance by a fixed delta on every read."""

    def __init__(self, step: BlendCounters) -> None:
        self._step = step
        self._current = BlendCounters()

    @property
    def measures_engagement(self) -> bool:
        return True

    def read(self) -> BlendCounters:
        self._current = self._current + self._step
        return self._current


def samples() -> list[Sample]:
    """Three questions; two share a passage."""
    return [
        Sample(
            "s0", ["Paris is in France.", "Rome is in Italy."], "Capital?", ["Paris"]
        ),
        Sample("s1", ["Rome is in Italy.", "Oslo is in Norway."], "Nordic?", ["Oslo"]),
        Sample(
            "s2",
            ["Bern is Swiss.", "Vienna is Austrian.", "Prague is Czech."],
            "Swiss?",
            ["Bern"],
            evidence_indices=[0],
        ),
    ]


class FakeJudgeClient(JudgeClient):
    """Stands in for the judge endpoint: replies by script + records prompts.

    The default script rates 4 when the user answer equals the reference
    answer and 0 otherwise, reading both from the answer-first prompt's
    "### User Answer:" and "### Reference Answer:" lines, and from the
    reference-first prompt's "User Answer:" / "Reference Answer:" lines.
    """

    def __init__(
        self,
        reply: Callable[[str], str] | None = None,
        failing: int = 0,
    ) -> None:
        self._reply = reply or rate_exact
        self._failing = failing
        self.prompts: list[str] = []
        self.closed = False

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._failing > 0:
            self._failing -= 1
            raise RuntimeError("judge down")
        return self._reply(prompt)

    async def close(self) -> None:
        self.closed = True


def _field(prompt: str, name: str) -> str:
    for line in prompt.splitlines():
        for prefix in (f"### {name}: ", f"{name}: "):
            if line.startswith(prefix):
                return line[len(prefix) :]
    return ""


def rate_exact(prompt: str) -> str:
    """Rate 4 when the prompt's user answer equals its reference, else 0."""
    same = _field(prompt, "User Answer") == _field(prompt, "Reference Answer")
    return "4" if same else "0"
