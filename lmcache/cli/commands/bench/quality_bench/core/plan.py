# SPDX-License-Identifier: Apache-2.0
"""Prompt plans and reuse manifests for ``lmcache bench quality``.

A QA dataset gives a question and its passages, but has no information about how
those passages' KV data got into the cache. A :class:`QualityPlan` makes that
explicit. That is, it specifies which store requests run before measuring,
the byte-identical measured prompt every arm receives, and, per passage, where
its cached KV came from (the reuse manifest).
"""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any
import enum
import hashlib
import json
import random

# First Party
from lmcache.cli.commands.bench.engine_bench.quality.alignment import (
    ChunkAligner,
    Filler,
)
from lmcache.cli.commands.bench.engine_bench.quality.dataset import Sample
from lmcache.cli.commands.bench.engine_bench.quality.prompts import (
    ANSWER_TAG_PROMPT,
    compose_qa_messages,
    compose_store_messages,
)
from lmcache.cli.commands.bench.engine_bench.quality.scoring import normalize_answer
from lmcache.cli.commands.bench.quality_bench.core.config import ReuseScenario

_STORE_REQUEST_PREFIX = "store"
_MEASURED_REQUEST_PREFIX = "q"

# Misplaced blocks listed in a PlanAlignmentError; the rest are counted.
_MAX_LISTED_PROBLEMS = 5


class PlanAlignmentError(ValueError):
    """Some passage would not be reused: it is off a chunk boundary, or
    tokenizes differently in two prompts, once the chat template is applied.

    Attributes:
        problems: One line per misplaced block.
    """

    def __init__(self, problems: list[str], filler: Filler) -> None:
        self.problems = problems
        listed = "\n  ".join(problems[:_MAX_LISTED_PROBLEMS])
        more = len(problems) - _MAX_LISTED_PROBLEMS
        super().__init__(
            f"{len(problems)} passage placement(s) would lose cache reuse "
            f"with --filler {filler.value}; the run would measure less reuse "
            f"than its plan claims:\n  {listed}"
            + (f"\n  … and {more} more" if more > 0 else "")
            + "\nTry another --filler, or check --chunk-size and "
            "--template-kwargs against the engine."
        )


class StoreOrigin(enum.Enum):
    """Where a passage's cached KV was computed."""

    ALONE = "alone"
    """Stored by a request holding only this passage (after the system
    block)."""

    WITH_SAMPLE_PASSAGES = "with_sample_passages"
    """Stored together with the rest of this question's passages, in dataset
    order, so its cached KV saw different neighbors than it has when
    measured."""

    IN_UNIT = "in_unit"
    """Stored inside a ``concatenated`` unit: its question's context, in
    request order, after other questions' contexts and without the system
    prompt."""


class EvidenceSource(enum.Enum):
    """How a sample's answer-bearing passages were identified."""

    DATASET = "dataset"
    """The dataset marks them (``is_supporting``, ``supporting_facts``,
    ``gold``)."""

    ANSWER_STRING = "answer_string"
    """No marking -- passages containing a gold answer string."""

    NONE = "none"
    """No marking, and no passage contains a gold answer string."""


@dataclass(frozen=True)
class StoreRequest:
    """A request sent before measuring, to put passages in the cache.

    Attributes:
        request_id: Identifier, referenced by the manifest.
        messages: The chat messages to send (``max_tokens=1``).
    """

    request_id: str
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class PassageProvenance:
    """One passage of a measured prompt and where its cached KV came from.

    Attributes:
        passage_id: Stable id: a digest of the raw passage text.
        origin: How the passage was stored.
        store_request_id: The store request that cached it.
        stored_position: Its index among the passages of that store request.
        measured_position: Its index among the passages of the measured
            prompt.
        is_evidence: Whether it holds the answer (see
            :class:`EvidenceSource`).
        measured_earlier_in: Request ids of earlier measured questions whose
            prompts also hold this passage, in measured order. A caching
            candidate caches measured prompts too; the LMCache blend
            server's fingerprint index by default keeps the newest copy of
            identical content, so the passage's KV may come from the last of
            these (itself blended) rather than from ``store_request_id``. Empty
            means ``store_request_id`` is the only cached copy. See the
            design doc, "Reuse from earlier measured questions".
        pad_tokens: Filler tokens added to align it to whole chunks (``0``
            in plans made before this was recorded).
    """

    passage_id: str
    origin: StoreOrigin
    store_request_id: str
    stored_position: int
    measured_position: int
    is_evidence: bool
    measured_earlier_in: tuple[str, ...] = ()
    pad_tokens: int = 0


@dataclass(frozen=True)
class SamplePlan:
    """The measured request for one question, with its reuse manifest.

    Attributes:
        sample: The dataset sample.
        request_id: Identifier of the measured request (same on every arm).
        messages: The measured chat messages. Byte-identical on every arm.
        passages: Provenance of each passage, in measured order.
        evidence_source: How ``is_evidence`` was decided.
        expected_reused_tokens: Tokens of the measured prompt covered by a
            whole stored chunk (see :func:`build_plan`), i.e. what a cache
            matching chunks anywhere in the prompt could reuse. ``None`` in
            plans made before this was recorded.
        unit_position: ``concatenated`` only: the index of this question's
            context within its unit (``0`` = first).
        duplicate_of: ``concatenated`` only: the question whose identical
            context was stored in its place, or ``None``.
    """

    sample: Sample
    request_id: str
    messages: list[dict[str, str]]
    passages: list[PassageProvenance]
    evidence_source: EvidenceSource
    expected_reused_tokens: int | None = None
    unit_position: int | None = None
    duplicate_of: str | None = None

    @property
    def prompt_hash(self) -> str:
        """Digest of the measured messages. Equal prompts have the same hash."""
        return _digest(json.dumps(self.messages, sort_keys=True))

    @property
    def prompt_text(self) -> str:
        """The measured prompt's full text (for locating answers in it)."""
        return self.messages[0]["content"]

    def manifest_entry(
        self, scenario: ReuseScenario, chunk_size: int
    ) -> dict[str, object]:
        """Return this question's reuse-manifest record.

        Args:
            scenario: The plan's reuse scenario.
            chunk_size: The chunk size passages were aligned to.

        Returns:
            A JSON-serializable record with keys ``qid``, ``scenario``,
            ``chunk_size``, ``request_id``, ``prompt_hash``,
            ``evidence_source``, ``evidence_passage_ids``,
            ``expected_reused_tokens``, ``unit_position``, ``duplicate_of``
            and ``passages``
            (one dict per passage: ``passage_id``, ``origin``,
            ``store_request_id``, ``stored_position``, ``measured_position``,
            ``is_evidence``, ``measured_earlier_in``, ``pad_tokens``).
        """
        return {
            "qid": self.sample.sample_id,
            "scenario": scenario.value,
            "chunk_size": chunk_size,
            "request_id": self.request_id,
            "prompt_hash": self.prompt_hash,
            "evidence_source": self.evidence_source.value,
            "evidence_passage_ids": [
                p.passage_id for p in self.passages if p.is_evidence
            ],
            "expected_reused_tokens": self.expected_reused_tokens,
            "unit_position": self.unit_position,
            "duplicate_of": self.duplicate_of,
            "passages": [
                {
                    "passage_id": p.passage_id,
                    "origin": p.origin.value,
                    "store_request_id": p.store_request_id,
                    "stored_position": p.stored_position,
                    "measured_position": p.measured_position,
                    "is_evidence": p.is_evidence,
                    "measured_earlier_in": list(p.measured_earlier_in),
                    "pad_tokens": p.pad_tokens,
                }
                for p in self.passages
            ],
        }


@dataclass(frozen=True)
class PaddingStats:
    """How much of the measured prompts is chunk-alignment filler.

    Attributes:
        filler: The filler used.
        pad_tokens: Filler tokens over every measured prompt.
        prompt_tokens: Tokens of every measured prompt, chat template
            included.
    """

    filler: str
    pad_tokens: int
    prompt_tokens: int

    @property
    def share(self) -> float:
        """Filler tokens as a share of measured prompt tokens."""
        return self.pad_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


@dataclass(frozen=True)
class QualityPlan:
    """Everything every arm sends, in order.

    Attributes:
        scenario: How passages are cached.
        chunk_size: The chunk size passages are aligned to.
        store_requests: Requests to send before measuring, in order.
        samples: One measured request per question, in dataset order.
        padding: Filler statistics, or ``None`` for plans made before they
            were recorded.
    """

    scenario: ReuseScenario
    chunk_size: int
    store_requests: list[StoreRequest]
    samples: list[SamplePlan]
    padding: PaddingStats | None = None

    @property
    def prompts_digest(self) -> str:
        """Digest over every store and measured prompt, in order."""
        material = json.dumps(
            {
                "store": [r.messages for r in self.store_requests],
                "measured": [s.messages for s in self.samples],
            },
            sort_keys=True,
        )
        return _digest(material)

    def expected_reused_tokens(self) -> int | None:
        """Sum of every question's ``expected_reused_tokens``.

        Returns:
            The total, or ``None`` when the plan predates the field.
        """
        values = [s.expected_reused_tokens for s in self.samples]
        if any(v is None for v in values):
            return None
        return sum(v for v in values if v is not None)

    def evidence_coverage(self) -> dict[str, int]:
        """Count questions by how their evidence passages were identified.

        Returns:
            Every :class:`EvidenceSource` value, with its question count.
        """
        counts = {source.value: 0 for source in EvidenceSource}
        for sample in self.samples:
            counts[sample.evidence_source.value] += 1
        return counts

    def measured_earlier_share(self) -> float:
        """Share of measured passages that an earlier question also held.

        A dataset property, known before any request: it bounds how much of
        a caching candidate's reuse may come from earlier measured prompts instead
        of the store requests (see ``PassageProvenance.measured_earlier_in``).

        Returns:
            The share over every passage of every measured prompt. ``0.0``
            when no passage recurs.
        """
        passages = [p for sample in self.samples for p in sample.passages]
        return sum(1 for p in passages if p.measured_earlier_in) / len(passages)

    def to_record(self) -> dict[str, object]:
        """Return the whole plan as a JSON-serializable record.

        ``bench quality run`` saves it so that ``bench quality analyze`` can
        rebuild the plan without the tokenizer or the dataset. Each sample's
        record is its :meth:`SamplePlan.manifest_entry` plus the question,
        the gold answers, the passages and the measured messages.

        Returns:
            A record that :meth:`from_record` turns back into an equal plan.
        """
        record: dict[str, object] = {
            "scenario": self.scenario.value,
            "chunk_size": self.chunk_size,
            "store_requests": [
                {"request_id": r.request_id, "messages": r.messages}
                for r in self.store_requests
            ],
            "samples": [
                {
                    **s.manifest_entry(self.scenario, self.chunk_size),
                    "question": s.sample.question,
                    "answers": s.sample.answers,
                    "documents": s.sample.documents,
                    "evidence_indices": s.sample.evidence_indices,
                    "messages": s.messages,
                }
                for s in self.samples
            ],
        }
        if self.padding is not None:
            record["padding"] = {
                "filler": self.padding.filler,
                "pad_tokens": self.padding.pad_tokens,
                "prompt_tokens": self.padding.prompt_tokens,
            }
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "QualityPlan":
        """Rebuild a plan from a :meth:`to_record` record.

        Args:
            record: A record written by :meth:`to_record`.

        Returns:
            The plan.

        Raises:
            ValueError: If the record is malformed, or a question's messages
                no longer match its recorded ``prompt_hash`` (the file was
                edited or corrupted).
        """
        try:
            scenario = ReuseScenario(record["scenario"])
            padding = record.get("padding")
            plan = cls(
                scenario=scenario,
                chunk_size=int(record["chunk_size"]),
                store_requests=[
                    StoreRequest(r["request_id"], r["messages"])
                    for r in record["store_requests"]
                ],
                samples=[_sample_plan_from_record(s) for s in record["samples"]],
                padding=(
                    None
                    if padding is None
                    else PaddingStats(
                        str(padding["filler"]),
                        int(padding["pad_tokens"]),
                        int(padding["prompt_tokens"]),
                    )
                ),
            )
            recorded_hashes = [s["prompt_hash"] for s in record["samples"]]
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"malformed plan record: {e!r}") from e
        for sample_plan, recorded in zip(plan.samples, recorded_hashes, strict=True):
            if sample_plan.prompt_hash != recorded:
                raise ValueError(
                    f"question {sample_plan.sample.sample_id}: its messages do "
                    f"not match the recorded prompt_hash (edited or corrupted)"
                )
        return plan


def _digest(text: str) -> str:
    """Return a short, stable hex digest of *text*."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _passage_id(passage: str) -> str:
    """Return the stable id of a raw passage."""
    return _digest(passage)[:12]


def _sample_plan_from_record(record: Mapping[str, Any]) -> SamplePlan:
    """Rebuild one question's plan from its ``QualityPlan.to_record`` entry."""
    sample = Sample(
        sample_id=record["qid"],
        documents=list(record["documents"]),
        question=record["question"],
        answers=list(record["answers"]),
        evidence_indices=list(record["evidence_indices"]),
    )
    passages = [
        PassageProvenance(
            passage_id=p["passage_id"],
            origin=StoreOrigin(p["origin"]),
            store_request_id=p["store_request_id"],
            stored_position=int(p["stored_position"]),
            measured_position=int(p["measured_position"]),
            is_evidence=bool(p["is_evidence"]),
            measured_earlier_in=tuple(p["measured_earlier_in"]),
            pad_tokens=int(p.get("pad_tokens", 0)),
        )
        for p in record["passages"]
    ]
    expected = record.get("expected_reused_tokens")
    position = record.get("unit_position")
    return SamplePlan(
        sample=sample,
        request_id=record["request_id"],
        messages=record["messages"],
        passages=passages,
        evidence_source=EvidenceSource(record["evidence_source"]),
        expected_reused_tokens=None if expected is None else int(expected),
        unit_position=None if position is None else int(position),
        duplicate_of=record.get("duplicate_of"),
    )


def _evidence_indices(sample: Sample) -> tuple[set[int], EvidenceSource]:
    """Decide which of *sample*'s passages hold the answer.

    Args:
        sample: The dataset sample.

    Returns:
        Passage indices (dataset order) and how they were identified.
    """
    if sample.evidence_indices:
        return set(sample.evidence_indices), EvidenceSource.DATASET
    golds = [normalize_answer(a) for a in sample.answers]
    found = {
        i
        for i, passage in enumerate(sample.documents)
        if any(g and f" {g} " in f" {normalize_answer(passage)} " for g in golds)
    }
    if found:
        return found, EvidenceSource.ANSWER_STRING
    return set(), EvidenceSource.NONE


def _measured_order(sample: Sample, scenario: ReuseScenario, seed: int) -> list[int]:
    """Return the order *sample*'s passages are presented in.

    ``INDEPENDENT`` keeps dataset order. ``REORDERED`` applies a
    permutation seeded by the sample id, never the identity when there is
    more than one passage, so every question's passages move.

    Args:
        sample: The dataset sample.
        scenario: The reuse scenario.
        seed: The run seed.

    Returns:
        Passage indices in presentation order.
    """
    order = list(range(len(sample.documents)))
    if scenario is ReuseScenario.INDEPENDENT or len(order) < 2:
        return order
    random.Random(f"{seed}:{sample.sample_id}").shuffle(order)
    if order == sorted(order):
        order = order[1:] + order[:1]
    return order


def build_plan(
    samples: list[Sample],
    aligner: ChunkAligner,
    scenario: ReuseScenario,
    chunk_size: int,
    seed: int,
    unit_samples: list[Sample] | None = None,
    min_unit_tokens: int = 8192,
) -> QualityPlan:
    """Build the store requests, measured prompts, and manifest for a run.

    ``INDEPENDENT`` stores every distinct passage once, alone, in order of
    first appearance. ``REORDERED`` stores each question's passages
    together in dataset order (identical passage lists are stored once), and
    measures them in a new order. ``CONCATENATED`` is described in
    :func:`_build_concatenated`. Either way, every passage a question uses
    is stored before any question is measured, so measuring a subset of the
    dataset does not change what the store requests provide. A passage that
    also appeared in an earlier measured question may instead be reused from
    that question's cached prompt (recorded by ``measured_earlier_in``).

    For ``INDEPENDENT`` and ``REORDERED``, every prompt is then rendered
    with the chat template and tokenized as the engine will, and every
    passage must start on a chunk boundary and tokenize identically wherever
    it appears (see :meth:`ChunkAligner.place_blocks`). Otherwise the plan is
    refused: the run would silently measure less reuse than it claims.

    Every question also gets ``expected_reused_tokens``: how much of its
    rendered prompt is covered by whole chunks of the rendered store
    requests, found at any offset -- the reuse an engine like the LMCache
    blend server can find (see :func:`_expected_reuse`).

    Args:
        samples: Questions to measure, in order.
        aligner: Pads blocks to *chunk_size*. Must have been built with it.
        scenario: How passages are cached.
        chunk_size: The chunk size, recorded in the manifest.
        seed: Seed for the re-ordering permutations and the
            ``concatenated`` shuffle.
        unit_samples: ``concatenated`` only: every question units are built
            from, with *samples* as a prefix, so measuring fewer questions
            does not change the units. Defaults to *samples*.
        min_unit_tokens: ``concatenated`` only: the minimum unit size.

    Returns:
        The plan.

    Raises:
        ValueError: If *samples* is empty, or is not a prefix of
            *unit_samples*.
        PlanAlignmentError: If any passage would not be reused.
    """
    if not samples:
        raise ValueError("samples must be non-empty")
    if scenario is ReuseScenario.CONCATENATED:
        return _build_concatenated(
            samples,
            unit_samples if unit_samples is not None else samples,
            aligner,
            chunk_size,
            seed,
            min_unit_tokens,
        )

    prompt = ANSWER_TAG_PROMPT
    system_block = aligner.system_block(prompt.system)
    store_requests: list[StoreRequest] = []
    # Passage ids and blocks each request holds, in order, for the check.
    request_blocks: dict[str, tuple[list[str], list[str]]] = {}
    # Store request id and position per stored unit, so repeats reuse them.
    stored_alone: dict[str, str] = {}
    stored_lists: dict[tuple[str, ...], str] = {}
    # Measured request ids so far, per passage id, in measured order.
    measured_in: dict[str, list[str]] = {}
    sample_plans: list[SamplePlan] = []

    for index, sample in enumerate(samples):
        measured_id = f"{_MEASURED_REQUEST_PREFIX}{index}"
        blocks = [aligner.passage_block(p) for p in sample.documents]
        ids = [_passage_id(p) for p in sample.documents]

        if scenario is ReuseScenario.INDEPENDENT:
            for passage_id, block in zip(ids, blocks, strict=True):
                if passage_id not in stored_alone:
                    request_id = f"{_STORE_REQUEST_PREFIX}{len(store_requests)}"
                    store_requests.append(
                        StoreRequest(
                            request_id, compose_store_messages(system_block, [block])
                        )
                    )
                    request_blocks[request_id] = ([passage_id], [block])
                    stored_alone[passage_id] = request_id
        else:
            key = tuple(ids)
            if key not in stored_lists:
                request_id = f"{_STORE_REQUEST_PREFIX}{len(store_requests)}"
                store_requests.append(
                    StoreRequest(
                        request_id, compose_store_messages(system_block, blocks)
                    )
                )
                request_blocks[request_id] = (list(ids), list(blocks))
                stored_lists[key] = request_id

        order = _measured_order(sample, scenario, seed)
        evidence, evidence_source = _evidence_indices(sample)
        provenance = []
        for measured_position, stored_index in enumerate(order):
            if scenario is ReuseScenario.INDEPENDENT:
                origin = StoreOrigin.ALONE
                store_id = stored_alone[ids[stored_index]]
                stored_position = 0
            else:
                origin = StoreOrigin.WITH_SAMPLE_PASSAGES
                store_id = stored_lists[tuple(ids)]
                stored_position = stored_index
            provenance.append(
                PassageProvenance(
                    passage_id=ids[stored_index],
                    origin=origin,
                    store_request_id=store_id,
                    stored_position=stored_position,
                    measured_position=measured_position,
                    is_evidence=stored_index in evidence,
                    measured_earlier_in=tuple(measured_in.get(ids[stored_index], [])),
                    pad_tokens=aligner.pad_tokens(sample.documents[stored_index]),
                )
            )
        # dict.fromkeys: a passage repeated within one question counts once.
        for passage_id in dict.fromkeys(ids):
            measured_in.setdefault(passage_id, []).append(measured_id)

        sample_plans.append(
            SamplePlan(
                sample=sample,
                request_id=measured_id,
                messages=compose_qa_messages(
                    system_block,
                    [blocks[i] for i in order],
                    sample.question,
                    prompt.question_template,
                ),
                passages=provenance,
                evidence_source=evidence_source,
            )
        )
        request_blocks[measured_id] = (
            [ids[i] for i in order],
            [blocks[i] for i in order],
        )

    padding = _check_placements(aligner, store_requests, sample_plans, request_blocks)
    expected, _ = _expected_reuse(aligner, chunk_size, store_requests, sample_plans)
    return QualityPlan(
        scenario=scenario,
        chunk_size=chunk_size,
        store_requests=store_requests,
        samples=[
            replace(s, expected_reused_tokens=e)
            for s, e in zip(sample_plans, expected, strict=True)
        ],
        padding=padding,
    )


def _context_blocks(aligner: ChunkAligner, sample: Sample) -> list[str]:
    """A question's passages in the unpadded framed layout, dataset order."""
    return [aligner.framed(p) for p in sample.documents]


def _build_concatenated(
    samples: list[Sample],
    unit_samples: list[Sample],
    aligner: ChunkAligner,
    chunk_size: int,
    seed: int,
    min_unit_tokens: int,
) -> QualityPlan:
    """Build a ``CONCATENATED`` plan.

    Every question in *unit_samples* is visited in a seeded shuffle. Its
    context -- its passages, framed and in request order -- joins the
    current unit until the unit reaches *min_unit_tokens*; a context
    identical to one already placed is not placed again
    (``duplicate_of``). Units are stored without the system prompt, so a
    question's context is never a prefix of its unit: every reuse is a
    blend, at a new position, of KV computed after other contexts. Only
    units holding a measured question's context are sent, and they do not
    depend on how many questions are measured.

    Nothing is padded. A stored chunk is reusable only where all its tokens
    recur in the measured prompt, so chunks straddling a context's edges are
    recomputed -- ``expected_reused_tokens`` details expected reuse.

    Args:
        samples: Questions to measure, a prefix of *unit_samples*.
        unit_samples: Every question units are built from.
        aligner: Renders and tokenizes prompts.
        chunk_size: The engine's chunk size.
        seed: Seed for the shuffle.
        min_unit_tokens: Minimum unit size, in tokens.

    Returns:
        The plan.

    Raises:
        ValueError: If *samples* is not a prefix of *unit_samples*.
    """
    if unit_samples[: len(samples)] != samples:
        raise ValueError("the measured samples must be a prefix of the unit samples")
    prompt = ANSWER_TAG_PROMPT

    order = list(range(len(unit_samples)))
    random.Random(f"{seed}:units").shuffle(order)
    owner_of: dict[tuple[str, ...], int] = {}
    units: list[list[int]] = []
    current: list[int] = []
    size = 0
    for index in order:
        key = tuple(unit_samples[index].documents)
        if key in owner_of:
            continue
        owner_of[key] = index
        current.append(index)
        size += aligner.token_length(
            "".join(_context_blocks(aligner, unit_samples[index]))
        )
        if size >= min_unit_tokens:
            units.append(current)
            current, size = [], 0
    if current:
        units.append(current)
    unit_of = {
        member: (unit, position)
        for unit, members in enumerate(units)
        for position, member in enumerate(members)
    }

    owners = [owner_of[tuple(sample.documents)] for sample in samples]
    store_requests: list[StoreRequest] = []
    unit_request: dict[int, str] = {}
    for unit in sorted({unit_of[owner][0] for owner in owners}):
        request_id = f"{_STORE_REQUEST_PREFIX}{len(store_requests)}"
        text = "".join(
            block
            for member in units[unit]
            for block in _context_blocks(aligner, unit_samples[member])
        )
        store_requests.append(
            StoreRequest(request_id, [{"role": "user", "content": text}])
        )
        unit_request[unit] = request_id

    measured_in: dict[str, list[str]] = {}
    sample_plans: list[SamplePlan] = []
    for index, (sample, owner) in enumerate(zip(samples, owners, strict=True)):
        measured_id = f"{_MEASURED_REQUEST_PREFIX}{index}"
        unit, position = unit_of[owner]
        offset = sum(
            len(unit_samples[member].documents) for member in units[unit][:position]
        )
        ids = [_passage_id(p) for p in sample.documents]
        evidence, evidence_source = _evidence_indices(sample)
        provenance = [
            PassageProvenance(
                passage_id=passage_id,
                origin=StoreOrigin.IN_UNIT,
                store_request_id=unit_request[unit],
                stored_position=offset + measured_position,
                measured_position=measured_position,
                is_evidence=measured_position in evidence,
                measured_earlier_in=tuple(measured_in.get(passage_id, [])),
            )
            for measured_position, passage_id in enumerate(ids)
        ]
        for passage_id in dict.fromkeys(ids):
            measured_in.setdefault(passage_id, []).append(measured_id)
        sample_plans.append(
            SamplePlan(
                sample=sample,
                request_id=measured_id,
                messages=compose_qa_messages(
                    prompt.system,
                    _context_blocks(aligner, sample),
                    sample.question,
                    prompt.question_template,
                ),
                passages=provenance,
                evidence_source=evidence_source,
                unit_position=position,
                duplicate_of=(
                    unit_samples[owner].sample_id if owner != index else None
                ),
            )
        )

    expected, prompt_tokens = _expected_reuse(
        aligner, chunk_size, store_requests, sample_plans
    )
    return QualityPlan(
        scenario=ReuseScenario.CONCATENATED,
        chunk_size=chunk_size,
        store_requests=store_requests,
        samples=[
            replace(s, expected_reused_tokens=e)
            for s, e in zip(sample_plans, expected, strict=True)
        ],
        padding=PaddingStats("none", 0, prompt_tokens),
    )


# Leading tokens a stored chunk is first looked up by, before a full compare.
_PROBE_TOKENS = 8


def _expected_reuse(
    aligner: ChunkAligner,
    chunk_size: int,
    store_requests: list[StoreRequest],
    sample_plans: list[SamplePlan],
) -> tuple[list[int], int]:
    """Count each measured prompt's tokens covered by a whole stored chunk.

    Mirrors the LMCache blend server's matcher: each rendered store request
    is cut into full chunks from its first token (a shorter tail is not
    indexed), and a chunk is reusable wherever all its tokens recur, in
    order, at any offset of the rendered measured prompt. Chunks the
    candidate caches from earlier measured prompts are not counted.

    Args:
        aligner: Renders and tokenizes prompts as the engine will.
        chunk_size: The engine's chunk size.
        store_requests: The plan's store requests.
        sample_plans: The plan's measured requests.

    Returns:
        Covered tokens per measured prompt, and the measured prompts' total
        token count.
    """
    probe = min(_PROBE_TOKENS, chunk_size)
    chunks: dict[tuple[int, ...], set[tuple[int, ...]]] = {}
    for request in store_requests:
        _, ids = aligner.render_and_encode(request.messages)
        for start in range(0, len(ids) - chunk_size + 1, chunk_size):
            chunk = tuple(ids[start : start + chunk_size])
            chunks.setdefault(chunk[:probe], set()).add(chunk)

    covered_counts: list[int] = []
    prompt_tokens = 0
    for sample_plan in sample_plans:
        _, ids = aligner.render_and_encode(sample_plan.messages)
        prompt_tokens += len(ids)
        covered = bytearray(len(ids))
        for start in range(len(ids) - chunk_size + 1):
            candidates = chunks.get(tuple(ids[start : start + probe]))
            if candidates and tuple(ids[start : start + chunk_size]) in candidates:
                covered[start : start + chunk_size] = b"\x01" * chunk_size
        covered_counts.append(sum(covered))
    return covered_counts, prompt_tokens


def _check_placements(
    aligner: ChunkAligner,
    store_requests: list[StoreRequest],
    sample_plans: list[SamplePlan],
    request_blocks: Mapping[str, tuple[list[str], list[str]]],
) -> PaddingStats:
    """Check every passage's placement in every rendered prompt.

    Args:
        aligner: The aligner that padded the blocks.
        store_requests: The plan's store requests.
        sample_plans: The plan's measured requests.
        request_blocks: Per request id, its passage ids and blocks in order.

    Returns:
        Filler statistics over the measured prompts.

    Raises:
        PlanAlignmentError: If any passage is misplaced in any prompt, or
            tokenizes differently in two of them.
    """
    problems: list[str] = []
    first_seen: dict[str, tuple[str, tuple[int, ...]]] = {}
    prompt_tokens = 0
    requests = [(r.request_id, r.messages) for r in store_requests] + [
        (s.request_id, s.messages) for s in sample_plans
    ]
    for request_id, messages in requests:
        passage_ids, blocks = request_blocks[request_id]
        num_tokens, placements = aligner.place_blocks(messages, blocks)
        if not request_id.startswith(_STORE_REQUEST_PREFIX):
            prompt_tokens += num_tokens
        for passage_id, placement in zip(passage_ids, placements, strict=True):
            if placement.problem:
                problems.append(
                    f"passage {passage_id} in {request_id}: {placement.problem}"
                )
                continue
            seen = first_seen.setdefault(passage_id, (request_id, placement.tokens))
            if seen[1] != placement.tokens:
                problems.append(
                    f"passage {passage_id}: tokenizes differently in "
                    f"{seen[0]} and {request_id}, so their chunks differ"
                )
    if problems:
        raise PlanAlignmentError(problems, aligner.filler)
    pad_tokens = sum(
        aligner.system_pad_tokens + sum(p.pad_tokens for p in s.passages)
        for s in sample_plans
    )
    return PaddingStats(aligner.filler.value, pad_tokens, prompt_tokens)


__all__ = [
    "EvidenceSource",
    "PaddingStats",
    "PlanAlignmentError",
    "PassageProvenance",
    "QualityPlan",
    "SamplePlan",
    "StoreOrigin",
    "StoreRequest",
    "build_plan",
]
