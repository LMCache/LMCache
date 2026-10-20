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
from dataclasses import dataclass
from typing import Any
import enum
import hashlib
import json
import random

# First Party
from lmcache.cli.commands.bench.engine_bench.quality.alignment import ChunkAligner
from lmcache.cli.commands.bench.engine_bench.quality.dataset import Sample
from lmcache.cli.commands.bench.engine_bench.quality.prompts import (
    QA_SYSTEM_PROMPT,
    compose_qa_messages,
    compose_store_messages,
)
from lmcache.cli.commands.bench.engine_bench.quality.scoring import normalize_answer
from lmcache.cli.commands.bench.quality_bench.core.config import ReuseScenario

_STORE_REQUEST_PREFIX = "store"
_MEASURED_REQUEST_PREFIX = "q"


class StoreOrigin(enum.Enum):
    """Where a passage's cached KV was computed."""

    ALONE = "alone"
    """Stored by a request holding only this passage (after the system
    block)."""

    WITH_SAMPLE_PASSAGES = "with_sample_passages"
    """Stored together with the rest of this question's passages, in dataset
    order, so its cached KV saw different neighbors than it has when
    measured."""


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
    """

    passage_id: str
    origin: StoreOrigin
    store_request_id: str
    stored_position: int
    measured_position: int
    is_evidence: bool
    measured_earlier_in: tuple[str, ...] = ()


@dataclass(frozen=True)
class SamplePlan:
    """The measured request for one question, with its reuse manifest.

    Attributes:
        sample: The dataset sample.
        request_id: Identifier of the measured request (same on every arm).
        messages: The measured chat messages. Byte-identical on every arm.
        passages: Provenance of each passage, in measured order.
        evidence_source: How ``is_evidence`` was decided.
    """

    sample: Sample
    request_id: str
    messages: list[dict[str, str]]
    passages: list[PassageProvenance]
    evidence_source: EvidenceSource

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
            ``evidence_source``, ``evidence_passage_ids`` and ``passages``
            (one dict per passage: ``passage_id``, ``origin``,
            ``store_request_id``, ``stored_position``, ``measured_position``,
            ``is_evidence``, ``measured_earlier_in``).
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
            "passages": [
                {
                    "passage_id": p.passage_id,
                    "origin": p.origin.value,
                    "store_request_id": p.store_request_id,
                    "stored_position": p.stored_position,
                    "measured_position": p.measured_position,
                    "is_evidence": p.is_evidence,
                    "measured_earlier_in": list(p.measured_earlier_in),
                }
                for p in self.passages
            ],
        }


@dataclass(frozen=True)
class QualityPlan:
    """Everything every arm sends, in order.

    Attributes:
        scenario: How passages are cached.
        chunk_size: The chunk size passages are aligned to.
        store_requests: Requests to send before measuring, in order.
        samples: One measured request per question, in dataset order.
    """

    scenario: ReuseScenario
    chunk_size: int
    store_requests: list[StoreRequest]
    samples: list[SamplePlan]

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
        return {
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
            plan = cls(
                scenario=scenario,
                chunk_size=int(record["chunk_size"]),
                store_requests=[
                    StoreRequest(r["request_id"], r["messages"])
                    for r in record["store_requests"]
                ],
                samples=[_sample_plan_from_record(s) for s in record["samples"]],
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
        )
        for p in record["passages"]
    ]
    return SamplePlan(
        sample=sample,
        request_id=record["request_id"],
        messages=record["messages"],
        passages=passages,
        evidence_source=EvidenceSource(record["evidence_source"]),
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
) -> QualityPlan:
    """Build the store requests, measured prompts, and manifest for a run.

    ``INDEPENDENT`` stores every distinct passage once, alone, in order of
    first appearance. ``REORDERED`` stores each question's passages
    together in dataset order (identical passage lists are stored once), and
    measures them in a new order. Either way, every passage a question uses
    is stored before any question is measured, so measuring a subset of the
    dataset does not change what the store requests provide. A passage that
    also appeared in an earlier measured question may instead be reused from
    that question's cached prompt (recorded by ``measured_earlier_in``).

    Args:
        samples: Questions to measure, in order.
        aligner: Pads blocks to *chunk_size*. Must have been built with it.
        scenario: How passages are cached.
        chunk_size: The chunk size, recorded in the manifest.
        seed: Seed for the re-ordering permutations.

    Returns:
        The plan.

    Raises:
        ValueError: If *samples* is empty.
    """
    if not samples:
        raise ValueError("samples must be non-empty")

    system_block = aligner.system_block(QA_SYSTEM_PROMPT)
    store_requests: list[StoreRequest] = []
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
                    system_block, [blocks[i] for i in order], sample.question
                ),
                passages=provenance,
                evidence_source=evidence_source,
            )
        )

    return QualityPlan(
        scenario=scenario,
        chunk_size=chunk_size,
        store_requests=store_requests,
        samples=sample_plans,
    )


__all__ = [
    "EvidenceSource",
    "PassageProvenance",
    "QualityPlan",
    "SamplePlan",
    "StoreOrigin",
    "StoreRequest",
    "build_plan",
]
