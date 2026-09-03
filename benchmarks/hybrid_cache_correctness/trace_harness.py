# SPDX-License-Identifier: Apache-2.0
"""Deterministic trace replay and first-divergence reporting for hybrid caches."""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
import argparse
import hashlib
import json
import math
import re

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_LEVEL_ORDER = {"output": 0, "logits": 1, "request": 2, "cache": 3, "lifecycle": 4}


class LifecyclePhase(str, Enum):
    """Lifecycle events relevant to asynchronous hybrid-cache correctness."""

    STORE_SUBMITTED = "store_submitted"
    STORE_COMPLETE = "store_complete"
    RETRIEVE_SUBMITTED = "retrieve_submitted"
    RETRIEVE_COMPLETE = "retrieve_complete"
    SOURCE_REUSABLE = "source_reusable"
    REQUEST_ABORTED = "request_aborted"
    REQUEST_PREEMPTED = "request_preempted"
    REQUEST_RESUMED = "request_resumed"


def _require_digest(name: str, value: str) -> None:
    if not _DIGEST_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a sha256 digest")


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class TopKEntry:
    """One token and log-probability from a frame's top-k output."""

    token_id: int
    logprob: float

    def __post_init__(self) -> None:
        if self.token_id < 0:
            raise ValueError("token_id must be non-negative")
        _require_finite("logprob", self.logprob)


@dataclass(frozen=True)
class OutputFrame:
    """Output-level evidence for one decode step."""

    token_id: int
    logprob: float
    top_k: tuple[TopKEntry, ...]
    logits: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.token_id < 0:
            raise ValueError("token_id must be non-negative")
        _require_finite("logprob", self.logprob)
        top_k = tuple(self.top_k)
        if len({entry.token_id for entry in top_k}) != len(top_k):
            raise ValueError("top_k token ids must be unique")
        object.__setattr__(self, "top_k", top_k)
        if self.logits is not None:
            logits = tuple(self.logits)
            if not logits:
                raise ValueError("logits must not be empty")
            for value in logits:
                _require_finite("logit", value)
            object.__setattr__(self, "logits", logits)


@dataclass(frozen=True)
class RequestStateFrame:
    """Request state needed to localize scheduling/cache divergence."""

    request_generation: int
    accepted_seq_len: int
    block_table_digest: str
    prefix_digest: str
    drop_round_digest: str

    def __post_init__(self) -> None:
        if self.request_generation < 0:
            raise ValueError("request_generation must be non-negative")
        if self.accepted_seq_len < 0:
            raise ValueError("accepted_seq_len must be non-negative")
        for name in ("block_table_digest", "prefix_digest", "drop_round_digest"):
            _require_digest(name, getattr(self, name))


@dataclass(frozen=True)
class CacheGroupFrame:
    """Per-rank physical and content evidence for one cache group."""

    group_id: str
    rank: int
    semantic_kind: str
    logical_start: int
    logical_end: int
    physical_page_ids: tuple[int, ...]
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    content_digest: str
    revision: str

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id must not be empty")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if not self.semantic_kind:
            raise ValueError("semantic_kind must not be empty")
        if self.logical_start < 0 or self.logical_end <= self.logical_start:
            raise ValueError("logical span must be a non-empty half-open range")
        pages = tuple(self.physical_page_ids)
        if any(page < 0 for page in pages):
            raise ValueError("physical page ids must be non-negative")
        object.__setattr__(self, "physical_page_ids", pages)
        if not self.dtype:
            raise ValueError("dtype must not be empty")
        shape = tuple(self.shape)
        stride = tuple(self.stride)
        if not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError("shape dimensions must be positive")
        if len(stride) != len(shape) or any(value < 0 for value in stride):
            raise ValueError("stride must have one non-negative value per dimension")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "stride", stride)
        _require_digest("content_digest", self.content_digest)
        if not self.revision:
            raise ValueError("revision must not be empty")

    @property
    def identity(self) -> tuple[str, int]:
        """Return the stable group/rank key used during comparison."""
        return self.group_id, self.rank


@dataclass(frozen=True)
class TraceFrame:
    """All layered evidence captured at one decode step."""

    step: int
    output: OutputFrame
    request: RequestStateFrame
    cache_groups: tuple[CacheGroupFrame, ...]

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")
        groups = tuple(sorted(self.cache_groups, key=lambda group: group.identity))
        identities = [group.identity for group in groups]
        if len(identities) != len(set(identities)):
            raise ValueError("cache group/rank identities must be unique per frame")
        object.__setattr__(self, "cache_groups", groups)


@dataclass(frozen=True)
class LifecycleEvent:
    """Ordered lifecycle evidence associated with a request operation."""

    sequence: int
    step: int
    phase: LifecyclePhase
    request_generation: int
    operation_id: int
    group_id: str | None = None
    detail_digest: str | None = None

    def __post_init__(self) -> None:
        for name in ("sequence", "step", "request_generation", "operation_id"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.group_id == "":
            raise ValueError("group_id must not be empty")
        if self.detail_digest is not None:
            _require_digest("detail_digest", self.detail_digest)


@dataclass(frozen=True)
class HybridCorrectnessTrace:
    """Deterministic multi-level correctness trace for one benchmark run."""

    run_id: str
    request_id: str
    frames: tuple[TraceFrame, ...]
    lifecycle_events: tuple[LifecycleEvent, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported trace schema_version")
        if not self.run_id or not self.request_id:
            raise ValueError("run_id and request_id must not be empty")
        frames = tuple(self.frames)
        if not frames:
            raise ValueError("a trace must contain at least one frame")
        steps = [frame.step for frame in frames]
        if steps != sorted(steps) or len(steps) != len(set(steps)):
            raise ValueError("trace frame steps must be strictly increasing")
        generations = [frame.request.request_generation for frame in frames]
        if generations != sorted(generations):
            raise ValueError("request generations must be monotonic")
        previous_length_by_generation: dict[int, int] = {}
        for frame in frames:
            generation = frame.request.request_generation
            previous = previous_length_by_generation.get(generation, -1)
            if frame.request.accepted_seq_len < previous:
                raise ValueError(
                    "accepted_seq_len must be monotonic within a generation"
                )
            previous_length_by_generation[generation] = frame.request.accepted_seq_len
        object.__setattr__(self, "frames", frames)
        events = tuple(self.lifecycle_events)
        sequences = [event.sequence for event in events]
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            raise ValueError("lifecycle event sequences must be strictly increasing")
        object.__setattr__(self, "lifecycle_events", events)
        metadata = tuple(sorted(self.metadata))
        if any(not key or not value for key, value in metadata):
            raise ValueError("metadata keys and values must not be empty")
        if len({key for key, _ in metadata}) != len(metadata):
            raise ValueError("metadata keys must be unique")
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class TraceDivergence:
    """One localized difference between a reference and candidate trace."""

    level: str
    step: int
    subject: str
    fields: tuple[str, ...]


@dataclass(frozen=True)
class FrameComparison:
    """Numerical output metrics for one common decode step."""

    step: int
    token_match: bool
    logprob_abs_diff: float
    top_k_overlap: float
    logits_max_abs_diff: float | None
    logits_mean_abs_diff: float | None
    logits_kl_divergence: float | None
    logits_cosine_similarity: float | None


@dataclass(frozen=True)
class TraceComparisonReport:
    """Deterministic layered comparison with a first-divergence pointer."""

    reference_digest: str
    candidate_digest: str
    matched: bool
    first_divergence: TraceDivergence | None
    divergences: tuple[TraceDivergence, ...]
    frame_metrics: tuple[FrameComparison, ...]


def sha256_digest(data: bytes | bytearray | memoryview) -> str:
    """Return the canonical digest used for cache content and trace identity."""
    return "sha256:" + hashlib.sha256(bytes(data)).hexdigest()


def trace_digest(trace: HybridCorrectnessTrace) -> str:
    """Return a stable digest of the canonical serialized trace."""
    return sha256_digest(_canonical_json(_trace_to_dict(trace)).encode())


def write_trace(trace: HybridCorrectnessTrace, path: Path) -> None:
    """Write a canonical trace document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(_trace_to_dict(trace)) + "\n", encoding="utf-8")


def read_trace(path: Path) -> HybridCorrectnessTrace:
    """Read and validate a trace document."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("trace document must contain a JSON object")
    return _trace_from_dict(payload)


def write_report(report: TraceComparisonReport, path: Path) -> None:
    """Write a canonical comparison report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(asdict(report)) + "\n", encoding="utf-8")


def compare_traces(
    reference: HybridCorrectnessTrace,
    candidate: HybridCorrectnessTrace,
    *,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-4,
) -> TraceComparisonReport:
    """Compare all trace levels and report the earliest localized divergence."""
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("comparison tolerances must be non-negative")
    divergences: list[TraceDivergence] = []
    metrics: list[FrameComparison] = []
    reference_frames = {frame.step: frame for frame in reference.frames}
    candidate_frames = {frame.step: frame for frame in candidate.frames}

    if reference.request_id != candidate.request_id:
        divergences.append(
            TraceDivergence(
                level="request",
                step=min(reference.frames[0].step, candidate.frames[0].step),
                subject="trace",
                fields=("request_id",),
            )
        )

    for step in sorted(reference_frames.keys() | candidate_frames.keys()):
        reference_frame = reference_frames.get(step)
        candidate_frame = candidate_frames.get(step)
        if reference_frame is None or candidate_frame is None:
            divergences.append(
                TraceDivergence(
                    level="output",
                    step=step,
                    subject="frame",
                    fields=(
                        "missing_reference"
                        if reference_frame is None
                        else "missing_candidate",
                    ),
                )
            )
            continue
        frame_metrics, frame_divergences = _compare_frame(
            reference_frame,
            candidate_frame,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        metrics.append(frame_metrics)
        divergences.extend(frame_divergences)

    if reference.lifecycle_events != candidate.lifecycle_events:
        mismatch_index = _first_mismatch_index(
            reference.lifecycle_events, candidate.lifecycle_events
        )
        reference_event = (
            reference.lifecycle_events[mismatch_index]
            if mismatch_index < len(reference.lifecycle_events)
            else None
        )
        candidate_event = (
            candidate.lifecycle_events[mismatch_index]
            if mismatch_index < len(candidate.lifecycle_events)
            else None
        )
        event = reference_event or candidate_event
        assert event is not None
        divergences.append(
            TraceDivergence(
                level="lifecycle",
                step=event.step,
                subject=f"event[{mismatch_index}]",
                fields=_different_fields(reference_event, candidate_event),
            )
        )

    ordered = tuple(
        sorted(
            divergences,
            key=lambda item: (item.step, _LEVEL_ORDER[item.level], item.subject),
        )
    )
    return TraceComparisonReport(
        reference_digest=trace_digest(reference),
        candidate_digest=trace_digest(candidate),
        matched=not ordered,
        first_divergence=ordered[0] if ordered else None,
        divergences=ordered,
        frame_metrics=tuple(metrics),
    )


def _compare_frame(
    reference: TraceFrame,
    candidate: TraceFrame,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[FrameComparison, tuple[TraceDivergence, ...]]:
    divergences: list[TraceDivergence] = []
    output_fields: list[str] = []
    if reference.output.token_id != candidate.output.token_id:
        output_fields.append("token_id")
    logprob_diff = abs(reference.output.logprob - candidate.output.logprob)
    if not _is_close(
        reference.output.logprob,
        candidate.output.logprob,
        absolute_tolerance,
        relative_tolerance,
    ):
        output_fields.append("logprob")
    reference_top_k = {entry.token_id for entry in reference.output.top_k}
    candidate_top_k = {entry.token_id for entry in candidate.output.top_k}
    denominator = max(len(reference_top_k), len(candidate_top_k), 1)
    top_k_overlap = len(reference_top_k & candidate_top_k) / denominator
    if reference_top_k != candidate_top_k:
        output_fields.append("top_k")
    if output_fields:
        divergences.append(
            TraceDivergence("output", reference.step, "decode", tuple(output_fields))
        )

    logits_metrics, logits_fields = _compare_logits(
        reference.output.logits,
        candidate.output.logits,
        absolute_tolerance,
        relative_tolerance,
    )
    if logits_fields:
        divergences.append(
            TraceDivergence("logits", reference.step, "decode", logits_fields)
        )

    request_fields = _different_dataclass_fields(reference.request, candidate.request)
    if request_fields:
        divergences.append(
            TraceDivergence("request", reference.step, "request", request_fields)
        )

    reference_groups = {group.identity: group for group in reference.cache_groups}
    candidate_groups = {group.identity: group for group in candidate.cache_groups}
    for identity in sorted(reference_groups.keys() | candidate_groups.keys()):
        reference_group = reference_groups.get(identity)
        candidate_group = candidate_groups.get(identity)
        fields = _different_fields(reference_group, candidate_group)
        if fields:
            divergences.append(
                TraceDivergence(
                    "cache",
                    reference.step,
                    f"{identity[0]}@rank{identity[1]}",
                    fields,
                )
            )

    return (
        FrameComparison(
            step=reference.step,
            token_match=reference.output.token_id == candidate.output.token_id,
            logprob_abs_diff=logprob_diff,
            top_k_overlap=top_k_overlap,
            logits_max_abs_diff=logits_metrics[0],
            logits_mean_abs_diff=logits_metrics[1],
            logits_kl_divergence=logits_metrics[2],
            logits_cosine_similarity=logits_metrics[3],
        ),
        tuple(divergences),
    )


def _compare_logits(
    reference: tuple[float, ...] | None,
    candidate: tuple[float, ...] | None,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[
    tuple[float | None, float | None, float | None, float | None], tuple[str, ...]
]:
    if reference is None and candidate is None:
        return (None, None, None, None), ()
    if reference is None or candidate is None:
        return (None, None, None, None), ("presence",)
    if len(reference) != len(candidate):
        return (None, None, None, None), ("length",)
    differences = [
        abs(left - right) for left, right in zip(reference, candidate, strict=True)
    ]
    maximum = max(differences, default=0.0)
    mean = sum(differences) / len(differences)
    reference_log_probabilities = _log_softmax(reference)
    candidate_log_probabilities = _log_softmax(candidate)
    kl_divergence = sum(
        math.exp(left) * (left - right)
        for left, right in zip(
            reference_log_probabilities,
            candidate_log_probabilities,
            strict=True,
        )
    )
    dot = sum(left * right for left, right in zip(reference, candidate, strict=True))
    reference_norm = math.sqrt(sum(value * value for value in reference))
    candidate_norm = math.sqrt(sum(value * value for value in candidate))
    if reference_norm == 0 and candidate_norm == 0:
        cosine = 1.0
    elif reference_norm == 0 or candidate_norm == 0:
        cosine = 0.0
    else:
        cosine = max(-1.0, min(1.0, dot / (reference_norm * candidate_norm)))
    fields = (
        ("values",)
        if any(
            not _is_close(left, right, absolute_tolerance, relative_tolerance)
            for left, right in zip(reference, candidate, strict=True)
        )
        else ()
    )
    return (maximum, mean, kl_divergence, cosine), fields


def _log_softmax(values: tuple[float, ...]) -> tuple[float, ...]:
    maximum = max(values)
    exponentials = tuple(math.exp(value - maximum) for value in values)
    log_total = maximum + math.log(sum(exponentials))
    return tuple(value - log_total for value in values)


def _is_close(left: float, right: float, absolute: float, relative: float) -> bool:
    return abs(left - right) <= absolute + relative * abs(left)


def _different_dataclass_fields(reference: Any, candidate: Any) -> tuple[str, ...]:
    reference_values = asdict(reference)
    candidate_values = asdict(candidate)
    return tuple(
        key
        for key in reference_values
        if reference_values[key] != candidate_values[key]
    )


def _different_fields(
    reference: object | None, candidate: object | None
) -> tuple[str, ...]:
    if reference is None:
        return ("missing_reference",)
    if candidate is None:
        return ("missing_candidate",)
    return _different_dataclass_fields(reference, candidate)


def _first_mismatch_index(
    reference: tuple[Any, ...], candidate: tuple[Any, ...]
) -> int:
    for index, (left, right) in enumerate(zip(reference, candidate, strict=False)):
        if left != right:
            return index
    return min(len(reference), len(candidate))


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _trace_to_dict(trace: HybridCorrectnessTrace) -> dict[str, object]:
    payload = asdict(trace)
    payload["metadata"] = dict(trace.metadata)
    return payload


def _trace_from_dict(payload: Mapping[str, Any]) -> HybridCorrectnessTrace:
    frames = tuple(
        TraceFrame(
            step=int(frame["step"]),
            output=OutputFrame(
                token_id=int(frame["output"]["token_id"]),
                logprob=float(frame["output"]["logprob"]),
                top_k=tuple(
                    TopKEntry(
                        token_id=int(entry["token_id"]), logprob=float(entry["logprob"])
                    )
                    for entry in frame["output"]["top_k"]
                ),
                logits=(
                    tuple(float(value) for value in frame["output"]["logits"])
                    if frame["output"]["logits"] is not None
                    else None
                ),
            ),
            request=RequestStateFrame(**frame["request"]),
            cache_groups=tuple(
                CacheGroupFrame(**group) for group in frame["cache_groups"]
            ),
        )
        for frame in payload["frames"]
    )
    events = tuple(
        LifecycleEvent(
            sequence=int(event["sequence"]),
            step=int(event["step"]),
            phase=LifecyclePhase(event["phase"]),
            request_generation=int(event["request_generation"]),
            operation_id=int(event["operation_id"]),
            group_id=event.get("group_id"),
            detail_digest=event.get("detail_digest"),
        )
        for event in payload.get("lifecycle_events", [])
    )
    metadata_payload = payload.get("metadata", {})
    if not isinstance(metadata_payload, dict):
        raise ValueError("trace metadata must be a JSON object")
    return HybridCorrectnessTrace(
        run_id=str(payload["run_id"]),
        request_id=str(payload["request_id"]),
        frames=frames,
        lifecycle_events=events,
        metadata=tuple(
            (str(key), str(value)) for key, value in metadata_payload.items()
        ),
        schema_version=str(payload.get("schema_version", "")),
    )


def main() -> None:
    """Compare two trace files from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-5)
    parser.add_argument("--relative-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    report = compare_traces(
        read_trace(args.reference),
        read_trace(args.candidate),
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    rendered = _canonical_json(asdict(report))
    if args.output is not None:
        write_report(report, args.output)
    print(rendered)
    raise SystemExit(0 if report.matched else 1)


if __name__ == "__main__":
    main()
