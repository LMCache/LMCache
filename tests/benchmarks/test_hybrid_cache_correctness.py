# SPDX-License-Identifier: Apache-2.0
"""Tests for deterministic hybrid-cache trace comparison."""

# Standard
from dataclasses import replace
from pathlib import Path
import json

# Third Party
import pytest

# First Party
from benchmarks.hybrid_cache_correctness import (
    CacheGroupFrame,
    HybridCorrectnessTrace,
    LifecycleEvent,
    LifecyclePhase,
    OutputFrame,
    RequestStateFrame,
    TopKEntry,
    TraceFrame,
    compare_traces,
    read_trace,
    sha256_digest,
    trace_digest,
    write_report,
    write_trace,
)


def _digest(label: str) -> str:
    return sha256_digest(label.encode())


def _group(
    group_id: str = "main",
    *,
    rank: int = 0,
    content: str = "main-content",
) -> CacheGroupFrame:
    return CacheGroupFrame(
        group_id=group_id,
        rank=rank,
        semantic_kind="compressed_dense",
        logical_start=0,
        logical_end=16,
        physical_page_ids=(4, 5),
        dtype="bfloat16",
        shape=(2, 2, 8, 128),
        stride=(2048, 1024, 128, 1),
        content_digest=_digest(content),
        revision="kv-v3",
    )


def _frame(
    step: int = 0,
    *,
    token_id: int = 7,
    logprob: float = -0.2,
    logits: tuple[float, ...] | None = (0.1, 0.2, 0.3),
    groups: tuple[CacheGroupFrame, ...] | None = None,
    accepted_seq_len: int = 16,
) -> TraceFrame:
    return TraceFrame(
        step=step,
        output=OutputFrame(
            token_id=token_id,
            logprob=logprob,
            top_k=(TopKEntry(7, -0.2), TopKEntry(9, -0.4)),
            logits=logits,
        ),
        request=RequestStateFrame(
            request_generation=3,
            accepted_seq_len=accepted_seq_len,
            block_table_digest=_digest(f"blocks-{step}"),
            prefix_digest=_digest("prefix"),
            drop_round_digest=_digest(f"drop-{step}"),
        ),
        cache_groups=groups if groups is not None else (_group(),),
    )


def _trace(
    *frames: TraceFrame,
    events: tuple[LifecycleEvent, ...] | None = None,
) -> HybridCorrectnessTrace:
    return HybridCorrectnessTrace(
        run_id="run-1",
        request_id="request-1",
        frames=frames or (_frame(),),
        lifecycle_events=(
            events
            if events is not None
            else (
                LifecycleEvent(
                    sequence=0,
                    step=0,
                    phase=LifecyclePhase.RETRIEVE_SUBMITTED,
                    request_generation=3,
                    operation_id=2,
                    group_id="main",
                    detail_digest=_digest("retrieve"),
                ),
            )
        ),
        metadata=(("backend", "paged"), ("cache_mode", "l1+l2")),
    )


def test_identical_trace_has_no_divergence() -> None:
    """A deterministic replay compares equal at every level."""
    trace = _trace(_frame(0), _frame(1, accepted_seq_len=17))

    report = compare_traces(trace, trace)

    assert report.matched
    assert report.first_divergence is None
    assert not report.divergences
    assert report.reference_digest == report.candidate_digest
    assert all(metric.token_match for metric in report.frame_metrics)
    assert all(metric.logits_max_abs_diff == 0 for metric in report.frame_metrics)
    assert all(metric.logits_kl_divergence == 0 for metric in report.frame_metrics)


def test_small_numerical_differences_respect_tolerance() -> None:
    """Logprob and logits noise within tolerance remains a match."""
    reference = _trace(_frame())
    candidate_frame = replace(
        _frame(),
        output=replace(
            _frame().output,
            logprob=-0.200001,
            logits=(0.100001, 0.199999, 0.300001),
        ),
    )

    report = compare_traces(reference, _trace(candidate_frame))

    assert report.matched
    assert report.frame_metrics[0].logits_max_abs_diff == pytest.approx(1e-6)
    assert report.frame_metrics[0].logits_cosine_similarity == pytest.approx(1.0)


def test_first_token_divergence_is_localized() -> None:
    """The earliest changed token is reported with output-level priority."""
    reference = _trace(_frame(0), _frame(1, accepted_seq_len=17))
    changed = replace(
        _frame(1, accepted_seq_len=17), output=replace(_frame().output, token_id=8)
    )
    candidate = _trace(_frame(0), changed)

    report = compare_traces(reference, candidate)

    assert not report.matched
    assert report.first_divergence is not None
    assert report.first_divergence.level == "output"
    assert report.first_divergence.step == 1
    assert report.first_divergence.fields == ("token_id",)


def test_logits_divergence_reports_numerical_metrics() -> None:
    """Logit drift reports max/mean error, KL, and cosine similarity."""
    reference = _trace(_frame(logits=(0.0, 1.0, 2.0)))
    candidate = _trace(_frame(logits=(0.0, 1.0, 3.0)))

    report = compare_traces(reference, candidate)

    assert report.first_divergence is not None
    assert report.first_divergence.level == "logits"
    metric = report.frame_metrics[0]
    assert metric.logits_max_abs_diff == 1.0
    assert metric.logits_mean_abs_diff == pytest.approx(1 / 3)
    assert metric.logits_kl_divergence is not None
    assert metric.logits_kl_divergence > 0
    assert metric.logits_cosine_similarity is not None
    assert 0 < metric.logits_cosine_similarity < 1


def test_extreme_logits_produce_finite_kl_divergence() -> None:
    """Stable log-softmax avoids divide-by-zero when probabilities underflow."""
    reference = _trace(_frame(logits=(-10_000.0, 0.0)))
    candidate = _trace(_frame(logits=(0.0, -10_000.0)))

    report = compare_traces(reference, candidate)

    divergence = report.frame_metrics[0].logits_kl_divergence
    assert divergence is not None
    assert divergence == pytest.approx(10_000.0)


def test_logits_presence_or_length_mismatch_is_explicit() -> None:
    """Unavailable or differently shaped logits never receive fake metrics."""
    reference = _trace(_frame(logits=None))
    present = compare_traces(reference, _trace(_frame(logits=(1.0, 2.0))))
    different_length = compare_traces(
        _trace(_frame(logits=(1.0, 2.0))),
        _trace(_frame(logits=(1.0, 2.0, 3.0))),
    )

    assert present.first_divergence is not None
    assert present.first_divergence.fields == ("presence",)
    assert present.frame_metrics[0].logits_max_abs_diff is None
    assert different_length.first_divergence is not None
    assert different_length.first_divergence.fields == ("length",)


def test_request_divergence_precedes_cache_divergence_at_same_step() -> None:
    """First-divergence ordering follows output, logits, request, cache."""
    reference = _trace(_frame())
    changed_frame = replace(
        _frame(),
        request=replace(_frame().request, accepted_seq_len=15),
        cache_groups=(_group(content="changed"),),
    )

    report = compare_traces(reference, _trace(changed_frame))

    assert report.first_divergence is not None
    assert report.first_divergence.level == "request"
    assert report.first_divergence.fields == ("accepted_seq_len",)
    assert {item.level for item in report.divergences} == {"request", "cache"}


def test_cache_group_missing_content_and_revision_are_localized() -> None:
    """Per-group/rank evidence identifies missing and incompatible cache state."""
    reference = _trace(_frame(groups=(_group(), _group("indexer"))))
    changed_main = replace(
        _group(),
        content_digest=_digest("changed"),
        revision="kv-v4",
    )
    candidate = _trace(_frame(groups=(changed_main,)))

    report = compare_traces(reference, candidate)

    cache = [item for item in report.divergences if item.level == "cache"]
    assert len(cache) == 2
    assert cache[0].subject == "indexer@rank0"
    assert cache[0].fields == ("missing_candidate",)
    assert cache[1].subject == "main@rank0"
    assert cache[1].fields == ("content_digest", "revision")


def test_top_k_overlap_is_reported() -> None:
    """Top-k set drift is visible independently from the selected token."""
    reference = _trace(_frame())
    changed_output = replace(
        _frame().output,
        top_k=(TopKEntry(7, -0.2), TopKEntry(10, -0.5)),
    )
    report = compare_traces(reference, _trace(replace(_frame(), output=changed_output)))

    assert report.first_divergence is not None
    assert report.first_divergence.fields == ("top_k",)
    assert report.frame_metrics[0].top_k_overlap == 0.5


def test_missing_decode_frame_is_reported() -> None:
    """A truncated replay identifies its first absent step."""
    reference = _trace(_frame(0), _frame(1, accepted_seq_len=17))
    candidate = _trace(_frame(0))

    report = compare_traces(reference, candidate)

    assert report.first_divergence is not None
    assert report.first_divergence.step == 1
    assert report.first_divergence.subject == "frame"
    assert report.first_divergence.fields == ("missing_candidate",)


def test_lifecycle_event_divergence_is_localized() -> None:
    """Async completion order changes are reported at lifecycle level."""
    reference = _trace()
    changed = replace(
        reference.lifecycle_events[0],
        phase=LifecyclePhase.RETRIEVE_COMPLETE,
    )

    report = compare_traces(reference, _trace(events=(changed,)))

    assert report.first_divergence is not None
    assert report.first_divergence.level == "lifecycle"
    assert report.first_divergence.subject == "event[0]"
    assert report.first_divergence.fields == ("phase",)


def test_request_identity_mismatch_is_not_a_match() -> None:
    """Equal frames from another request remain a request-level divergence."""
    reference = _trace()
    candidate = replace(reference, request_id="request-2", run_id="run-2")

    report = compare_traces(reference, candidate)

    assert not report.matched
    assert report.first_divergence is not None
    assert report.first_divergence.level == "request"
    assert report.first_divergence.subject == "trace"
    assert report.first_divergence.fields == ("request_id",)


def test_trace_round_trip_and_digest_are_deterministic(tmp_path: Path) -> None:
    """Canonical JSON replay preserves the complete evidence identity."""
    trace = _trace(_frame(0), _frame(1, accepted_seq_len=17))
    path = tmp_path / "trace.json"

    write_trace(trace, path)
    replayed = read_trace(path)

    assert replayed == trace
    assert trace_digest(replayed) == trace_digest(trace)
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(path.read_text(encoding="utf-8"))["metadata"] == {
        "backend": "paged",
        "cache_mode": "l1+l2",
    }


def test_comparison_report_is_machine_readable(tmp_path: Path) -> None:
    """Reports serialize with the first divergence and stable trace digests."""
    report = compare_traces(_trace(_frame(token_id=7)), _trace(_frame(token_id=8)))
    path = tmp_path / "report.json"

    write_report(report, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert not payload["matched"]
    assert payload["first_divergence"]["level"] == "output"
    assert payload["reference_digest"].startswith("sha256:")


@pytest.mark.parametrize(
    "trace",
    [
        lambda: _trace(_frame(1), _frame(0)),
        lambda: _trace(_frame(0, accepted_seq_len=17), _frame(1, accepted_seq_len=16)),
    ],
)
def test_invalid_replay_order_is_rejected(trace: object) -> None:
    """Non-monotonic steps or accepted lengths cannot masquerade as replay."""
    with pytest.raises(ValueError, match="must be"):
        trace()  # type: ignore[operator]


def test_duplicate_group_rank_and_nonfinite_evidence_are_rejected() -> None:
    """Ambiguous group state and NaN metrics fail during trace construction."""
    with pytest.raises(ValueError, match="identities"):
        _frame(groups=(_group(), _group()))
    with pytest.raises(ValueError, match="finite"):
        _frame(logprob=float("nan"))


def test_sha256_digest_accepts_buffer_protocol_values() -> None:
    """Tensor adapters can hash byte-oriented content without copies in callers."""
    expected = sha256_digest(b"cache-bytes")
    assert sha256_digest(bytearray(b"cache-bytes")) == expected
    assert sha256_digest(memoryview(b"cache-bytes")) == expected


def test_negative_tolerance_is_rejected() -> None:
    """Invalid comparison configuration never produces a report."""
    with pytest.raises(ValueError, match="tolerances"):
        compare_traces(_trace(), _trace(), absolute_tolerance=-1)
