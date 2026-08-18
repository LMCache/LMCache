# SPDX-License-Identifier: Apache-2.0
"""Tests for the rag-qa-quality workload."""

# Standard
from unittest.mock import AsyncMock, MagicMock
import json

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.stats import RequestResult
from lmcache.cli.commands.bench.engine_bench.workloads import rag_qa_quality
from lmcache.cli.commands.bench.engine_bench.workloads.rag_qa_quality import (
    RagQaQualityConfig,
    RagQaQualityWorkload,
    parse_template_kwargs,
)

# Local
from ..fake_tokenizer import make_fake_tokenizer

_CHUNK = rag_qa_quality.DEFAULT_DOC_ALIGN_TOKENS


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _result(request_id: str, successful: bool = True) -> RequestResult:
    return RequestResult(
        request_id=request_id,
        successful=successful,
        ttft=0.25,
        request_latency=1.0,
        num_input_tokens=100,
        num_output_tokens=12,
        decode_speed=10.0,
        submit_time=0.0,
        first_token_time=0.25,
        finish_time=1.0,
        error="" if successful else "boom",
    )


def _write_dataset(tmp_path, records: list) -> str:
    path = tmp_path / "qa.json"
    path.write_text(json.dumps(records))
    return str(path)


_RECORDS = [
    {
        "id": "s0",
        "ctxs": [{"text": "alpha body"}, {"text": "beta body"}],
        "question": "who?",
        "answers": ["Paris"],
    },
    {
        "id": "s1",
        # Shares "alpha body" with s0 — the corpus must prefill it once.
        "ctxs": [{"text": "alpha body"}, {"text": "gamma body"}],
        "question": "where?",
        "answers": ["Berlin"],
    },
]


def _make_workload(
    tmp_path,
    monkeypatch,
    records: list = _RECORDS,
    responses: list = [],  # noqa: B006
    successful: bool = True,
    num_samples: int = 10,
    max_output_length: int = 64,
    doc_align_tokens: int = _CHUNK,
    template_kwargs: dict[str, bool | int | str] = {},  # noqa: B006
    output_path: str = "",
):
    """Build a workload wired to fakes, returning ``(workload, sender)``.

    ``responses`` supplies each measured request's response text in order.
    ``output_path`` defaults to a file under ``tmp_path``, which is not
    known until call time.
    """
    monkeypatch.setattr(rag_qa_quality, "_FILLER_VOCAB_SIZE", 200)
    monkeypatch.setattr(
        rag_qa_quality, "try_load_tokenizer", lambda _name: make_fake_tokenizer()
    )

    config = RagQaQualityConfig.resolve(
        dataset=_write_dataset(tmp_path, records),
        num_samples=num_samples,
        max_output_length=max_output_length,
        doc_align_tokens=doc_align_tokens,
        template_kwargs=template_kwargs,
        output_path=output_path or str(tmp_path / "out.json"),
    )

    sender = MagicMock()
    sender.send_warmup_request = AsyncMock(side_effect=lambda rid, _m: _result(rid))
    pending = list(responses)

    async def send_request(request_id, messages, max_tokens=128):
        result = _result(request_id, successful=successful)
        text = pending.pop(0) if pending else ""
        workload.request_finished(result, text)
        return result

    sender.send_request = AsyncMock(side_effect=send_request)

    workload = RagQaQualityWorkload(
        config=config,
        request_sender=sender,
        stats_collector=MagicMock(),
        progress_monitor=MagicMock(),
        model_name="fake-model",
        seed=7,
    )
    return workload, sender


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestRagQaQualityConfig:
    def test_defaults(self) -> None:
        cfg = RagQaQualityConfig(dataset="musique")
        assert cfg.num_samples == 50
        assert cfg.max_output_length == 1024
        assert cfg.doc_align_tokens == _CHUNK
        assert cfg.template_kwargs == {}

    def test_empty_dataset_rejected(self) -> None:
        with pytest.raises(ValueError, match="dataset must be non-empty"):
            RagQaQualityConfig(dataset="")

    def test_num_samples_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="num_samples must be >= 1"):
            RagQaQualityConfig(dataset="musique", num_samples=0)

    def test_max_output_length_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_output_length must be >= 1"):
            RagQaQualityConfig(dataset="musique", max_output_length=0)

    def test_doc_align_tokens_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="doc_align_tokens must be >= 1"):
            RagQaQualityConfig(dataset="musique", doc_align_tokens=0)

    def test_empty_output_path_rejected(self) -> None:
        with pytest.raises(ValueError, match="output_path must be non-empty"):
            RagQaQualityConfig(dataset="musique", output_path="")

    def test_resolve_copies_template_kwargs(self) -> None:
        """A later mutation of the caller's dict must not change the config."""
        kwargs = {"enable_thinking": False}
        cfg = RagQaQualityConfig.resolve(
            dataset="musique",
            num_samples=5,
            max_output_length=32,
            doc_align_tokens=_CHUNK,
            template_kwargs=kwargs,
            output_path="out.json",
        )
        kwargs["enable_thinking"] = True
        assert cfg.template_kwargs == {"enable_thinking": False}


class TestParseTemplateKwargs:
    def test_coerces_booleans(self) -> None:
        """Templates compare against typed values, not strings."""
        assert parse_template_kwargs(["enable_thinking=false"]) == (
            {"enable_thinking": False}
        )
        assert parse_template_kwargs(["a=True"]) == {"a": True}

    def test_coerces_integers(self) -> None:
        assert parse_template_kwargs(["budget=2048", "offset=-1"]) == (
            {"budget": 2048, "offset": -1}
        )

    def test_keeps_strings(self) -> None:
        assert parse_template_kwargs(["reasoning_effort=high"]) == (
            {"reasoning_effort": "high"}
        )

    def test_empty_list(self) -> None:
        assert parse_template_kwargs([]) == {}

    def test_value_may_contain_equals(self) -> None:
        assert parse_template_kwargs(["a=b=c"]) == {"a": "b=c"}

    @pytest.mark.parametrize("item", ["novalue", "=value"])
    def test_malformed_items_rejected(self, item: str) -> None:
        with pytest.raises(ValueError, match="expected the form KEY=VALUE"):
            parse_template_kwargs([item])


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    def test_documents_are_padded_to_whole_chunks(self, tmp_path, monkeypatch) -> None:
        """A document must occupy whole chunks or none of it is reusable."""
        workload, _ = _make_workload(tmp_path, monkeypatch)
        for block in workload._document_blocks.values():
            assert workload._token_length(block) % _CHUNK == 0

    def test_documents_honour_a_custom_alignment(self, tmp_path, monkeypatch) -> None:
        """Alignment tracks the deployment's chunk size, not the default."""
        align = 64
        workload, _ = _make_workload(tmp_path, monkeypatch, doc_align_tokens=align)
        for block in workload._document_blocks.values():
            assert workload._token_length(block) % align == 0
        total = workload._chat_prefix_tokens() + workload._token_length(
            workload._system_block
        )
        assert total % align == 0

    def test_system_block_ends_on_a_chunk_boundary(self, tmp_path, monkeypatch) -> None:
        """So every document starts on a boundary."""
        workload, _ = _make_workload(tmp_path, monkeypatch)
        total = workload._chat_prefix_tokens() + workload._token_length(
            workload._system_block
        )
        assert total % _CHUNK == 0

    def test_shared_documents_are_padded_once(self, tmp_path, monkeypatch) -> None:
        """A shared passage must stay byte-identical wherever it appears."""
        workload, _ = _make_workload(tmp_path, monkeypatch)
        # Three distinct passages across two samples, one of them shared.
        assert len(workload._document_blocks) == 3
        assert len(workload._corpus) == 3

    def test_composite_contains_every_document_and_the_question(
        self, tmp_path, monkeypatch
    ) -> None:
        workload, _ = _make_workload(tmp_path, monkeypatch)
        content = workload._build_composite(workload._samples[0])[0]["content"]
        assert content.startswith(workload._system_block)
        for document in workload._samples[0].documents:
            assert workload._document_blocks[document] in content
        assert "who?" in content

    def test_missing_tokenizer_is_fatal(self, tmp_path, monkeypatch) -> None:
        """Without one, documents cannot be aligned."""
        monkeypatch.setattr(rag_qa_quality, "try_load_tokenizer", lambda _n: None)
        with pytest.raises(RuntimeError, match="needs the tokenizer"):
            RagQaQualityWorkload(
                config=RagQaQualityConfig(dataset=_write_dataset(tmp_path, _RECORDS)),
                request_sender=MagicMock(),
                stats_collector=MagicMock(),
                progress_monitor=MagicMock(),
                model_name="fake-model",
            )

    def test_num_samples_truncates_in_dataset_order(
        self, tmp_path, monkeypatch
    ) -> None:
        """Order is stable so two runs measure the same questions."""
        workload, _ = _make_workload(tmp_path, monkeypatch, num_samples=1)
        assert [s.sample_id for s in workload._samples] == ["s0"]


class TestRunFingerprint:
    def test_is_stable_for_identical_configuration(self, tmp_path, monkeypatch) -> None:
        first, _ = _make_workload(tmp_path, monkeypatch)
        second, _ = _make_workload(tmp_path, monkeypatch)
        assert first._run_fingerprint() == second._run_fingerprint()

    @pytest.mark.parametrize(
        "override",
        [
            {"num_samples": 1},
            {"max_output_length": 128},
            {"template_kwargs": {"enable_thinking": False}},
        ],
    )
    def test_changes_when_the_prompts_or_budget_change(
        self, tmp_path, monkeypatch, override
    ) -> None:
        """A diff across runs differing in these would be an input delta."""
        baseline, _ = _make_workload(tmp_path, monkeypatch)
        changed, _ = _make_workload(tmp_path, monkeypatch, **override)
        assert baseline._run_fingerprint() != changed._run_fingerprint()


# ---------------------------------------------------------------------------
# Warmup and dispatch
# ---------------------------------------------------------------------------


class TestWarmup:
    @pytest.mark.asyncio
    async def test_prefills_each_distinct_document_once(
        self, tmp_path, monkeypatch
    ) -> None:
        workload, sender = _make_workload(tmp_path, monkeypatch)
        await workload.warmup()
        assert sender.send_warmup_request.await_count == 3

    @pytest.mark.asyncio
    async def test_prefill_carries_the_system_block(
        self, tmp_path, monkeypatch
    ) -> None:
        """The prefill shares the composite's prefix, so chunk phases match."""
        workload, sender = _make_workload(tmp_path, monkeypatch)
        await workload.warmup()
        content = sender.send_warmup_request.await_args_list[0].args[1][0]["content"]
        assert content.startswith(workload._system_block)
        assert content[len(workload._system_block) :] in workload._corpus


class TestStep:
    @pytest.mark.asyncio
    async def test_scores_a_parsed_answer(self, tmp_path, monkeypatch) -> None:
        workload, _ = _make_workload(
            tmp_path,
            monkeypatch,
            responses=["<final_answer>Paris</final_answer>"],
        )
        assert await workload.step(0.0) == 0.0

        score = workload._aggregator.scores()[0]
        assert score.sample_id == "s0"
        assert score.parsed is True
        assert score.f1 == 1.0
        assert score.answer == "Paris"

    @pytest.mark.asyncio
    async def test_unparsed_answer_is_recorded_but_not_scored(
        self, tmp_path, monkeypatch
    ) -> None:
        workload, _ = _make_workload(
            tmp_path, monkeypatch, responses=["I think it is Paris."]
        )
        await workload.step(0.0)

        score = workload._aggregator.scores()[0]
        assert score.parsed is False
        assert score.f1 == 0.0

    @pytest.mark.asyncio
    async def test_failed_request_is_not_scored(self, tmp_path, monkeypatch) -> None:
        workload, _ = _make_workload(
            tmp_path,
            monkeypatch,
            responses=["<final_answer>Paris</final_answer>"],
            successful=False,
        )
        await workload.step(0.0)
        assert workload._aggregator.scores()[0].parsed is False

    @pytest.mark.asyncio
    async def test_uses_the_configured_output_budget(
        self, tmp_path, monkeypatch
    ) -> None:
        workload, sender = _make_workload(tmp_path, monkeypatch)
        await workload.step(0.0)
        assert sender.send_request.await_args.kwargs["max_tokens"] == 64

    @pytest.mark.asyncio
    async def test_returns_negative_when_samples_are_exhausted(
        self, tmp_path, monkeypatch
    ) -> None:
        workload, _ = _make_workload(tmp_path, monkeypatch)
        assert await workload.step(0.0) == 0.0
        assert await workload.step(0.0) == 0.0
        assert await workload.step(0.0) == -1.0

    @pytest.mark.asyncio
    async def test_measures_each_sample_once(self, tmp_path, monkeypatch) -> None:
        workload, sender = _make_workload(tmp_path, monkeypatch)
        while await workload.step(0.0) >= 0:
            pass
        assert sender.send_request.await_count == 2
        assert [s.sample_id for s in workload._aggregator.scores()] == ["s0", "s1"]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestReporting:
    @pytest.mark.asyncio
    async def test_metric_sections_report_quality_and_parse_rate(
        self, tmp_path, monkeypatch
    ) -> None:
        workload, _ = _make_workload(
            tmp_path,
            monkeypatch,
            responses=["<final_answer>Paris</final_answer>", "no answer here"],
        )
        while await workload.step(0.0) >= 0:
            pass

        sections = {
            s.key: dict((k, v) for k, _, v in s.entries)
            for s in workload.extra_metric_sections()
        }
        assert sections["quality"]["samples"] == 2
        assert sections["quality"]["parsed"] == 1
        assert sections["quality"]["parse_rate"] == 0.5
        assert sections["quality"]["f1_mean"] == 1.0

    @pytest.mark.asyncio
    async def test_quality_is_the_only_extra_section(
        self, tmp_path, monkeypatch
    ) -> None:
        """Quality is measured from the answers alone, with no engine probing."""
        workload, _ = _make_workload(tmp_path, monkeypatch)
        await workload.step(0.0)

        assert [s.key for s in workload.extra_metric_sections()] == ["quality"]

    @pytest.mark.asyncio
    async def test_results_file_pairs_by_sample_id(self, tmp_path, monkeypatch) -> None:
        workload, _ = _make_workload(
            tmp_path,
            monkeypatch,
            responses=["<final_answer>Paris</final_answer>", "no answer tags here"],
        )
        while await workload.step(0.0) >= 0:
            pass
        workload._write_results()

        payload = json.loads((tmp_path / "out.json").read_text())
        by_id = {s["sample_id"]: s for s in payload["per_sample"]}
        assert by_id["s0"]["f1"] == 1.0
        # Null, not 0.0, so a cross-run diff can skip unscored samples.
        assert by_id["s1"]["f1"] is None
        assert by_id["s1"]["parsed"] is False

    @pytest.mark.asyncio
    async def test_results_file_records_the_fingerprint_and_summary(
        self, tmp_path, monkeypatch
    ) -> None:
        workload, _ = _make_workload(tmp_path, monkeypatch)
        while await workload.step(0.0) >= 0:
            pass
        workload._write_results()

        payload = json.loads((tmp_path / "out.json").read_text())
        assert payload["run_fingerprint"] == workload._run_fingerprint()
        assert payload["summary"]["num_samples"] == 2
        assert payload["model"] == "fake-model"

    def test_results_file_directory_is_created(self, tmp_path, monkeypatch) -> None:
        nested = tmp_path / "nested" / "dir" / "out.json"
        workload, _ = _make_workload(tmp_path, monkeypatch, output_path=str(nested))
        workload._write_results()
        assert nested.is_file()
