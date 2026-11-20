# SPDX-License-Identifier: Apache-2.0
"""Tests for QA dataset loading and dataset-name resolution."""

# Standard
import json

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.quality.dataset import (
    HUB_DATASET_NAMES,
    describe_hub_datasets,
    load_samples,
    resolve_dataset_path,
)


def _write_json(tmp_path, name: str, records: list) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(records))
    return str(path)


def _write_jsonl(tmp_path, name: str, records: list) -> str:
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return str(path)


class TestLoadSamplesMusique:
    """MuSiQue ships ``paragraphs`` with ``paragraph_text`` bodies."""

    def _record(self, **overrides) -> dict:
        record = {
            "id": "m1",
            "paragraphs": [
                {"idx": 0, "title": "First", "paragraph_text": "body one"},
                {"idx": 1, "title": "Second", "paragraph_text": "body two"},
            ],
            "question": "who?",
            "answer": "Someone",
            "answer_aliases": ["Some One"],
        }
        record.update(overrides)
        return record

    def test_loads_documents_in_order(self, tmp_path) -> None:
        path = _write_jsonl(tmp_path, "m.jsonl", [self._record()])
        sample = load_samples(path)[0]
        assert sample.sample_id == "m1"
        assert sample.documents == ["First\nbody one", "Second\nbody two"]
        assert sample.question == "who?"

    def test_merges_answer_aliases_after_the_primary(self, tmp_path) -> None:
        path = _write_jsonl(tmp_path, "m.jsonl", [self._record()])
        assert load_samples(path)[0].answers == ["Someone", "Some One"]

    def test_jsonl_blank_lines_are_skipped(self, tmp_path) -> None:
        path = tmp_path / "m.jsonl"
        path.write_text(json.dumps(self._record()) + "\n\n")
        assert len(load_samples(str(path))) == 1


class TestLoadSamplesHotpot:
    def test_hugging_face_struct_shape(self, tmp_path) -> None:
        """Parquet stores parallel title/sentences lists."""
        record = {
            "_id": "h1",
            "question": "who?",
            "answer": "Someone",
            "context": {
                "title": ["First", "Second"],
                "sentences": [["a ", "b"], ["c"]],
            },
        }
        sample = load_samples(_write_json(tmp_path, "h.json", [record]))[0]
        assert sample.documents == ["First\na b", "Second\nc"]
        assert sample.answers == ["Someone"]

    def test_official_json_pair_shape(self, tmp_path) -> None:
        """The official release stores ``[title, [sentence, ...]]`` pairs."""
        record = {
            "_id": "h2",
            "question": "who?",
            "answers": ["Someone"],
            "context": [["First", ["a ", "b"]], ["Second", ["c"]]],
        }
        sample = load_samples(_write_json(tmp_path, "h.json", [record]))[0]
        assert sample.documents == ["First\na b", "Second\nc"]

    def test_malformed_context_entries_are_skipped(self, tmp_path) -> None:
        record = {
            "question": "who?",
            "answers": ["Someone"],
            "context": [["First", ["a"]], ["too", "many", "parts"]],
        }
        sample = load_samples(_write_json(tmp_path, "h.json", [record]))[0]
        assert sample.documents == ["First\na"]


class TestLoadSamplesCacheBlendStyle:
    def test_ctxs_with_title_and_text(self, tmp_path) -> None:
        record = {
            "ctxs": [{"title": "T", "text": "body"}],
            "question": "who?",
            "answers": ["Someone"],
        }
        assert load_samples(_write_json(tmp_path, "c.json", [record]))[0].documents == (
            ["T\nbody"]
        )

    def test_untitled_passage_keeps_only_its_body(self, tmp_path) -> None:
        record = {
            "ctxs": [{"title": "", "text": "body"}],
            "question": "who?",
            "answers": ["Someone"],
        }
        assert load_samples(_write_json(tmp_path, "c.json", [record]))[0].documents == (
            ["body"]
        )

    def test_falls_back_to_the_record_index_for_an_id(self, tmp_path) -> None:
        record = {"ctxs": [{"text": "body"}], "question": "who?", "answers": ["A"]}
        assert load_samples(_write_json(tmp_path, "c.json", [record]))[0].sample_id == (
            "0"
        )


class TestLoadSamplesRagasHandoff:
    """NVIDIA's ``inputs_<dataset>.jsonl``: RAGAS field names, a ``_meta`` row."""

    _META = {"id": "_meta", "dataset": "hotpotqa", "system_prompt": "..."}

    def _record(self, **overrides) -> dict:
        record = {
            "id": "h1",
            "user_input": "Where is it based?",
            "reference": "Bassendean",
            "retrieved_contexts": ["Passage one.", "Passage two.", "Passage three."],
            "context_metadata": [
                {"source_id": "A", "gold": False},
                {"source_id": "B", "gold": True},
                {"source_id": "C", "gold": False},
            ],
        }
        record.update(overrides)
        return record

    def test_reads_question_reference_and_contexts(self, tmp_path) -> None:
        path = _write_jsonl(tmp_path, "inputs.jsonl", [self._META, self._record()])
        [sample] = load_samples(path)
        assert sample.sample_id == "h1"
        assert sample.question == "Where is it based?"
        assert sample.answers == ["Bassendean"]
        assert sample.documents == ["Passage one.", "Passage two.", "Passage three."]

    def test_meta_row_is_skipped(self, tmp_path) -> None:
        path = _write_jsonl(tmp_path, "inputs.jsonl", [self._META, self._record()])
        assert [s.sample_id for s in load_samples(path)] == ["h1"]

    def test_gold_flags_mark_evidence(self, tmp_path) -> None:
        path = _write_jsonl(tmp_path, "inputs.jsonl", [self._record()])
        [sample] = load_samples(path)
        assert sample.evidence_indices == [1]

    def test_evidence_follows_its_passage_past_dropped_ones(self, tmp_path) -> None:
        """An empty passage is dropped; evidence must still point at its text."""
        record = self._record(retrieved_contexts=["", "Passage two.", "Passage three."])
        path = _write_jsonl(tmp_path, "inputs.jsonl", [record])
        [sample] = load_samples(path)
        assert sample.documents[sample.evidence_indices[0]] == "Passage two."

    def test_without_metadata_no_evidence_is_marked(self, tmp_path) -> None:
        record = self._record()
        del record["context_metadata"]
        path = _write_jsonl(tmp_path, "inputs.jsonl", [record])
        assert load_samples(path)[0].evidence_indices == []

    def test_longbench_aliases_follow_the_reference(self, tmp_path) -> None:
        """LongBench hand-offs keep alternate answers in ``extra``."""
        record = self._record(
            reference="Nightwing",
            extra={"answer_aliases": ["Nightwing", "Dick Grayson"]},
        )
        path = _write_jsonl(tmp_path, "inputs.jsonl", [record])
        assert load_samples(path)[0].answers == ["Nightwing", "Dick Grayson"]


class TestLoadSamplesEvidence:
    def test_musique_supporting_paragraphs(self, tmp_path) -> None:
        record = {
            "id": "m1",
            "paragraphs": [
                {"title": "A", "paragraph_text": "a", "is_supporting": False},
                {"title": "B", "paragraph_text": "b", "is_supporting": True},
            ],
            "question": "q?",
            "answer": "x",
        }
        path = _write_jsonl(tmp_path, "m.jsonl", [record])
        assert load_samples(path)[0].evidence_indices == [1]

    def test_hotpot_struct_supporting_facts(self, tmp_path) -> None:
        record = {
            "id": "h1",
            "context": {
                "title": ["A", "B", "C"],
                "sentences": [["a."], ["b."], ["c."]],
            },
            "supporting_facts": {"title": ["C", "A"], "sent_id": [0, 0]},
            "question": "q?",
            "answer": "x",
        }
        path = _write_json(tmp_path, "h.json", [record])
        assert load_samples(path)[0].evidence_indices == [0, 2]

    def test_hotpot_pair_supporting_facts(self, tmp_path) -> None:
        record = {
            "_id": "h1",
            "context": [["A", ["a."]], ["B", ["b."]]],
            "supporting_facts": [["B", 0]],
            "question": "q?",
            "answer": "x",
        }
        path = _write_json(tmp_path, "h.json", [record])
        assert load_samples(path)[0].evidence_indices == [1]


class TestLoadSamplesRejection:
    """Unusable records are skipped rather than silently scored."""

    @pytest.mark.parametrize(
        "record",
        [
            {"question": "who?", "answers": ["A"]},  # no passages
            {"ctxs": [{"text": "b"}], "answers": ["A"]},  # no question
            {"ctxs": [{"text": "b"}], "question": "who?"},  # no gold answers
            {"ctxs": [{"text": "b"}], "question": "who?", "answers": []},
        ],
    )
    def test_unusable_records_yield_no_samples(self, tmp_path, record) -> None:
        path = _write_json(tmp_path, "bad.json", [record])
        with pytest.raises(ValueError, match="no usable QA samples"):
            load_samples(path)

    def test_usable_records_survive_alongside_unusable_ones(self, tmp_path) -> None:
        records = [
            {"question": "who?", "answers": ["A"]},
            {"ctxs": [{"text": "b"}], "question": "who?", "answers": ["A"]},
        ]
        assert len(load_samples(_write_json(tmp_path, "mixed.json", records))) == 1

    def test_top_level_must_be_a_list(self, tmp_path) -> None:
        path = tmp_path / "obj.json"
        path.write_text(json.dumps({"question": "who?"}))
        with pytest.raises(ValueError, match="expected a JSON list"):
            load_samples(str(path))


class TestResolveDatasetPath:
    def test_existing_file_passes_through(self, tmp_path) -> None:
        path = _write_json(tmp_path, "local.json", [])
        assert resolve_dataset_path(path) == path

    def test_unknown_name_names_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="not a known dataset name"):
            resolve_dataset_path("nope")

        with pytest.raises(ValueError, match="musique"):
            resolve_dataset_path("nope")

    def test_known_names_are_sorted_and_non_empty(self) -> None:
        assert HUB_DATASET_NAMES == tuple(sorted(HUB_DATASET_NAMES))
        assert "musique" in HUB_DATASET_NAMES

    def test_every_known_name_is_described(self) -> None:
        description = describe_hub_datasets()
        for name in HUB_DATASET_NAMES:
            assert name in description
