# SPDX-License-Identifier: Apache-2.0
"""Multi-passage QA dataset loading for quality workloads.

Datasets are not vendored.  Named datasets download from the Hugging Face
Hub on first use and are cached by ``huggingface_hub``; any other value is
treated as a local file path.

``huggingface_hub`` and ``pyarrow`` are imported where they are used rather
than at module scope.  ``lmcache --help`` reaches this module to build its
dataset help text, and the lmcache-cli wheel depends on neither, so importing
either at module scope would make the whole CLI unusable on that install.
"""

# Standard
from dataclasses import dataclass
import json
import os

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass
class Sample:
    """One QA task: retrieved passages, a question, and its gold answers.

    Attributes:
        sample_id: Dataset identifier.  Used to pair samples across runs, so
            it must be stable.
        documents: Passages retrieved for this question, in dataset order.
        question: The question to ask about ``documents``.
        answers: Gold answers; more than one means alternate phrasings.
    """

    sample_id: str
    documents: list[str]
    question: str
    answers: list[str]


@dataclass(frozen=True)
class HubDataset:
    """A Hugging Face Hub file that :func:`load_samples` can read.

    Attributes:
        repo_id: The dataset repository.
        filename: Path of the file within that repository.
        summary: One-line description for CLI help and error messages.
    """

    repo_id: str
    filename: str
    summary: str


# Only schemas verified against the live file belong here: an unreadable
# shape fails at run time, after a dataset has already been downloaded.
_HUB_DATASETS: dict[str, HubDataset] = {
    "musique": HubDataset(
        repo_id="dgslibisey/MuSiQue",
        filename="musique_ans_v1.0_dev.jsonl",
        summary="MuSiQue answerable dev — 20 passages/question, ~2.1k tokens",
    ),
    "hotpotqa": HubDataset(
        repo_id="hotpotqa/hotpot_qa",
        filename="distractor/validation-00000-of-00001.parquet",
        summary="HotpotQA distractor validation — 10 passages/question",
    ),
}

HUB_DATASET_NAMES: tuple[str, ...] = tuple(sorted(_HUB_DATASETS))


def describe_hub_datasets() -> str:
    """Return a one-line-per-dataset description for CLI help text."""
    return "; ".join(
        f"{name}: {_HUB_DATASETS[name].summary}" for name in HUB_DATASET_NAMES
    )


def resolve_dataset_path(dataset: str) -> str:
    """Resolve a dataset name or path to a readable local file.

    Args:
        dataset: A name from :data:`HUB_DATASET_NAMES`, or a path to a local
            ``.json``, ``.jsonl``, or ``.parquet`` file.

    Returns:
        Path to a local file.

    Raises:
        ValueError: If *dataset* is neither a known name nor an existing file.
        RuntimeError: If ``huggingface_hub`` is not installed, or a known name
            could not be downloaded.
    """
    entry = _HUB_DATASETS.get(dataset)
    if entry is None:
        if os.path.isfile(dataset):
            return dataset
        raise ValueError(
            f"Dataset {dataset!r} is not a known dataset name and is not an "
            f"existing file. Known names: {', '.join(HUB_DATASET_NAMES)}."
        )

    logger.info(
        "Resolving dataset %r from the Hugging Face Hub (%s/%s)",
        dataset,
        entry.repo_id,
        entry.filename,
    )
    try:
        # Third Party
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            f"Downloading dataset {dataset!r} requires huggingface_hub. "
            f"Install it with `pip install huggingface_hub`, or pass a local "
            f"file path instead."
        ) from e

    try:
        return hf_hub_download(
            repo_id=entry.repo_id,
            filename=entry.filename,
            repo_type="dataset",
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not download dataset {dataset!r} from {entry.repo_id}: {e}. "
            f"Pass a local file path instead, or set HF_HOME to a cache that "
            f"already holds it."
        ) from e


def _read_records(path: str) -> list[dict[str, object]]:
    """Read a dataset file into a list of raw records.

    Args:
        path: Path to a ``.json``, ``.jsonl``, or ``.parquet`` file.

    Returns:
        The file's records, unmodified.

    Raises:
        ValueError: If a JSON file's top level is not a list.
        RuntimeError: If a Parquet file is given but ``pyarrow`` is missing.
    """
    lowered = path.lower()

    if lowered.endswith((".parquet", ".pq")):
        try:
            # Third Party
            import pyarrow.parquet as pq
        except ImportError as e:
            raise RuntimeError(
                "Reading a Parquet dataset requires pyarrow. Install it with "
                "`pip install pyarrow`, or use a JSON/JSONL dataset."
            ) from e
        return pq.read_table(path).to_pylist()

    if lowered.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of QA records")
    return data


def _passage_text(passage: object) -> str:
    """Render one passage as text, joining its title when it has one."""
    if isinstance(passage, dict):
        title = str(passage.get("title", "")).strip()
        body = str(passage.get("text") or passage.get("paragraph_text") or "").strip()
        return f"{title}\n{body}" if title else body
    return str(passage).strip()


def _documents_from_context(context: object) -> list[str]:
    """Extract passages from a HotpotQA-style ``context`` field.

    Handles both shapes in circulation: the Parquet struct of parallel
    ``title``/``sentences`` lists, and the official JSON's
    ``[title, [sentence, ...]]`` pairs.

    Args:
        context: The record's ``context`` value, in either shape.

    Returns:
        One string per passage; empty when the shape is unrecognized.
    """
    if isinstance(context, dict):
        titles = context.get("title") or []
        sentence_groups = context.get("sentences") or []
        if not isinstance(titles, list) or not isinstance(sentence_groups, list):
            return []
        return [
            _passage_text({"title": title, "text": "".join(str(s) for s in sentences)})
            for title, sentences in zip(titles, sentence_groups, strict=False)
        ]

    if isinstance(context, list):
        documents: list[str] = []
        for entry in context:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            title, sentences = entry
            body = (
                "".join(str(s) for s in sentences)
                if isinstance(sentences, list)
                else str(sentences)
            )
            documents.append(_passage_text({"title": title, "text": body}))
        return documents

    return []


def _extract_documents(record: dict[str, object]) -> list[str]:
    """Extract this record's passages, whichever schema it uses.

    Recognizes MuSiQue's ``paragraphs``, the CacheBlend-style ``ctxs`` /
    ``contexts``, and HotpotQA's ``context``.

    Args:
        record: One raw dataset record.

    Returns:
        One string per passage in dataset order; empty when unrecognized.
    """
    paragraphs = record.get("paragraphs")
    if isinstance(paragraphs, list) and paragraphs:
        return [_passage_text(p) for p in paragraphs]

    passages = record.get("ctxs") or record.get("contexts")
    if isinstance(passages, list) and passages:
        return [_passage_text(p) for p in passages]

    if record.get("context") is not None:
        return _documents_from_context(record["context"])

    return []


def _extract_answers(record: dict[str, object]) -> list[str]:
    """Extract this record's gold answers, including alternate phrasings.

    Args:
        record: One raw dataset record.

    Returns:
        Non-empty answer strings, primary answer first.
    """
    raw = record.get("answers")
    if raw is None:
        raw = record.get("answer")

    answers: list[str] = []
    items = raw if isinstance(raw, list) else [raw]
    for item in items:
        if isinstance(item, list):
            answers.extend(str(x) for x in item)
        elif item is not None:
            answers.append(str(item))

    aliases = record.get("answer_aliases")
    if isinstance(aliases, list):
        answers.extend(str(a) for a in aliases)

    return [a for a in (answer.strip() for answer in answers) if a]


def load_samples(path: str) -> list[Sample]:
    """Load a QA dataset file into :class:`Sample` objects.

    Records missing passages, a question, or gold answers are skipped: each
    is unusable for scoring.

    Args:
        path: Local file path, as returned by :func:`resolve_dataset_path`.

    Returns:
        The usable samples, in dataset order.

    Raises:
        ValueError: If the file yields no usable samples, or a JSON file's
            top level is not a list.
        RuntimeError: If a Parquet file is given but ``pyarrow`` is missing.
    """
    records = _read_records(path)

    samples: list[Sample] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        documents = [doc for doc in _extract_documents(record) if doc]
        question = str(record.get("question") or record.get("query") or "").strip()
        answers = _extract_answers(record)
        if not documents or not question or not answers:
            continue
        raw_id = record.get("_id") or record.get("id") or index
        samples.append(
            Sample(
                sample_id=str(raw_id),
                documents=documents,
                question=question,
                answers=answers,
            )
        )

    if not samples:
        raise ValueError(
            f"{path}: no usable QA samples found. Records need passages "
            f"(paragraphs/ctxs/context), a question, and gold answers."
        )

    logger.info("Loaded %d QA samples from %s", len(samples), path)
    return samples
