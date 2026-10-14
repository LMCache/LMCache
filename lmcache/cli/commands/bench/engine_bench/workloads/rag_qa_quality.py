# SPDX-License-Identifier: Apache-2.0
"""RAG-style question-answering quality workload for ``lmcache bench engine``.

Measures whether reusing cached KV changes the model's *answers*, not just
how fast it produces them.  Documents are prefilled individually, then
composed into one request::

    [system prompt][doc_a][doc_b]…[doc_n][question]

so each is reused at a position it was never cached at — the RAG serving
pattern.  Reports one arm; diff two runs by sample id to compare stacks.

See ``docs/source/cli/bench.rst`` for usage and output.
"""

# Standard
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os

# First Party
from lmcache.cli.commands.bench.engine_bench.config import WarmupPolicy
from lmcache.cli.commands.bench.engine_bench.progress import ProgressMonitor
from lmcache.cli.commands.bench.engine_bench.quality.alignment import ChunkAligner
from lmcache.cli.commands.bench.engine_bench.quality.dataset import (
    Sample,
    load_samples,
    resolve_dataset_path,
)
from lmcache.cli.commands.bench.engine_bench.quality.prompts import (
    QA_QUESTION_TEMPLATE,
    QA_SYSTEM_PROMPT,
    compose_qa_messages,
    compose_store_messages,
)
from lmcache.cli.commands.bench.engine_bench.quality.scoring import (
    QualityAggregator,
    SampleScore,
    best_f1,
    extract_final_answer,
)
from lmcache.cli.commands.bench.engine_bench.request_sender import RequestSender
from lmcache.cli.commands.bench.engine_bench.stats import StatsCollector
from lmcache.cli.commands.bench.engine_bench.tokenizers import try_load_tokenizer
from lmcache.cli.commands.bench.engine_bench.workloads.base import (
    BaseWorkload,
    MetricSection,
)
from lmcache.logging import init_logger

logger = init_logger(__name__)

# LMCache's own default chunk size (``lmcache/v1/config.py``).  Configurable
# rather than detected: the baseline stack has no LMCache server to ask, and
# two runs that padded differently would no longer share prompts.
DEFAULT_DOC_ALIGN_TOKENS = 256

# Warmup and measured completions arrive through the same callback, so the
# two are told apart by request-id prefix.
_PREFILL_REQUEST_PREFIX = "prefill_doc"
_MEASURED_REQUEST_PREFIX = "sample"


@dataclass
class RagQaQualityConfig:
    """Workload-specific config for the rag-qa-quality workload.

    Attributes:
        dataset: A known dataset name or a path to a local QA file.
        num_samples: Samples to measure, taken in dataset order so two runs
            measure the same questions.
        max_output_length: Token budget per answer.  Must fit a reasoning
            model's thinking block as well as the answer tags.
        doc_align_tokens: Documents and the system block are padded to a
            multiple of this, so each document occupies whole cache chunks at
            the same phase in the prefill and in the composite request.  Set
            it to the deployment's LMCache chunk size; a mismatch leaves
            documents off-phase and reuse partial.
        template_kwargs: Chat-template variables, already coerced to
            bool/int/str.  Empty means the model's own template default
            applies.  The workload does not send these itself — the caller
            puts them on the shared ``RequestSender`` as
            ``chat_template_kwargs``; here they only enter the run
            fingerprint and the results file, so runs configured differently
            are not mistaken for comparable.
        output_path: Where the per-sample results JSON is written.
    """

    dataset: str
    num_samples: int = 50
    max_output_length: int = 1024
    doc_align_tokens: int = DEFAULT_DOC_ALIGN_TOKENS
    template_kwargs: dict[str, bool | int | str] = field(default_factory=dict)
    output_path: str = "rag_qa_quality.json"

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("dataset must be non-empty")
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {self.num_samples}")
        if self.max_output_length < 1:
            raise ValueError(
                f"max_output_length must be >= 1, got {self.max_output_length}"
            )
        if self.doc_align_tokens < 1:
            raise ValueError(
                f"doc_align_tokens must be >= 1, got {self.doc_align_tokens}"
            )
        if not self.output_path:
            raise ValueError("output_path must be non-empty")

    @classmethod
    def resolve(
        cls,
        dataset: str,
        num_samples: int,
        max_output_length: int,
        doc_align_tokens: int,
        template_kwargs: Mapping[str, bool | int | str],
        output_path: str,
    ) -> "RagQaQualityConfig":
        """Build a validated config from CLI arguments.

        Args:
            dataset: Dataset name or local path.
            num_samples: Number of samples to measure.
            max_output_length: Token budget per answer.
            doc_align_tokens: Token alignment for documents and the system
                block; should match the deployment's LMCache chunk size.
            template_kwargs: Coerced chat-template variables.  Copied, so a
                later mutation of the caller's mapping does not change the
                config.
            output_path: Destination for the per-sample results JSON.

        Returns:
            The validated config.
        """
        return cls(
            dataset=dataset,
            num_samples=num_samples,
            max_output_length=max_output_length,
            doc_align_tokens=doc_align_tokens,
            template_kwargs=dict(template_kwargs),
            output_path=output_path,
        )


def parse_template_kwargs(items: list[str]) -> dict[str, bool | int | str]:
    """Parse repeatable ``KEY=VALUE`` chat-template arguments.

    Values are typed: a template testing ``enable_thinking`` sees the string
    ``"false"`` as truthy and keeps thinking on.

    Args:
        items: Raw ``KEY=VALUE`` strings from the CLI.

    Returns:
        The parsed mapping.

    Raises:
        ValueError: If an item has no ``=`` or an empty key.
    """
    parsed: dict[str, bool | int | str] = {}
    for item in items:
        key, separator, raw = item.partition("=")
        key = key.strip()
        raw = raw.strip()
        if not separator or not key:
            raise ValueError(
                f"Invalid template kwarg {item!r}; expected the form KEY=VALUE"
            )
        if raw.lower() in ("true", "false"):
            parsed[key] = raw.lower() == "true"
        elif raw.lstrip("-").isdigit():
            parsed[key] = int(raw)
        else:
            parsed[key] = raw
    return parsed


class RagQaQualityWorkload(BaseWorkload):
    """Prefills documents individually, then scores answers about them."""

    def __init__(
        self,
        config: RagQaQualityConfig,
        request_sender: RequestSender,
        stats_collector: StatsCollector,
        progress_monitor: ProgressMonitor,
        model_name: str,
        seed: int = 42,
    ) -> None:
        """Initialize the workload and build every prompt up front.

        Args:
            config: Validated workload config.
            request_sender: Shared request sender.
            stats_collector: Shared stats collector.
            progress_monitor: Shared progress monitor.
            model_name: Model whose tokenizer sizes the padding.
            seed: Random seed for padding selection.

        Raises:
            ValueError: If the dataset yields no usable samples.
            RuntimeError: If the model's tokenizer cannot be loaded.
        """
        super().__init__(request_sender, stats_collector, progress_monitor)
        self._config = config
        self._model_name = model_name
        self._seed = seed
        self._aggregator = QualityAggregator()
        self._responses: dict[str, str] = {}
        self._index = 0

        self._tokenizer = try_load_tokenizer(model_name)
        if self._tokenizer is None:
            raise RuntimeError(
                f"rag-qa-quality needs the tokenizer for {model_name!r} to align "
                f"documents to cache chunks. Install transformers and pass "
                f"--model with a HuggingFace repo ID or a local path."
            )
        self._aligner = ChunkAligner(
            self._tokenizer, model_name, config.doc_align_tokens, seed
        )

        dataset_path = resolve_dataset_path(config.dataset)
        samples = load_samples(dataset_path)
        self._samples = samples[: config.num_samples]

        self._system_block = self._build_system_block()
        self._document_blocks = self._build_document_blocks()
        self._corpus = self._build_corpus()

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_block(self) -> str:
        """Build the shared system prompt, padded to end on a chunk boundary."""
        return self._aligner.system_block(QA_SYSTEM_PROMPT)

    def _build_document_blocks(self) -> dict[str, str]:
        """Pad every distinct document to a whole number of chunks.

        Returns:
            Raw document text -> padded form.  Keyed by the raw text so a
            shared document is padded once and stays byte-identical.
        """
        blocks: dict[str, str] = {}
        for sample in self._samples:
            for document in sample.documents:
                if document not in blocks:
                    blocks[document] = self._aligner.passage_block(document)
        logger.info("Padded %d distinct documents", len(blocks))
        return blocks

    def _build_corpus(self) -> list[str]:
        """Return each distinct padded document, in first-appearance order."""
        corpus: list[str] = []
        seen: set[str] = set()
        for sample in self._samples:
            for document in sample.documents:
                block = self._document_blocks[document]
                if block not in seen:
                    seen.add(block)
                    corpus.append(block)
        return corpus

    def _build_composite(self, sample: Sample) -> list[dict[str, str]]:
        """Build the measured request for *sample*."""
        return compose_qa_messages(
            self._system_block,
            [self._document_blocks[d] for d in sample.documents],
            sample.question,
        )

    def _run_fingerprint(self) -> str:
        """Return a digest of everything that determines the prompts.

        Runs are comparable only when this matches; otherwise a diff would
        report an input delta as a quality delta.
        """
        material = json.dumps(
            {
                "dataset": self._config.dataset,
                "num_samples": self._config.num_samples,
                "max_output_length": self._config.max_output_length,
                "template_kwargs": self._config.template_kwargs,
                "doc_align_tokens": self._config.doc_align_tokens,
                "model": self._model_name,
                "seed": self._seed,
                "sample_ids": [s.sample_id for s in self._samples],
                "system_prompt": QA_SYSTEM_PROMPT,
                "question_template": QA_QUESTION_TEMPLATE,
            },
            sort_keys=True,
        )
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Benchmark lifecycle
    # ------------------------------------------------------------------

    def log_config(self) -> None:
        """Log key workload config before the benchmark starts."""
        c = self._config
        bold, cyan, yellow, reset = "\033[1m", "\033[96m", "\033[93m", "\033[0m"
        print(
            f"{bold}{'═' * 50}{reset}\n"
            f"{bold} Workload: {cyan}rag-qa-quality{reset}\n"
            f"{bold}{'─' * 50}{reset}\n"
            f"  Dataset:          {yellow}{c.dataset}{reset}\n"
            f"  Samples:          {yellow}{len(self._samples)}{reset}\n"
            f"  Distinct docs:    {yellow}{len(self._corpus)}{reset}\n"
            f"  Max output:       {yellow}{c.max_output_length}{reset} tokens\n"
            f"  Doc alignment:    {yellow}{c.doc_align_tokens}{reset} tokens\n"
            f"  Template kwargs:  {yellow}{c.template_kwargs or 'model default'}"
            f"{reset}\n"
            f"  Run fingerprint:  {yellow}{self._run_fingerprint()}{reset}\n"
            f"  Results:          {yellow}{c.output_path}{reset}\n"
            f"{bold}{'═' * 50}{reset}"
        )

    async def warmup(self) -> None:
        """Prefill every distinct document on its own to store its KV."""
        total = len(self._corpus)
        for index, block in enumerate(self._corpus):
            request_id = f"{_PREFILL_REQUEST_PREFIX}{index}"
            messages = compose_store_messages(self._system_block, [block])
            self._progress_monitor.log_message(f"Prefill {index + 1}/{total}")
            self._progress_monitor.on_request_sent(request_id)
            result = await self._request_sender.send_warmup_request(
                request_id, messages
            )
            if not result.successful:
                self._progress_monitor.log_message(
                    f"Prefill {request_id} failed: {result.error}"
                )
        self._progress_monitor.log_message(f"Prefilled {total} documents")

    async def step(self, time_offset: float) -> float:
        """Measure one sample: send its composite request and score it."""
        if self._index >= len(self._samples):
            return -1.0

        sample = self._samples[self._index]
        request_id = f"{_MEASURED_REQUEST_PREFIX}{self._index}"
        self._index += 1

        self._progress_monitor.on_request_sent(request_id)
        result = await self._request_sender.send_request(
            request_id,
            self._build_composite(sample),
            max_tokens=self._config.max_output_length,
        )
        # The sender fires its callbacks before returning, so the response
        # text is available to score in this same step.
        self._drain_finished_queue()

        response = self._responses.pop(request_id, "")
        answer = extract_final_answer(response)
        parsed = bool(answer) and result.successful
        self._aggregator.record(
            SampleScore(
                sample_id=sample.sample_id,
                parsed=parsed,
                f1=best_f1(answer, sample.answers) if parsed else 0.0,
                answer=answer,
                ttft=result.ttft,
                num_output_tokens=result.num_output_tokens,
            )
        )
        if not parsed:
            self._progress_monitor.log_message(
                f"{request_id}: no <final_answer> in response (not scored)"
            )
        return 0.0

    def on_request_finished(self, request_id: str, output: str) -> None:
        """Hold a measured request's text until its step scores it.

        Warmup completions arrive here too and are dropped.
        """
        if request_id.startswith(_MEASURED_REQUEST_PREFIX):
            self._responses[request_id] = output

    def run(self, warmup_policy: WarmupPolicy = WarmupPolicy.RUN) -> None:
        """Run the benchmark, then write the per-sample results file.

        Args:
            warmup_policy: ``WarmupPolicy.SKIP`` starts the measured run
                immediately, without prefilling the documents.
        """
        super().run(warmup_policy)
        self._write_results()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def extra_metric_sections(self) -> list[MetricSection]:
        """Return the answer-quality section for the summary."""
        summary = self._aggregator.summarize()
        return [
            MetricSection(
                key="quality",
                label="Answer Quality",
                entries=[
                    ("samples", "Samples measured", summary.num_samples),
                    ("parsed", "Samples scored", summary.num_parsed),
                    ("parse_rate", "Parse rate", round(summary.parse_rate, 4)),
                    ("f1_mean", "Mean F1 (scored only)", round(summary.f1_mean, 4)),
                    ("fingerprint", "Run fingerprint", self._run_fingerprint()),
                ],
            )
        ]

    def _write_results(self) -> None:
        """Write per-sample results and the summary to ``output_path``.

        An unparsed sample's ``f1`` is null, not zero, so a cross-run diff can
        pair by sample id and skip what either run failed to score.
        """
        summary = self._aggregator.summarize()
        payload = {
            "run_fingerprint": self._run_fingerprint(),
            "config": asdict(self._config),
            "model": self._model_name,
            "summary": asdict(summary),
            "per_sample": [
                {
                    "sample_id": score.sample_id,
                    "f1": round(score.f1, 4) if score.parsed else None,
                    "parsed": score.parsed,
                    "answer": score.answer,
                    "ttft": round(score.ttft, 4),
                    "num_output_tokens": score.num_output_tokens,
                }
                for score in self._aggregator.scores()
            ],
        }
        directory = os.path.dirname(self._config.output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self._config.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        logger.info("Wrote quality results to %s", self._config.output_path)
        print(f"Quality results written to {self._config.output_path}")


__all__ = [
    "DEFAULT_DOC_ALIGN_TOKENS",
    "RagQaQualityConfig",
    "RagQaQualityWorkload",
    "parse_template_kwargs",
]
