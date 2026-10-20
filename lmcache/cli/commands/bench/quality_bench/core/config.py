# SPDX-License-Identifier: Apache-2.0
"""Configuration for ``lmcache bench quality``."""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass, field
import enum


class ArmRole(enum.Enum):
    """The serving stack an arm runs, and how its answers are used.

    Every arm receives byte-identical prompts. Only the stack behind the URL
    differs, so a difference in answers arises from the stack.
    """

    REFERENCE = "reference"
    """The stack every other arm is judged against, e.g. the original model
    on plain vLLM with prefix caching off."""

    CONTROL = "control"
    """Optional: a second stack expected to answer like the reference, e.g.
    the reference with vLLM automatic prefix caching on (same KV as
    recompute). Its gap to the reference is a noise floor; it does not set
    the verdict."""

    CANDIDATE = "candidate"
    """The stack under test, e.g. a quantized, compressed or post-trained
    model, a KV cache optimization, or another engine."""


class ReuseScenario(enum.Enum):
    """How the passages reach the cache before each question is asked."""

    INDEPENDENT = "independent"
    """Every distinct passage is stored on its own, then questions combine
    them. This is the "documents-cached-separately" product scenario."""

    REORDERED = "reordered"
    """Each question's passages are stored together in dataset order, then
    the question presents them in a seeded new order. This is the "re-ordered
    documents" scenario."""


class TaskType(enum.Enum):
    """A dataset's task type, which determines how it's scored.

    Only short-answer QA is supported. The task type is declared rather than
    inferred so that a new type is added deliberately, with its own prompt
    and scorer, instead of being scored with QA assumptions.
    """

    QA = "qa"
    """A question with short gold answers (names, figures, short phrases)."""


class VerdictMetric(enum.Enum):
    """Which per-question score the verdict compares."""

    MATCH = "match"
    """1 if the answer states a gold answer (see ``answer_match``), else 0."""

    F1 = "f1"
    """Best SQuAD token F1 against the gold answers."""

    JUDGE = "judge"
    """An LLM judge's answer-accuracy rating (RAGAS ``AnswerAccuracy``):
    0 to 1 in steps of 0.25. The standard reference metric."""


@dataclass(frozen=True)
class ArmEndpoint:
    """One arm -- its role and the OpenAI-compatible URL serving it.

    Attributes:
        role: What the stack behind ``url`` is.
        url: Base URL of the engine, e.g. ``http://host:8000``.
    """

    role: ArmRole
    url: str


@dataclass(frozen=True)
class JudgeSettings:
    """The LLM judge used by ``VerdictMetric.JUDGE``.

    Attributes:
        url: Base URL of an OpenAI-compatible endpoint serving the judge.
            Should not be one of the arms under test.
        model: Model name to request from the judge endpoint.
        api_key: API key sent to the judge endpoint.
        max_tokens: Completion budget per judge call. Reasoning judges need
            room to think before rating.
        concurrency: Judge calls in flight at once.
    """

    url: str
    model: str
    api_key: str = "EMPTY"
    max_tokens: int = 4096
    concurrency: int = 8


@dataclass
class QualityRunConfig:
    """Validated settings for one ``bench quality run``.

    A run sends the plan to one or more arms and saves their answers; the
    verdict is reached later by ``bench quality analyze``. Several runs into
    the same results directory -- one per arm, e.g. on different GPU
    servers -- must share every setting in :meth:`plan_settings`.

    Attributes:
        model: Served model name. Also loads the tokenizer used to align
            passages to cache chunks.
        dataset: Known dataset name or local QA file path.
        arms: The arms to run: at least one, and at most one per role.
        scenario: How passages are cached before each question.
        num_samples: Questions to measure, taken in dataset order -- ``0``
            measures every usable question in the file.
        chunk_size: LMCache chunk size the passages are aligned to. Must
            match a caching candidate's chunk size (e.g. the LMCache
            blend server's ``--chunk-size``).
        reference_repeats: Reference passes, when a reference arm is run.
            Two or more give the per-question flip rate between identical
            runs, i.e. a noise floor.
        max_output_tokens: Token budget per answer. Must fit a reasoning
            model's thinking as well as the answer tags.
        template_kwargs: Chat-template variables sent as
            ``chat_template_kwargs`` (e.g. ``reasoning_effort``).
        task: The dataset's task type.
        blend_metrics_url: For a candidate served through the LMCache blend
            server, its Prometheus ``/metrics`` URL (``--capture-blend-metrics``), read
            during the candidate pass. Empty means blend engagement is not
            checked and the verdict says so. Only valid with a candidate arm.
        output_dir: Results directory, used as given. Empty means a new
            timestamped directory under ``quality_results``.
        seed: Seed for passage padding and the re-ordering permutations.
    """

    model: str
    dataset: str
    arms: list[ArmEndpoint]
    scenario: ReuseScenario = ReuseScenario.INDEPENDENT
    num_samples: int = 0
    chunk_size: int = 256
    reference_repeats: int = 2
    max_output_tokens: int = 2048
    template_kwargs: dict[str, bool | int | str] = field(default_factory=dict)
    task: TaskType = TaskType.QA
    blend_metrics_url: str = ""
    output_dir: str = ""
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")
        if not self.dataset:
            raise ValueError("dataset must be non-empty")
        if not self.arms:
            raise ValueError(
                "pass at least one of --reference-url, --candidate-url, --control-url"
            )
        roles = [arm.role for arm in self.arms]
        for role in ArmRole:
            if roles.count(role) > 1:
                raise ValueError(f"at most one {role.value} arm is allowed")
        if any(not arm.url for arm in self.arms):
            raise ValueError("every arm needs a non-empty url")
        if self.blend_metrics_url and ArmRole.CANDIDATE not in roles:
            raise ValueError("--capture-blend-metrics needs --candidate-url")
        if self.num_samples < 0:
            raise ValueError(f"num_samples must be >= 0, got {self.num_samples}")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.reference_repeats < 1:
            raise ValueError(
                f"reference_repeats must be >= 1, got {self.reference_repeats}"
            )
        if self.max_output_tokens < 1:
            raise ValueError(
                f"max_output_tokens must be >= 1, got {self.max_output_tokens}"
            )

    def arm(self, role: ArmRole) -> ArmEndpoint:
        """Return the arm with *role*.

        Args:
            role: The role to look up.

        Returns:
            The configured arm.

        Raises:
            KeyError: If no arm has *role*.
        """
        for endpoint in self.arms:
            if endpoint.role is role:
                return endpoint
        raise KeyError(role.value)

    def has_arm(self, role: ArmRole) -> bool:
        """Return whether an arm with *role* is configured."""
        return any(endpoint.role is role for endpoint in self.arms)

    def plan_settings(self) -> Mapping[str, object]:
        """Return the settings that determine the prompts every arm receives.

        Runs into one results directory must agree on all of them, and they
        are recorded in the directory's ``plan.json``.

        Returns:
            A JSON-serializable mapping.
        """
        return {
            "model": self.model,
            "dataset": self.dataset,
            "task": self.task.value,
            "scenario": self.scenario.value,
            "num_samples": self.num_samples,
            "chunk_size": self.chunk_size,
            "max_output_tokens": self.max_output_tokens,
            "template_kwargs": dict(self.template_kwargs),
            "seed": self.seed,
        }


@dataclass
class QualityAnalyzeConfig:
    """Validated settings for one ``bench quality analyze``.

    Attributes:
        results_dir: Directory one or more ``bench quality run`` invocations
            wrote into.
        verdict_metric: Score the verdict compares.
        judge: The LLM judge. Required when ``verdict_metric`` is
            ``JUDGE``, otherwise unused.
        tolerance: Largest acceptable mean score difference between the
            candidate arm and the reference, in either direction, in score
            units (``0.05`` = 5 points). A gain counts too -- a change meant
            to preserve answers should not improve them, so a large one
            points at the setup.
        min_blend_share: When blend metrics were captured, the smallest
            share of looked-up prompt tokens the blend server must serve by
            blending, overall and in every window of questions, for a
            verdict to be valid. Unused otherwise.
    """

    results_dir: str
    verdict_metric: VerdictMetric = VerdictMetric.MATCH
    judge: JudgeSettings | None = None
    tolerance: float = 0.05
    min_blend_share: float = 0.3

    def __post_init__(self) -> None:
        if not self.results_dir:
            raise ValueError("results_dir must be non-empty")
        if not 0.0 <= self.tolerance <= 1.0:
            raise ValueError(f"tolerance must be in [0, 1], got {self.tolerance}")
        if not 0.0 <= self.min_blend_share <= 1.0:
            raise ValueError(
                f"min_blend_share must be in [0, 1], got {self.min_blend_share}"
            )
        if self.verdict_metric is VerdictMetric.JUDGE:
            if self.judge is None:
                raise ValueError("verdict_metric judge needs judge settings")
            if not self.judge.url or not self.judge.model:
                raise ValueError("the judge needs a url and a model")
            if self.judge.max_tokens < 1:
                raise ValueError(
                    f"judge max_tokens must be >= 1, got {self.judge.max_tokens}"
                )
            if self.judge.concurrency < 1:
                raise ValueError(
                    f"judge concurrency must be >= 1, got {self.judge.concurrency}"
                )

    def scoring_settings(self) -> Mapping[str, object]:
        """Return the settings that determine the scores.

        Together with the plan's settings, these decide whether two
        summaries are comparable.

        Returns:
            A JSON-serializable mapping.
        """
        settings: dict[str, object] = {"verdict_metric": self.verdict_metric.value}
        if self.verdict_metric is VerdictMetric.JUDGE and self.judge is not None:
            # The judge's model decides its scores -- its URL does not.
            settings["judge_model"] = self.judge.model
        return settings
