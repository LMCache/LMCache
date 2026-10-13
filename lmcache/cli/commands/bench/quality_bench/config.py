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
    """Full recompute: plain vLLM with prefix caching off. These are the
    answers every other arm is judged against."""

    PREFIX_CACHE = "apc"
    """vLLM automatic prefix caching, no CacheBlend. Computes the same KV as
    recompute -- its gap to the reference is a noise floor."""

    BLEND = "blend"
    """vLLM + CacheBlend (and prefix caching). The arm under test."""


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
class QualityBenchConfig:
    """Validated settings for one ``bench quality`` run.

    Attributes:
        model: Served model name. Also loads the tokenizer used to align
            passages to cache chunks.
        dataset: Known dataset name or local QA file path.
        arms: The arms to run. Must hold exactly one ``REFERENCE`` and one
            ``BLEND`` arm, and at most one ``PREFIX_CACHE`` arm.
        scenario: How passages are cached before each question.
        num_samples: Questions to measure, taken in dataset order -- ``0``
            measures every usable question in the file.
        chunk_size: LMCache chunk size the passages are aligned to. Must
            match the blend server's ``--chunk-size``.
        reference_repeats: Reference runs. Two or more give the per-question
            flip rate between identical runs, i.e. a noise floor.
        max_output_tokens: Token budget per answer. Must fit a reasoning
            model's thinking as well as the answer tags.
        template_kwargs: Chat-template variables sent as
            ``chat_template_kwargs`` (e.g. ``reasoning_effort``).
        verdict_metric: Score the verdict compares.
        judge: The LLM judge. Required when ``verdict_metric`` is
            ``JUDGE``, otherwise unused.
        task: The dataset's task type.
        tolerance: Largest acceptable mean score difference between the
            blend arm and the reference, in either direction, in score units
            (``0.05`` = 5 points). A gain counts too -- reusing KV should
            not improve answers, so a large one points at the setup.
        min_blend_share: Smallest share of looked-up prompt tokens the blend
            server must serve by blending, overall and in every window of
            questions, for a verdict to be valid.
        blend_metrics_url: The blend server's Prometheus ``/metrics`` URL.
            Empty means engagement is not checked and the verdict says so.
        output_dir: Results root. Each run writes into its own timestamped
            subdirectory -- the judge cache sits directly in it, shared by
            every run.
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
    verdict_metric: VerdictMetric = VerdictMetric.MATCH
    judge: JudgeSettings | None = None
    task: TaskType = TaskType.QA
    tolerance: float = 0.05
    min_blend_share: float = 0.3
    blend_metrics_url: str = ""
    output_dir: str = "quality_results"
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")
        if not self.dataset:
            raise ValueError("dataset must be non-empty")
        roles = [arm.role for arm in self.arms]
        for required in (ArmRole.REFERENCE, ArmRole.BLEND):
            if roles.count(required) != 1:
                raise ValueError(
                    f"exactly one {required.value} arm is required, "
                    f"got {roles.count(required)}"
                )
        if roles.count(ArmRole.PREFIX_CACHE) > 1:
            raise ValueError("at most one apc arm is allowed")
        if any(not arm.url for arm in self.arms):
            raise ValueError("every arm needs a non-empty url")
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
        if not 0.0 <= self.tolerance <= 1.0:
            raise ValueError(f"tolerance must be in [0, 1], got {self.tolerance}")
        if not 0.0 <= self.min_blend_share <= 1.0:
            raise ValueError(
                f"min_blend_share must be in [0, 1], got {self.min_blend_share}"
            )
        if not self.output_dir:
            raise ValueError("output_dir must be non-empty")
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

    def arm(self, role: ArmRole) -> ArmEndpoint:
        """Return the arm with *role*.

        Args:
            role: The role to look up.

        Returns:
            The configured arm.

        Raises:
            KeyError: If no arm has *role* (only possible for
                ``PREFIX_CACHE``, which is optional).
        """
        for endpoint in self.arms:
            if endpoint.role is role:
                return endpoint
        raise KeyError(role.value)

    def has_arm(self, role: ArmRole) -> bool:
        """Return whether an arm with *role* is configured."""
        return any(endpoint.role is role for endpoint in self.arms)

    def fingerprint_material(self) -> Mapping[str, object]:
        """Return the settings that determine the prompts and the scoring.

        Two runs are comparable only when these match.

        Returns:
            A JSON-serializable mapping.
        """
        material: dict[str, object] = {
            "model": self.model,
            "dataset": self.dataset,
            "task": self.task.value,
            "scenario": self.scenario.value,
            "num_samples": self.num_samples,
            "chunk_size": self.chunk_size,
            "max_output_tokens": self.max_output_tokens,
            "template_kwargs": dict(self.template_kwargs),
            "seed": self.seed,
            "verdict_metric": self.verdict_metric.value,
        }
        if self.verdict_metric is VerdictMetric.JUDGE and self.judge is not None:
            # The judge's model decides its scores -- its URL does not.
            material["judge_model"] = self.judge.model
        return material
