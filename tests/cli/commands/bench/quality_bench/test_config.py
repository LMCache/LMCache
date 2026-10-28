# SPDX-License-Identifier: Apache-2.0
"""Tests for ``bench quality`` config validation."""

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.quality_bench.core.config import (
    ArmEndpoint,
    ArmRole,
    JudgeSettings,
    QualityAnalyzeConfig,
    QualityRunConfig,
    VerdictMetric,
)

_ARMS = [
    ArmEndpoint(ArmRole.REFERENCE, "http://ref"),
    ArmEndpoint(ArmRole.CANDIDATE, "http://candidate"),
]


def _run_config(**overrides) -> QualityRunConfig:
    fields: dict[str, object] = {"model": "m", "dataset": "d", "arms": list(_ARMS)}
    fields.update(overrides)
    return QualityRunConfig(**fields)  # type: ignore[arg-type]


def _analyze_config(**overrides) -> QualityAnalyzeConfig:
    fields: dict[str, object] = {"results_dir": "out"}
    fields.update(overrides)
    return QualityAnalyzeConfig(**fields)  # type: ignore[arg-type]


class TestQualityRunConfig:
    def test_minimal_config_is_valid(self) -> None:
        config = _run_config()
        assert config.arm(ArmRole.CANDIDATE).url == "http://candidate"
        assert not config.has_arm(ArmRole.CONTROL)

    @pytest.mark.parametrize("role", list(ArmRole))
    def test_a_single_arm_is_enough(self, role: ArmRole) -> None:
        """Arms can be run one at a time, e.g. one per GPU server."""
        config = _run_config(arms=[ArmEndpoint(role, "http://x")])
        assert config.has_arm(role)

    def test_at_least_one_arm_is_required(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            _run_config(arms=[])

    @pytest.mark.parametrize("role", list(ArmRole))
    def test_duplicate_roles_are_rejected(self, role: ArmRole) -> None:
        arm = ArmEndpoint(role, "http://x")
        with pytest.raises(ValueError, match=role.value):
            _run_config(arms=[arm, arm])

    def test_empty_url_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="url"):
            _run_config(arms=[_ARMS[0], ArmEndpoint(ArmRole.CANDIDATE, "")])

    def test_metrics_url_needs_a_candidate_arm(self) -> None:
        with pytest.raises(ValueError, match="--candidate-url"):
            _run_config(arms=[_ARMS[0]], blend_metrics_url="http://m/metrics")

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("model", ""),
            ("dataset", ""),
            ("num_samples", -1),
            ("chunk_size", 0),
            ("reference_repeats", 0),
            ("max_output_tokens", 0),
        ],
    )
    def test_invalid_values_are_rejected(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match=field):
            _run_config(**{field: value})

    def test_missing_arm_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            _run_config().arm(ArmRole.CONTROL)

    def test_plan_settings_change_with_the_prompt_settings(self) -> None:
        assert _run_config().plan_settings() != (
            _run_config(chunk_size=128).plan_settings()
        )

    def test_plan_settings_ignore_the_arms(self) -> None:
        """Runs of different arms into one directory must share a plan."""
        only_candidate = [ArmEndpoint(ArmRole.CANDIDATE, "http://elsewhere")]
        assert _run_config().plan_settings() == (
            _run_config(arms=only_candidate).plan_settings()
        )


class TestQualityAnalyzeConfig:
    def test_minimal_config_is_valid(self) -> None:
        assert _analyze_config().verdict_metric is VerdictMetric.MATCH

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("results_dir", ""),
            ("tolerance", 1.5),
            ("min_blend_share", -0.1),
            ("min_reuse_ratio", 1.5),
        ],
    )
    def test_invalid_values_are_rejected(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match=field):
            _analyze_config(**{field: value})

    def test_judge_metric_needs_a_judge(self) -> None:
        with pytest.raises(ValueError, match="judge"):
            _analyze_config(verdict_metric=VerdictMetric.JUDGE)

    @pytest.mark.parametrize(
        "settings",
        [
            JudgeSettings(url="", model="j"),
            JudgeSettings(url="http://judge", model=""),
            JudgeSettings(url="http://judge", model="j", max_tokens=0),
            JudgeSettings(url="http://judge", model="j", concurrency=0),
        ],
    )
    def test_invalid_judge_settings_are_rejected(self, settings: JudgeSettings) -> None:
        with pytest.raises(ValueError):
            _analyze_config(verdict_metric=VerdictMetric.JUDGE, judge=settings)

    def test_scoring_settings_record_the_metric_and_judge_model(self) -> None:
        judged = _analyze_config(
            verdict_metric=VerdictMetric.JUDGE,
            judge=JudgeSettings(url="http://judge", model="j"),
        ).scoring_settings()
        assert judged == {"verdict_metric": "judge", "judge_model": "j"}
        assert _analyze_config().scoring_settings() == {"verdict_metric": "match"}

    def test_scoring_settings_ignore_the_judge_url(self) -> None:
        def judged(url: str) -> object:
            return _analyze_config(
                verdict_metric=VerdictMetric.JUDGE,
                judge=JudgeSettings(url=url, model="j"),
            ).scoring_settings()

        assert judged("http://a") == judged("http://b")
