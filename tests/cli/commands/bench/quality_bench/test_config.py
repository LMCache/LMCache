# SPDX-License-Identifier: Apache-2.0
"""Tests for ``bench quality`` config validation."""

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.quality_bench.config import (
    ArmEndpoint,
    ArmRole,
    JudgeSettings,
    QualityBenchConfig,
    VerdictMetric,
)

_ARMS = [
    ArmEndpoint(ArmRole.REFERENCE, "http://ref"),
    ArmEndpoint(ArmRole.BLEND, "http://blend"),
]


def _config(**overrides) -> QualityBenchConfig:
    fields: dict[str, object] = {"model": "m", "dataset": "d", "arms": list(_ARMS)}
    fields.update(overrides)
    return QualityBenchConfig(**fields)  # type: ignore[arg-type]


class TestQualityBenchConfig:
    def test_minimal_config_is_valid(self) -> None:
        config = _config()
        assert config.arm(ArmRole.BLEND).url == "http://blend"
        assert not config.has_arm(ArmRole.PREFIX_CACHE)

    @pytest.mark.parametrize("missing", [ArmRole.REFERENCE, ArmRole.BLEND])
    def test_reference_and_blend_are_required(self, missing: ArmRole) -> None:
        arms = [a for a in _ARMS if a.role is not missing]
        with pytest.raises(ValueError, match=missing.value):
            _config(arms=arms)

    def test_duplicate_roles_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="blend"):
            _config(arms=[*_ARMS, ArmEndpoint(ArmRole.BLEND, "http://b2")])

    def test_at_most_one_apc_arm(self) -> None:
        apc = ArmEndpoint(ArmRole.PREFIX_CACHE, "http://apc")
        with pytest.raises(ValueError, match="apc"):
            _config(arms=[*_ARMS, apc, apc])

    def test_empty_url_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="url"):
            _config(arms=[_ARMS[0], ArmEndpoint(ArmRole.BLEND, "")])

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("model", ""),
            ("dataset", ""),
            ("num_samples", -1),
            ("chunk_size", 0),
            ("reference_repeats", 0),
            ("max_output_tokens", 0),
            ("tolerance", 1.5),
            ("min_blend_share", -0.1),
            ("output_dir", ""),
        ],
    )
    def test_invalid_values_are_rejected(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match=field):
            _config(**{field: value})

    def test_missing_optional_arm_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            _config().arm(ArmRole.PREFIX_CACHE)

    def test_fingerprint_changes_with_the_prompt_settings(self) -> None:
        assert _config().fingerprint_material() != (
            _config(chunk_size=128).fingerprint_material()
        )

    def test_fingerprint_ignores_the_arm_urls(self) -> None:
        """Arms differ by design; their prompts must not."""
        other = [ArmEndpoint(a.role, a.url + "x") for a in _ARMS]
        assert _config().fingerprint_material() == (
            _config(arms=other).fingerprint_material()
        )

    def test_judge_metric_needs_a_judge(self) -> None:
        with pytest.raises(ValueError, match="judge"):
            _config(verdict_metric=VerdictMetric.JUDGE)

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
            _config(verdict_metric=VerdictMetric.JUDGE, judge=settings)

    def test_fingerprint_records_the_metric_and_judge_model(self) -> None:
        judged = _config(
            verdict_metric=VerdictMetric.JUDGE,
            judge=JudgeSettings(url="http://judge", model="j"),
        ).fingerprint_material()
        assert judged["verdict_metric"] == "judge"
        assert judged["judge_model"] == "j"
        assert judged != _config().fingerprint_material()

    def test_fingerprint_ignores_the_judge_url(self) -> None:
        def judged(url: str) -> object:
            return _config(
                verdict_metric=VerdictMetric.JUDGE,
                judge=JudgeSettings(url=url, model="j"),
            ).fingerprint_material()

        assert judged("http://a") == judged("http://b")
