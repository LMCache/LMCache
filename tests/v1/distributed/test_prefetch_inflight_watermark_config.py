# SPDX-License-Identifier: Apache-2.0

"""Tests for the L2 prefetch in-flight watermark command line plumbing."""

# Standard
from collections.abc import Iterator
import logging

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import parse_args

_CONFIG_LOGGER = "lmcache.v1.distributed.config"
_BASE_ARGS = ["--l1-size-gb", "1", "--eviction-policy", "noop"]


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def config_warnings() -> Iterator[list[logging.LogRecord]]:
    """Collect WARNING records from the config module's logger, which does
    not propagate to pytest's caplog."""
    handler = _CaptureHandler()
    lg = logging.getLogger(_CONFIG_LOGGER)
    lg.addHandler(handler)
    try:
        yield handler.records
    finally:
        lg.removeHandler(handler)


def test_default_watermark():
    config = parse_args(_BASE_ARGS)
    assert config.prefetch_inflight_watermark == 0.5


def test_watermark_flag_is_plumbed():
    config = parse_args(_BASE_ARGS + ["--l2-prefetch-inflight-watermark", "0.3"])
    assert config.prefetch_inflight_watermark == 0.3


@pytest.mark.parametrize("value", ["0", "-0.1", "1.5"])
def test_watermark_out_of_range_is_rejected(value):
    with pytest.raises(ValueError, match="prefetch_inflight_watermark"):
        parse_args(_BASE_ARGS + ["--l2-prefetch-inflight-watermark", value])


def test_deprecated_max_in_flight_is_accepted_and_ignored(config_warnings):
    config = parse_args(_BASE_ARGS + ["--l2-prefetch-max-in-flight", "16"])

    assert config.prefetch_inflight_watermark == 0.5
    assert any(
        record.levelno == logging.WARNING
        and "--l2-prefetch-max-in-flight is deprecated" in record.getMessage()
        for record in config_warnings
    )


def test_no_deprecation_warning_without_old_flag(config_warnings):
    parse_args(_BASE_ARGS)
    assert not any(
        "--l2-prefetch-max-in-flight" in record.getMessage()
        for record in config_warnings
    )
