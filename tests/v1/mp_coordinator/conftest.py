# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for MP coordinator tests."""

# Standard
from collections.abc import Iterator
from unittest.mock import patch

# Third Party
import pytest

# First Party
from lmcache.v1.mp_coordinator.metric_sources import get_and_clear_metric_sources


@pytest.fixture(autouse=True)
def _isolate_metric_registration() -> Iterator[None]:
    """Start each test with no registered metric sources, and keep app tests
    from registering process-global OTel instruments."""
    get_and_clear_metric_sources()
    with patch("lmcache.v1.mp_coordinator.app.register_coordinator_metrics"):
        yield
