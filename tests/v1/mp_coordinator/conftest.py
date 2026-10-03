# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for MP coordinator tests."""

# Standard
from collections.abc import Iterator
from unittest.mock import patch

# Third Party
import pytest


@pytest.fixture(autouse=True)
def _isolate_key_directory_metric_registration() -> Iterator[None]:
    """Avoid registering process-global OTel instruments in app tests."""
    with patch("lmcache.v1.mp_coordinator.app.register_key_directory_metrics"):
        yield
