# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for engine-benchmark tests."""

# Third Party
import pytest

# Local
from .fake_tokenizer import FakeTokenizer, make_fake_tokenizer


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    """A tokenizer stand-in with plenty of single-token words."""
    return make_fake_tokenizer()
