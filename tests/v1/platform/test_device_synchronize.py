# SPDX-License-Identifier: Apache-2.0
"""Device synchronization follows tensor placement, including device indices."""

# Standard
from unittest.mock import MagicMock, call

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform import synchronize_device


def test_synchronize_follows_device_across_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU calls skip synchronization; accelerator calls retain their indices."""
    synchronize = MagicMock()
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    for device in ("cpu", "cuda:1", "cpu", "cuda:0"):
        synchronize_device(torch.device(device))
    assert synchronize.call_args_list == [
        call(torch.device("cuda:1")),
        call(torch.device("cuda:0")),
    ]


def test_unregistered_accelerator_does_not_skip_synchronization() -> None:
    """Unknown non-CPU devices fail instead of silently leaving work pending."""
    with pytest.raises(RuntimeError, match="No DeviceSpec registered"):
        synchronize_device(torch.device("meta"))
