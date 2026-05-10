# SPDX-License-Identifier: Apache-2.0
"""CUDA-backed device + stream context manager.

Wraps the historic ``with torch.cuda.device(d), torch.cuda.stream(s):``
pattern in a single context manager so the dispatcher in
:mod:`lmcache.v1.platform.device_ctx` can hide the CPU vs. CUDA split
from the multiprocess server.
"""

# Future
from __future__ import annotations

# Standard
from contextlib import contextmanager
from typing import Any, Iterator

# Third Party
import torch


@contextmanager
def make_cuda_device_context(
    device: torch.device,
    stream: Any | None,
) -> Iterator[None]:
    """Activate ``device`` and (optionally) ``stream`` for the block."""
    with torch.cuda.device(device):
        if stream is None:
            yield
        else:
            with torch.cuda.stream(stream):
                yield
