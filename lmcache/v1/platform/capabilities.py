# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for platform capability detection.

All platform-specific checks are centralized here so that no other
module needs to call ``torch.cuda.is_available()`` or
``hasattr(os, "eventfd")`` directly.
"""

# Standard
import os

# Third Party
import torch

HAS_CUDA: bool = torch.cuda.is_available()
HAS_EVENTFD: bool = hasattr(os, "eventfd")
