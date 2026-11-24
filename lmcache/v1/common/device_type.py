# SPDX-License-Identifier: Apache-2.0
"""The torch device type that ``lmcache.v1.platform`` detected.

``common`` must not import ``platform``, so ``platform`` writes the value here
right after detection. ``"cpu"`` is platform's own fallback when no
accelerator, or no torch, is found.
"""

torch_device_type = "cpu"
