# SPDX-License-Identifier: Apache-2.0
"""Lossless predictive coding of FP8 KV chunks on the GPU (``remote_serde: fp8lin``).

``coder`` holds the fitted linear predictor and the chunk codec; ``rans`` the Triton
entropy coder. This package does not import the rest of LMCache.
"""

# Local
from .coder import ChunkCodec, checksum

__all__ = ["ChunkCodec", "checksum"]
