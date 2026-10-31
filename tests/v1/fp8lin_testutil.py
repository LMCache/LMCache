# SPDX-License-Identifier: Apache-2.0
"""Synthetic fp8lin coder directories for tests (random but valid weights)."""

# Standard
from pathlib import Path
import json

# Third Party
import numpy as np


def write_coder(path: Path, L: int = 3, H: int = 2, D: int = 128,
                kv_scale: tuple[float, float] = (0.05, 0.02)) -> None:  # fmt: skip
    """Write an fp8lin coder directory with random cross-head + previous-layer maps.

    Prediction quality only affects the size of the coded output, never its
    correctness, so tests can use random weights.

    Args:
        path: Directory to write (must exist).
        L: Number of layers.
        H: Number of KV heads.
        D: Head dimension (fp8lin needs 128).
        kv_scale: E4M3 scale for K and V of every layer (non-FP8 KV caches).
    """
    rng = np.random.default_rng(0)
    C = H * D
    inv = 1.0 / (10000 ** (np.arange(0, D, 2) / D))
    th = np.concatenate([inv, inv])
    nz = 4 * C + 1  # groups curK, curV, p (2C), bias
    for layer in range(L):
        blocks = {}
        for b in range(2 * H):
            h, isv = b % H, b >= H
            m = C + h * D if isv else h * D
            cols = np.concatenate([np.arange(m), np.arange(2 * C, 4 * C), [nz - 1]])
            blocks[f"cols{b}"] = cols
            blocks[f"W{b}"] = (rng.standard_normal((len(cols), D)) * 0.02).astype(
                np.float32
            )
        np.savez(
            path / f"layer_{layer:03d}.npz",
            H=H,
            D=D,
            C=C,
            th=th,
            groups=np.array(["curK", "curV", "p"]),
            goff=np.array([0, C, 2 * C]),
            sd=np.ones(4 * C, np.float32),
            sig=np.full(2 * C, 8.0, np.float32),
            Wh=np.zeros((2 * H, 4), np.float32),
            **blocks,
        )
    meta = {
        "format": "fp8lin-coder-v1",
        "num_layers": L,
        "num_kv_heads": H,
        "head_dim": D,
        "kv_scale": {"k": [kv_scale[0]] * L, "v": [kv_scale[1]] * L},
        "width_index": 5,
        "lane_tokens": 128,
        "max_position": 4096,
    }
    (path / "meta.json").write_text(json.dumps(meta))
