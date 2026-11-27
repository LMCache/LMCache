# SPDX-License-Identifier: Apache-2.0

# Third Party
import torch


def rotary_embedding_k_fused(
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """Apply fused rotary embedding undo/redo to key tensor in-place.

    Reverses the rotary embedding at old_positions and applies the rotary
    embedding at new_positions. head_size is unused but kept for API
    compatibility with the CUDA equivalent.

    Args:
        old_positions: Token positions whose rotary embedding to reverse.
        new_positions: Token positions whose rotary embedding to apply.
        key: Key tensor to update in-place.
        head_size: Head size (unused; kept for API compatibility).
        cos_sin_cache: Precomputed cosine/sine cache indexed by position.
        is_neox: If True, uses NeoX-style rotary (contiguous halves);
            otherwise uses GPT-J-style (interleaved).
    """
    rot_dim = cos_sin_cache.shape[1]
    half_rot = rot_dim // 2

    old_cs = cos_sin_cache[old_positions]
    new_cs = cos_sin_cache[new_positions]

    oc, os = old_cs[:, :half_rot].unsqueeze(1), old_cs[:, half_rot:].unsqueeze(1)
    nc, ns = new_cs[:, :half_rot].unsqueeze(1), new_cs[:, half_rot:].unsqueeze(1)

    if is_neox:
        x = key[..., :half_rot]
        y = key[..., half_rot:rot_dim]
    else:
        x = key[..., :rot_dim:2]
        y = key[..., 1:rot_dim:2]

    x_rev = x * oc + y * os
    y_rev = y * oc - x * os

    x_out = x_rev * nc - y_rev * ns
    y_out = y_rev * nc + x_rev * ns

    if is_neox:
        key[..., :half_rot] = x_out
        key[..., half_rot:rot_dim] = y_out
    else:
        key[..., :rot_dim:2] = x_out
        key[..., 1:rot_dim:2] = y_out


def rotary_embedding_k_fused_strided(
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    head_stride: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """Strided rotary_embedding_k_fused: ``key``'s last dim is ``head_stride``
    per head; rotate only the leading ``head_size`` (K) in place via a view,
    leaving the trailing V. ``head_stride == head_size`` is the contiguous case.
    """
    rotary_embedding_k_fused(
        old_positions,
        new_positions,
        key[..., :head_size],
        head_size,
        cos_sin_cache,
        is_neox,
    )
