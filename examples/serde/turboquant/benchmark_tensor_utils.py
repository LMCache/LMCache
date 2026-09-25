# SPDX-License-Identifier: Apache-2.0
"""Memory-bounded tensor metrics for the TurboQuant microbenchmark."""

# Standard
from typing import NamedTuple
import math

# Third Party
import torch


class TensorErrorMetrics(NamedTuple):
    """Accuracy metrics for an original and recovered tensor pair."""

    corr: float
    mean_abs_err: float
    max_abs_err: float


def tensor_error_metrics(
    original: torch.Tensor,
    recovered: torch.Tensor,
    chunk_elements: int = 16 * 1024 * 1024,
) -> TensorErrorMetrics:
    """Compute error metrics without materializing full float32 copies.

    Chunking also keeps each dot product below the signed 32-bit element limit
    used by some CUDA BLAS implementations.
    """
    if original.shape != recovered.shape:
        raise ValueError("original and recovered tensors must have the same shape")
    if original.device != recovered.device:
        raise ValueError("original and recovered tensors must use the same device")
    if original.numel() == 0:
        raise ValueError("original and recovered tensors must be non-empty")
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")

    original_flat = original.reshape(-1)
    recovered_flat = recovered.reshape(-1)
    totals = torch.zeros(6, dtype=torch.float64, device=original.device)
    max_abs_error = torch.zeros((), dtype=torch.float32, device=original.device)

    for start in range(0, original.numel(), chunk_elements):
        end = min(start + chunk_elements, original.numel())
        original_chunk = original_flat[start:end].float()
        recovered_chunk = recovered_flat[start:end].float()

        totals[0].add_(original_chunk.sum())
        totals[1].add_(recovered_chunk.sum())
        totals[2].add_(torch.dot(original_chunk, original_chunk))
        totals[3].add_(torch.dot(recovered_chunk, recovered_chunk))
        totals[4].add_(torch.dot(original_chunk, recovered_chunk))

        abs_error = torch.abs(original_chunk - recovered_chunk)
        totals[5].add_(abs_error.sum())
        max_abs_error = torch.maximum(max_abs_error, abs_error.max())

    (
        sum_original,
        sum_recovered,
        sum_original_sq,
        sum_recovered_sq,
        sum_product,
        sum_abs_error,
    ) = totals.tolist()
    numel = float(original.numel())
    covariance = sum_product - sum_original * sum_recovered / numel
    original_variance = max(sum_original_sq - sum_original * sum_original / numel, 0.0)
    recovered_variance = max(
        sum_recovered_sq - sum_recovered * sum_recovered / numel, 0.0
    )
    denominator = math.sqrt(original_variance * recovered_variance)
    corr = covariance / denominator if denominator else float("nan")

    return TensorErrorMetrics(
        corr=corr,
        mean_abs_err=sum_abs_error / numel,
        max_abs_err=max_abs_error.item(),
    )
