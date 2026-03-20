# SPDX-License-Identifier: Apache-2.0

# Standard
import random

# Third Party
import torch

# Local
from .config import TestConfig, VLLMBufferFormat


def _create_random_tensor(
    shape: list,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create a random tensor, handling FP8 which doesn't support torch.rand."""
    if dtype == torch.float8_e4m3fn:
        # FP8 doesn't support direct random generation; create as bf16 then cast
        return torch.rand(shape, dtype=torch.bfloat16, device=device).to(dtype)
    return torch.rand(shape, dtype=dtype, device=device)


def _create_zero_tensor(
    shape: list,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create a zero tensor, handling FP8."""
    if dtype == torch.float8_e4m3fn:
        return torch.zeros(shape, dtype=torch.bfloat16, device=device).to(dtype)
    return torch.zeros(shape, dtype=dtype, device=device)


def create_vllm_tensors(
    config: TestConfig,
    device: torch.device,
) -> list:
    """Create vLLM paged buffer tensors based on the format.

    Returns a list of tensors:
    - NORMAL: L tensors [2, NB, BS, NH, HS]
    - CROSS_LAYER: 1 tensor [NB, NL, 2, BS, NH, HS]
    - MLA: L tensors [NB, BS, HS]
    - FLASH_INFER: L tensors [NB, 2, BS, NH, HS]
    - SGLANG_MHA: 2L tensors [NBBS, NH, HS] (first L=K, next L=V)
    - SGLANG_MLA: L tensors [NBBS, 1, HS]
    """
    nb = config.num_blocks
    bs = config.block_size
    nh = config.num_heads
    hs = config.head_size
    nl = config.num_layers
    nbbs = config.nbbs

    if config.vllm_format == VLLMBufferFormat.NORMAL:
        shape = [2, nb, bs, nh, hs]
        return [_create_random_tensor(shape, config.dtype, device) for _ in range(nl)]

    elif config.vllm_format == VLLMBufferFormat.CROSS_LAYER:
        shape = [nb, nl, 2, bs, nh, hs]
        return [_create_random_tensor(shape, config.dtype, device)]

    elif config.vllm_format == VLLMBufferFormat.MLA:
        shape = [nb, bs, hs]
        return [_create_random_tensor(shape, config.dtype, device) for _ in range(nl)]

    elif config.vllm_format == VLLMBufferFormat.FLASH_INFER:
        shape = [nb, 2, bs, nh, hs]
        return [_create_random_tensor(shape, config.dtype, device) for _ in range(nl)]

    elif config.vllm_format == VLLMBufferFormat.SGLANG_MHA:
        shape = [nbbs, nh, hs]
        return [
            _create_random_tensor(shape, config.dtype, device) for _ in range(2 * nl)
        ]

    elif config.vllm_format == VLLMBufferFormat.SGLANG_MLA:
        shape = [nbbs, 1, hs]
        return [_create_random_tensor(shape, config.dtype, device) for _ in range(nl)]

    raise ValueError(f"Unknown format: {config.vllm_format}")


def create_zero_vllm_tensors(
    config: TestConfig,
    device: torch.device,
) -> list:
    """Create zeroed vLLM tensors (same shapes as create_vllm_tensors)."""
    nb = config.num_blocks
    bs = config.block_size
    nh = config.num_heads
    hs = config.head_size
    nl = config.num_layers
    nbbs = config.nbbs

    if config.vllm_format == VLLMBufferFormat.NORMAL:
        shape = [2, nb, bs, nh, hs]
        return [_create_zero_tensor(shape, config.dtype, device) for _ in range(nl)]

    elif config.vllm_format == VLLMBufferFormat.CROSS_LAYER:
        shape = [nb, nl, 2, bs, nh, hs]
        return [_create_zero_tensor(shape, config.dtype, device)]

    elif config.vllm_format == VLLMBufferFormat.MLA:
        shape = [nb, bs, hs]
        return [_create_zero_tensor(shape, config.dtype, device) for _ in range(nl)]

    elif config.vllm_format == VLLMBufferFormat.FLASH_INFER:
        shape = [nb, 2, bs, nh, hs]
        return [_create_zero_tensor(shape, config.dtype, device) for _ in range(nl)]

    elif config.vllm_format == VLLMBufferFormat.SGLANG_MHA:
        shape = [nbbs, nh, hs]
        return [_create_zero_tensor(shape, config.dtype, device) for _ in range(2 * nl)]

    elif config.vllm_format == VLLMBufferFormat.SGLANG_MLA:
        shape = [nbbs, 1, hs]
        return [_create_zero_tensor(shape, config.dtype, device) for _ in range(nl)]

    raise ValueError(f"Unknown format: {config.vllm_format}")


def create_memory_objects(config: TestConfig, device: torch.device = None) -> list:
    """Create LMCache memory objects on the given device.

    If device is None or a CUDA device, tensors are created on GPU.
    If device is CPU, tensors are created as pinned CPU memory.

    Returns a list of num_memory_objects tensors:
    - Non-MLA: each [2, L, tokens_per_object, hidden_dim]
    - MLA: each [1, L, tokens_per_object, hidden_dim]
    """
    kv = config.kv_dim
    nl = config.num_layers
    t = config.tokens_per_object
    d = config.hidden_dim

    shape = [kv, nl, t, d]

    if device is None:
        device = torch.device("cuda")

    objects = []
    for _ in range(config.num_memory_objects):
        if config.dtype == torch.float8_e4m3fn:
            tensor = torch.zeros(shape, dtype=torch.bfloat16, device=device)
            tensor = tensor.to(config.dtype)
        else:
            tensor = torch.zeros(shape, dtype=config.dtype, device=device)
        if device.type == "cpu":
            tensor = tensor.pin_memory()
        objects.append(tensor)
    return objects


def create_block_ids(config: TestConfig, seed: int = 42) -> torch.Tensor:
    """Create random unique block indices as a pinned CPU int64 tensor.

    Returns tensor of shape [total_blocks] with unique values in [0, num_blocks).
    """
    rng = random.Random(seed)
    ids = rng.sample(range(config.num_blocks), config.total_blocks)
    return torch.tensor(ids, dtype=torch.int64, pin_memory=True)


def create_h2d_block_ids(
    config: TestConfig,
    exclude: torch.Tensor,
    seed: int = 123,
) -> torch.Tensor:
    """Create a set of block IDs disjoint from `exclude`.

    Used for the H2D target so we can verify correctness (writing to
    different blocks than we read from).
    """
    excluded_set = set(exclude.tolist())
    available = [i for i in range(config.num_blocks) if i not in excluded_set]
    rng = random.Random(seed)
    ids = rng.sample(available, config.total_blocks)
    return torch.tensor(ids, dtype=torch.int64, pin_memory=True)
