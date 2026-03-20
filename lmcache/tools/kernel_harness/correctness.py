# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Callable
import logging

# Third Party
import torch

# Local
from .config import Direction, TestConfig, VLLMBufferFormat
from .reference import reference_multi_layer_block_kv_transfer
from .tensor_factory import (
    create_block_ids,
    create_h2d_block_ids,
    create_memory_objects,
    create_vllm_tensors,
    create_zero_vllm_tensors,
)

logger = logging.getLogger(__name__)


def _get_block_data(
    vllm_tensors: list,
    config: TestConfig,
    block_idx: int,
) -> list:
    """Extract all layer data for a given block, returned as a list of tensors."""
    bs = config.block_size
    nl = config.num_layers
    results = []
    for layer_idx in range(nl):
        if config.vllm_format == VLLMBufferFormat.NORMAL:
            results.append(vllm_tensors[layer_idx][:, block_idx, :, :, :].clone())
        elif config.vllm_format == VLLMBufferFormat.CROSS_LAYER:
            results.append(vllm_tensors[0][block_idx, layer_idx, :, :, :, :].clone())
        elif config.vllm_format == VLLMBufferFormat.MLA:
            results.append(vllm_tensors[layer_idx][block_idx, :, :].clone())
        elif config.vllm_format == VLLMBufferFormat.FLASH_INFER:
            results.append(vllm_tensors[layer_idx][block_idx, :, :, :, :].clone())
        elif config.vllm_format == VLLMBufferFormat.SGLANG_MHA:
            token_start = block_idx * bs
            token_end = token_start + bs
            k = vllm_tensors[layer_idx][token_start:token_end, :, :].clone()
            v = vllm_tensors[nl + layer_idx][token_start:token_end, :, :].clone()
            results.append(torch.stack([k, v], dim=0))
        elif config.vllm_format == VLLMBufferFormat.SGLANG_MLA:
            token_start = block_idx * bs
            token_end = token_start + bs
            results.append(vllm_tensors[layer_idx][token_start:token_end, 0, :].clone())
    return results


def _check_blocks_equal(
    source_data: list,
    target_data: list,
    config: TestConfig,
) -> bool:
    """Check if two sets of per-layer block data are equal."""
    for layer_idx in range(config.num_layers):
        src = source_data[layer_idx]
        tgt = target_data[layer_idx]
        if not torch.equal(src, tgt):
            return False
    return True


def _check_block_is_zero(
    vllm_tensors: list,
    config: TestConfig,
    block_idx: int,
) -> bool:
    """Check if a block in the vLLM tensors is all zeros."""
    bs = config.block_size
    nl = config.num_layers
    for layer_idx in range(nl):
        if config.vllm_format == VLLMBufferFormat.NORMAL:
            block = vllm_tensors[layer_idx][:, block_idx, :, :, :]
        elif config.vllm_format == VLLMBufferFormat.CROSS_LAYER:
            block = vllm_tensors[0][block_idx, layer_idx, :, :, :, :]
        elif config.vllm_format == VLLMBufferFormat.MLA:
            block = vllm_tensors[layer_idx][block_idx, :, :]
        elif config.vllm_format == VLLMBufferFormat.FLASH_INFER:
            block = vllm_tensors[layer_idx][block_idx, :, :, :, :]
        elif config.vllm_format == VLLMBufferFormat.SGLANG_MHA:
            token_start = block_idx * bs
            token_end = token_start + bs
            k = vllm_tensors[layer_idx][token_start:token_end, :, :]
            v = vllm_tensors[nl + layer_idx][token_start:token_end, :, :]
            block = torch.cat([k, v], dim=0)
        elif config.vllm_format == VLLMBufferFormat.SGLANG_MLA:
            token_start = block_idx * bs
            token_end = token_start + bs
            block = vllm_tensors[layer_idx][token_start:token_end, 0, :]
        else:
            return False

        # For FP8, compare as float
        if block.dtype == torch.float8_e4m3fn:
            block = block.to(torch.float32)
        if block.abs().sum().item() != 0:
            return False
    return True


def run_correctness_test(
    config: TestConfig,
    kernel_fn: Callable = reference_multi_layer_block_kv_transfer,
    mem_device: torch.device = None,
) -> bool:
    """Run D2H -> H2D roundtrip correctness test.

    Steps:
    1. Create source vLLM tensors with random data on GPU
    2. Create empty memory objects on mem_device (default: GPU)
    3. D2H: copy source -> memory objects using block_ids_d2h
    4. H2D: copy memory objects -> target vLLM using block_ids_h2d (different blocks)
    5. Verify: target[h2d_block] == source[d2h_block] for all corresponding pairs
    6. Verify: untouched blocks in target remain zero

    Returns True if all checks pass.
    """
    device = torch.device("cuda")

    # Create tensors
    source_vllm = create_vllm_tensors(config, device)
    target_vllm = create_zero_vllm_tensors(config, device)
    mem_objects = create_memory_objects(config, mem_device)

    # Create disjoint block ID sets
    block_ids_d2h = create_block_ids(config, seed=42)
    block_ids_h2d = create_h2d_block_ids(config, exclude=block_ids_d2h, seed=123)

    # D2H: source vLLM -> memory objects
    kernel_fn(source_vllm, mem_objects, block_ids_d2h, config, Direction.D2H)
    torch.cuda.synchronize()

    # H2D: memory objects -> target vLLM
    kernel_fn(target_vllm, mem_objects, block_ids_h2d, config, Direction.H2D)
    torch.cuda.synchronize()

    # Verify: corresponding blocks should match
    passed = True
    for i in range(config.total_blocks):
        # Skip prefix blocks
        if i < config.skip_prefix_n_blocks:
            continue

        src_block_id = block_ids_d2h[i].item()
        tgt_block_id = block_ids_h2d[i].item()

        src_data = _get_block_data(source_vllm, config, src_block_id)
        tgt_data = _get_block_data(target_vllm, config, tgt_block_id)

        if not _check_blocks_equal(src_data, tgt_data, config):
            logger.error(
                "FAIL: Block mismatch at index %d (src_block=%d, tgt_block=%d)",
                i,
                src_block_id,
                tgt_block_id,
            )
            passed = False

    # Verify: blocks that were NOT written should remain zero
    h2d_set = set(block_ids_h2d.tolist())
    # Sample a few untouched blocks to check
    untouched_blocks = [
        b for b in range(min(config.num_blocks, 200)) if b not in h2d_set
    ][:10]
    for block_idx in untouched_blocks:
        if not _check_block_is_zero(target_vllm, config, block_idx):
            logger.error("FAIL: Untouched block %d is not zero", block_idx)
            passed = False

    # Verify skip_prefix_n_blocks: skipped blocks in target should be zero
    if config.skip_prefix_n_blocks > 0:
        for i in range(config.skip_prefix_n_blocks):
            tgt_block_id = block_ids_h2d[i].item()
            if not _check_block_is_zero(target_vllm, config, tgt_block_id):
                logger.error(
                    "FAIL: Skipped prefix block %d (tgt_block=%d) is not zero",
                    i,
                    tgt_block_id,
                )
                passed = False

    return passed


def run_skip_prefix_test(
    config: TestConfig,
    kernel_fn: Callable = reference_multi_layer_block_kv_transfer,
    mem_device: torch.device = None,
) -> bool:
    """Test that skip_prefix_n_blocks correctly skips the first N blocks.

    Creates a modified config with skip_prefix_n_blocks=4 and verifies
    that the first 4 blocks are not copied.
    """
    # Standard
    from dataclasses import replace

    modified_config = replace(config, skip_prefix_n_blocks=4)
    return run_correctness_test(modified_config, kernel_fn, mem_device)
