# SPDX-License-Identifier: Apache-2.0

# Third Party
import torch

# Local
from .config import Direction, TestConfig, VLLMBufferFormat


def _get_vllm_block(
    vllm_tensors: list,
    config: TestConfig,
    layer_idx: int,
    block_idx: int,
) -> torch.Tensor:
    """Extract a single block from vLLM tensors for a given layer.

    Returns:
    - Non-MLA formats: shape [2, BS, NH, HS]
    - MLA formats: shape [BS, HS]
    """
    bs = config.block_size

    if config.vllm_format == VLLMBufferFormat.NORMAL:
        # vllm_tensors[layer]: [2, NB, BS, NH, HS]
        return vllm_tensors[layer_idx][:, block_idx, :, :, :]

    elif config.vllm_format == VLLMBufferFormat.CROSS_LAYER:
        # vllm_tensors[0]: [NB, NL, 2, BS, NH, HS]
        return vllm_tensors[0][block_idx, layer_idx, :, :, :, :]

    elif config.vllm_format == VLLMBufferFormat.MLA:
        # vllm_tensors[layer]: [NB, BS, HS]
        return vllm_tensors[layer_idx][block_idx, :, :]

    elif config.vllm_format == VLLMBufferFormat.FLASH_INFER:
        # vllm_tensors[layer]: [NB, 2, BS, NH, HS]
        return vllm_tensors[layer_idx][block_idx, :, :, :, :]

    elif config.vllm_format == VLLMBufferFormat.SGLANG_MHA:
        # vllm_tensors[layer] = K, vllm_tensors[nl + layer] = V
        # each: [NBBS, NH, HS], flat token indexing
        token_start = block_idx * bs
        token_end = token_start + bs
        nl = config.num_layers
        k_block = vllm_tensors[layer_idx][token_start:token_end, :, :]
        v_block = vllm_tensors[nl + layer_idx][token_start:token_end, :, :]
        return torch.stack([k_block, v_block], dim=0)  # [2, BS, NH, HS]

    elif config.vllm_format == VLLMBufferFormat.SGLANG_MLA:
        # vllm_tensors[layer]: [NBBS, 1, HS], flat token indexing
        token_start = block_idx * bs
        token_end = token_start + bs
        return vllm_tensors[layer_idx][token_start:token_end, 0, :]  # [BS, HS]

    raise ValueError(f"Unknown format: {config.vllm_format}")


def _set_vllm_block(
    vllm_tensors: list,
    config: TestConfig,
    layer_idx: int,
    block_idx: int,
    data: torch.Tensor,
) -> None:
    """Write a block into vLLM tensors for a given layer."""
    bs = config.block_size

    if config.vllm_format == VLLMBufferFormat.NORMAL:
        vllm_tensors[layer_idx][:, block_idx, :, :, :] = data

    elif config.vllm_format == VLLMBufferFormat.CROSS_LAYER:
        vllm_tensors[0][block_idx, layer_idx, :, :, :, :] = data

    elif config.vllm_format == VLLMBufferFormat.MLA:
        vllm_tensors[layer_idx][block_idx, :, :] = data

    elif config.vllm_format == VLLMBufferFormat.FLASH_INFER:
        vllm_tensors[layer_idx][block_idx, :, :, :, :] = data

    elif config.vllm_format == VLLMBufferFormat.SGLANG_MHA:
        # data: [2, BS, NH, HS] → split into K and V flat tensors
        token_start = block_idx * bs
        token_end = token_start + bs
        nl = config.num_layers
        vllm_tensors[layer_idx][token_start:token_end, :, :] = data[0]
        vllm_tensors[nl + layer_idx][token_start:token_end, :, :] = data[1]

    elif config.vllm_format == VLLMBufferFormat.SGLANG_MLA:
        # data: [BS, HS] → write into flat tensor
        token_start = block_idx * bs
        token_end = token_start + bs
        vllm_tensors[layer_idx][token_start:token_end, 0, :] = data


def reference_multi_layer_block_kv_transfer(
    vllm_tensors: list,
    memory_objects: list,
    block_ids: torch.Tensor,
    config: TestConfig,
    direction: Direction,
) -> None:
    """Pure Python/PyTorch reference implementation of block transfer.

    Performs the copy between vLLM paged buffers and LMCache memory objects
    at block granularity.

    Args:
        vllm_tensors: vLLM paged buffer tensors on GPU.
        memory_objects: LMCache memory objects on pinned CPU.
        block_ids: Block indices into the vLLM paged buffer.
        config: Test configuration.
        direction: D2H (vLLM->LMCache) or H2D (LMCache->vLLM).
    """
    bs = config.block_size
    bpo = config.blocks_per_object
    skip = config.skip_prefix_n_blocks

    for obj_idx, mem_obj in enumerate(memory_objects):
        obj_block_ids = block_ids[obj_idx * bpo : (obj_idx + 1) * bpo]

        for local_block_idx, block_id in enumerate(obj_block_ids):
            # Global block index across all memory objects
            global_block_idx = obj_idx * bpo + local_block_idx
            if global_block_idx < skip:
                continue

            block_id = block_id.item()
            token_start = local_block_idx * bs
            token_end = token_start + bs

            for layer_idx in range(config.num_layers):
                vllm_block = _get_vllm_block(vllm_tensors, config, layer_idx, block_id)

                if config.is_mla:
                    # MLA: vllm_block is [BS, HS], mem_obj is [1, L, T, D]
                    if direction == Direction.D2H:
                        mem_obj[0, layer_idx, token_start:token_end, :] = (
                            vllm_block.reshape(bs, config.hidden_dim).cpu()
                        )
                    else:
                        _set_vllm_block(
                            vllm_tensors,
                            config,
                            layer_idx,
                            block_id,
                            mem_obj[0, layer_idx, token_start:token_end, :]
                            .reshape(bs, config.head_size)
                            .to(vllm_tensors[layer_idx].device),
                        )
                else:
                    # Non-MLA: vllm_block is [2, BS, NH, HS],
                    # mem_obj is [2, L, T, D]
                    if direction == Direction.D2H:
                        for kv in range(2):
                            mem_obj[kv, layer_idx, token_start:token_end, :] = (
                                vllm_block[kv].reshape(bs, config.hidden_dim).cpu()
                            )
                    else:
                        combined = torch.stack(
                            [
                                mem_obj[kv, layer_idx, token_start:token_end, :]
                                .reshape(bs, config.num_heads, config.head_size)
                                .to(vllm_tensors[0].device)
                                for kv in range(2)
                            ],
                            dim=0,
                        )  # [2, BS, NH, HS]
                        _set_vllm_block(
                            vllm_tensors, config, layer_idx, block_id, combined
                        )
