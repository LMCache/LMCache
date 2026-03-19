# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from enum import Enum, auto

# Third Party
import torch


class VLLMBufferFormat(Enum):
    """Which vLLM paged buffer configuration to test."""

    NORMAL = auto()  # NL_X_TWO_NB_BS_NH_HS: L separate tensors [2, NB, BS, NH, HS]
    CROSS_LAYER = auto()  # NB_NL_TWO_BS_NH_HS: single tensor [NB, NL, 2, BS, NH, HS]
    MLA = auto()  # NL_X_NB_BS_HS: L separate tensors [NB, BS, HS]


class Direction(Enum):
    """Transfer direction."""

    H2D = 0  # LMCache -> vLLM (host to device)
    D2H = 1  # vLLM -> LMCache (device to host)


DEFAULT_NUM_BLOCKS = 1000
DEFAULT_BLOCK_SIZE = 16
DEFAULT_NUM_MEMORY_OBJECTS = 4
DEFAULT_TOKENS_PER_OBJECT = 256
DEFAULT_NUM_WARMUP = 10
DEFAULT_NUM_BENCH = 100


@dataclass
class TestConfig:
    """Configuration for a single test case."""

    vllm_format: VLLMBufferFormat
    dtype: torch.dtype
    num_layers: int
    num_blocks: int
    block_size: int
    num_heads: int
    head_size: int
    num_memory_objects: int
    tokens_per_object: int
    num_warmup_iters: int
    num_bench_iters: int
    skip_prefix_n_blocks: int

    @property
    def hidden_dim(self) -> int:
        return self.num_heads * self.head_size

    @property
    def total_blocks(self) -> int:
        """Total number of block_ids = num_memory_objects * blocks_per_object."""
        return self.num_memory_objects * self.blocks_per_object

    @property
    def blocks_per_object(self) -> int:
        return self.tokens_per_object // self.block_size

    @property
    def is_mla(self) -> bool:
        return self.vllm_format == VLLMBufferFormat.MLA

    @property
    def kv_dim(self) -> int:
        """First dimension of LMCache memory object: 2 for non-MLA, 1 for MLA."""
        return 1 if self.is_mla else 2

    @property
    def name(self) -> str:
        dtype_name = "bf16" if self.dtype == torch.bfloat16 else "fp8"
        return f"{self.vllm_format.name.lower()}_{dtype_name}"


def get_all_test_configs(
    num_warmup: int = DEFAULT_NUM_WARMUP,
    num_bench: int = DEFAULT_NUM_BENCH,
    skip_prefix_n_blocks: int = 0,
) -> list:
    """Generate all 6 test configurations (3 formats x 2 dtypes)."""
    format_params = [
        # (format, num_layers, num_heads, head_size)
        (VLLMBufferFormat.NORMAL, 64, 8, 128),
        (VLLMBufferFormat.CROSS_LAYER, 64, 8, 128),
        (VLLMBufferFormat.MLA, 104, 1, 576),
    ]
    dtypes = [torch.bfloat16, torch.float8_e4m3fn]

    configs = []
    for fmt, nl, nh, hs in format_params:
        for dt in dtypes:
            configs.append(
                TestConfig(
                    vllm_format=fmt,
                    dtype=dt,
                    num_layers=nl,
                    num_blocks=DEFAULT_NUM_BLOCKS,
                    block_size=DEFAULT_BLOCK_SIZE,
                    num_heads=nh,
                    head_size=hs,
                    num_memory_objects=DEFAULT_NUM_MEMORY_OBJECTS,
                    tokens_per_object=DEFAULT_TOKENS_PER_OBJECT,
                    num_warmup_iters=num_warmup,
                    num_bench_iters=num_bench,
                    skip_prefix_n_blocks=skip_prefix_n_blocks,
                )
            )
    return configs


def filter_configs(
    configs: list,
    format_filter: str = "all",
    dtype_filter: str = "all",
) -> list:
    """Filter configs by format and dtype CLI arguments."""
    format_map = {
        "normal": VLLMBufferFormat.NORMAL,
        "cross_layer": VLLMBufferFormat.CROSS_LAYER,
        "mla": VLLMBufferFormat.MLA,
    }
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn,
    }

    result = configs
    if format_filter != "all":
        target_fmt = format_map[format_filter]
        result = [c for c in result if c.vllm_format == target_fmt]
    if dtype_filter != "all":
        target_dt = dtype_map[dtype_filter]
        result = [c for c in result if c.dtype == target_dt]
    return result
