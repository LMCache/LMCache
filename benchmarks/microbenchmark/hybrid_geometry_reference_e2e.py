# SPDX-License-Identifier: Apache-2.0
"""Run reproducible differential checks for hybrid KV-cache geometry."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from math import lcm
from typing import Any, cast
import argparse
import random
import time

# Third Party
import torch

# First Party
from lmcache.v1.kv_layer_groups import KernelGroupInfo
from lmcache.v1.multiprocess.group_view import slice_block_ids_per_group
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    downsample_and_stage_block_ids,
)
from lmcache.v1.platform.ops_types import PageBufferShapeDesc

# Local
from .hybrid_geometry_reference import (
    GroupGeometry,
    reference_block_ids_for_token_range,
    reference_num_physical_slots,
    reference_windowed_block_ids,
)

_BLOCK_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


@dataclass
class _Manager:
    geometries: list[GroupGeometry]
    chunk_tokens: int

    @property
    def num_kernel_groups(self) -> int:
        return len(self.geometries)

    def get_subchunk_sw_size_tokens(self, group_id: int) -> int:
        window = self.geometries[group_id].window_tokens
        return self.chunk_tokens if window is None else window


class _Context:
    def __init__(self, geometries: list[GroupGeometry], chunk_tokens: int):
        self.geometries = geometries
        self.lmcache_tokens_per_chunk = chunk_tokens
        self.kv_layer_groups_manager = _Manager(geometries, chunk_tokens)

    def calculate_num_blocks(self, num_tokens: int, group_id: int) -> int:
        return num_tokens // self.geometries[group_id].tokens_per_block

    def stage_block_ids(self, block_ids: list[list[int]]) -> list[list[int]]:
        return block_ids


def run_differential_checks(cases: int, seed: int) -> None:
    """Compare optimized runtime helpers against the reference model."""
    if cases <= 0:
        raise ValueError("cases must be positive")
    rng = random.Random(seed)
    started = time.monotonic()
    groups_checked = 0
    max_groups = 0

    for case_index in range(cases):
        group_count = rng.randint(1, 6)
        tokens_per_block = [rng.choice(_BLOCK_SIZES) for _ in range(group_count)]
        alignment = lcm(*tokens_per_block)
        start = rng.randint(0, 8) * alignment
        end = start + rng.randint(0, 8) * alignment
        geometries: list[GroupGeometry] = []
        allocated: dict[int, list[int]] = {}

        for group_id, block_size in enumerate(tokens_per_block):
            divisors = [
                candidate
                for candidate in _BLOCK_SIZES
                if candidate <= block_size and block_size % candidate == 0
            ]
            slots = rng.choice(divisors)
            possible_windows = [None] + list(
                range(block_size, 256 + block_size, block_size)
            )
            geometry = GroupGeometry(
                group_id=group_id,
                tokens_per_block=block_size,
                physical_slots_per_block=slots,
                window_tokens=rng.choice(possible_windows),
            )
            geometries.append(geometry)
            allocated[group_id] = [
                case_index * 10_000_000 + group_id * 1_000_000 + offset
                for offset in range(end // block_size + rng.randint(0, 4))
            ]
            logical_tokens = rng.randint(0, 128) * geometry.compression_ratio
            expected_slots = logical_tokens // geometry.compression_ratio
            actual_slots = reference_num_physical_slots(logical_tokens, geometry)
            shape_desc = PageBufferShapeDesc()
            shape_desc.bs = slots
            runtime_group = KernelGroupInfo(
                layer_indices=[group_id],
                shape_desc=shape_desc,
                dtype=torch.float16,
                tokens_per_block=block_size,
            )
            if (
                actual_slots != expected_slots
                or runtime_group.calculate_slots(logical_tokens) != actual_slots
            ):
                raise AssertionError(
                    f"physical-slot mismatch in case {case_index}, group {group_id}"
                )

        expected_slice = reference_block_ids_for_token_range(
            allocated,
            geometries,
            start,
            end,
        )
        actual_slice = slice_block_ids_per_group(
            allocated,
            tokens_per_block,
            start,
            end,
        )
        if actual_slice != [
            expected_slice[group_id] for group_id in range(group_count)
        ]:
            raise AssertionError(f"token-range mismatch in case {case_index}")

        num_chunks = rng.randint(0, 16)
        window_tables = {
            group_id: [
                case_index * 10_000_000 + group_id * 1_000_000 + offset
                for offset in range(num_chunks * 256 // geometry.tokens_per_block)
            ]
            for group_id, geometry in enumerate(geometries)
        }
        expected_window = reference_windowed_block_ids(
            window_tables,
            geometries,
            logical_chunk_tokens=256,
        )
        actual_window = downsample_and_stage_block_ids(
            cast(Any, _Context(geometries, chunk_tokens=256)),
            [list(window_tables[group_id]) for group_id in range(group_count)],
        )
        if actual_window != [
            expected_window[group_id] for group_id in range(group_count)
        ]:
            raise AssertionError(f"window mismatch in case {case_index}")

        groups_checked += group_count
        max_groups = max(max_groups, group_count)

    elapsed = time.monotonic() - started
    print(f"seed={seed}")
    print(f"cases={cases}")
    print(f"groups_checked={groups_checked}")
    print(f"max_groups_per_case={max_groups}")
    print(f"elapsed_seconds={elapsed:.6f}")
    print("token_range_differential=PASS")
    print("window_downsampling_differential=PASS")
    print("physical_slot_invariant=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    run_differential_checks(args.cases, args.seed)


if __name__ == "__main__":
    main()
