# SPDX-License-Identifier: Apache-2.0
"""Metadata shared by the SGLang in-process and multiprocess connectors."""

# Standard
from dataclasses import dataclass

# Third Party
import torch


@dataclass
class StoreMetadata:
    last_node: object
    token_ids: list[int]
    kv_indices: torch.Tensor
    offset: int
    request_id: str = ""


@dataclass
class LoadMetadata:
    token_ids: list[int]
    slot_mapping: torch.Tensor
    offset: int
    prefix_pad: int = 0
    request_id: str = ""
