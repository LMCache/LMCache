# SPDX-License-Identifier: Apache-2.0
"""Defines the data structures shared by the client and server APIs."""

# Standard
from dataclasses import dataclass

# Third Party
import torch


@dataclass(frozen=True)
class MemoryLayoutDesc:
    """
    Describes the layout of a memory object
    """

    shapes: list[torch.Size]
    dtypes: list[torch.dtype]

    def __post_init__(self):
        if len(self.shapes) != len(self.dtypes):
            raise ValueError(
                "MemoryLayoutDesc: shapes and dtype must have the same length"
            )
