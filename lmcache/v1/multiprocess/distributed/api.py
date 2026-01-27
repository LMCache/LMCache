# SPDX-License-Identifier: Apache-2.0
"""
Defines the data structures that will be used by the
distributed storage manager public functions

Could be implemented by native code in the future
"""

# Standard
from dataclasses import dataclass

# Third Party
import torch


@dataclass
class MemoryLayoutDesc:
    """
    Describes the layout of a memory object
    """

    shapes: list[torch.Size]
    dtypes: list[torch.dtype]

    def __post__init__(self):
        if len(self.shapes) != len(self.dtype):
            raise ValueError(
                "MemoryLayoutDesc: shapes and dtype must have the same length"
            )
