# SPDX-License-Identifier: Apache-2.0
"""Layout of a memory object, independent of either endpoint."""

# Standard
from dataclasses import dataclass

# Third Party
import torch  # The gRPC codec resolves these annotations at runtime.


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
