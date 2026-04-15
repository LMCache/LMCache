# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.compute.models.base import LMCBaseModel


class LMCLlamaModel(LMCBaseModel):
    def _process_qkv(self, q, k, v, layer):
        """Process QKV tensors for LLaMa model (no additional processing)."""
        return q, k, v
