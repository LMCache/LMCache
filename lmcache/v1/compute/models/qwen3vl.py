# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.compute.models.base import LMCBaseModel


class LMCQwen3VLModel(LMCBaseModel):
    """
    LMCache Q/K/V preprocessing for Qwen3VL family (incl. Qwen3-VL).
    - Most Qwen3VL variants do NOT use per-head q_norm / k_norm like Qwen3.
    - For robustness, we probe self_attn.{q_norm,k_norm}; if present, apply them.
    - Otherwise, return q/k/v unchanged.

    This keeps behavior correct across sub-variants while minimizing branching in
    higher-level cache logic.
    """

    def _process_qkv(self, q, k, v, layer):
        """
        Args:
            q, k, v: [*, hidden_size] projections produced by the layer's QKV linear(s).
            layer:  The transformer block/module that owns self_attn.

        Returns:
            (q, k, v) possibly normalized per-head if the layer exposes q_norm/k_norm.
        """
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            # Fallback: nothing to do
            return q, k, v

        head_dim = getattr(attn, "head_dim", None)
        q_norm = getattr(attn, "q_norm", None)
        k_norm = getattr(attn, "k_norm", None)

        # If Qwen3-VL variant defines q_norm/k_norm, apply them per-head.
        if head_dim and (q_norm is not None or k_norm is not None):
            def _maybe_norm(x, norm):
                if norm is None:
                    return x
                # reshape to [*, n_heads, head_dim], apply norm per head, then flatten back
                x_by_head = x.view(*x.shape[:-1], x.shape[-1] // head_dim, head_dim)
                x_by_head = norm(x_by_head)
                return x_by_head.view(*x.shape)

            q = _maybe_norm(q, q_norm)
            k = _maybe_norm(k, k_norm)

        return q, k, v
