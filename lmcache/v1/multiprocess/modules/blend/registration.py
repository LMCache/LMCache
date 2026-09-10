# SPDX-License-Identifier: Apache-2.0
"""Blend registration: rope state attach/detach (STORE hook: store.py)."""

# Standard
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Standard
    import weakref

    # First Party
    from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
        LMCacheDrivenTransferModule,
    )

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import DeviceIPCWrapper
from lmcache.v1.multiprocess.modules.blend.rope import _CBRopeState

logger = init_logger(__name__)


# Default for the wire-typed group_rot parameter; never mutated.
_EMPTY_GROUP_ROT: list[list[int]] = []


class RegistrationMixin:
    """Rope registration handlers of ``BlendModule``; state lives on the
    composed instance."""

    if TYPE_CHECKING:
        # State owned by BlendModule.__init__; declared so the mixin type-checks.
        _transfer_module: "LMCacheDrivenTransferModule"
        _cb_rope_state: dict[int, _CBRopeState]
        _cb_plan_invariants: "weakref.WeakKeyDictionary[Any, tuple]"

        def _cb_slot_buffers(
            self, gpu_context: Any, num_groups: int, n_pos: int
        ) -> Any: ...

        def _resolve_cb_plan_invariants(
            self, gpu_context: Any, rope_state: _CBRopeState, max_batch: int
        ) -> Any: ...

    def cb_register_rope(
        self,
        instance_id: int,
        cos_sin_caches_ipc: list[DeviceIPCWrapper],
        head_size: int,
        is_neox_style: bool,
        group_to_cache: list[int],
        # Annotation must equal the protocol payload class exactly (mq.py
        # same_type check); direct callers may still pass tuples/None entries.
        group_rot: list[list[int]] = _EMPTY_GROUP_ROT,
    ) -> None:
        """Attach CB re-RoPE state to a registered KV-cache instance.

        Idempotent; requires a prior ``REGISTER_KV_CACHE``. Strips any baked-in
        YaRN/longrope mscale so re-RoPE stays a pure rotation.

        Args:
            instance_id: KV-cache instance to attach rope state to.
            cos_sin_caches_ipc: IPC handles to vLLM's cos/sin rope cache(s),
                one per distinct rope.
            head_size: Rotary head dimension.
            is_neox_style: True for NeoX (contiguous halves), else GPT-J.
            group_to_cache: Per-engine-group index into the caches list;
                empty means every group uses cache 0.
            group_rot: Per-engine-group rotation window ``(offset_elems,
                width_elems)``, or ``None`` per entry to skip that group.
                Empty/omitted = legacy inference (rotate ``head_size`` dims at
                offset 0). MLA models must declare this: a single-plane MLA
                row is indistinguishable from a key-only cache to the legacy
                inference and would get its content dims rotated.

        Raises:
            ValueError: On a missing KV cache, bad ``group_to_cache``
                coverage, or a malformed ``group_rot`` entry.
        """
        entry = self._transfer_module.get_and_touch_context_entry(instance_id)
        if entry is None:
            raise ValueError(
                f"Instance {instance_id} has no paged KV cache registered; "
                "send REGISTER_KV_CACHE before CB_REGISTER_ROPE."
            )
        # Zero caches is legal (NoPE): rope state still carries scatter
        # geometry; every re-RoPE consumer skips.
        if group_to_cache:
            if min(group_to_cache) < 0 or max(group_to_cache) >= len(
                cos_sin_caches_ipc
            ):
                raise ValueError(
                    f"group_to_cache {group_to_cache} contains indices outside "
                    f"[0, {len(cos_sin_caches_ipc)}) for the sent cache(s)."
                )
            # Every engine group needs a mapping; fail here, not mid-retrieve.
            max_eg_idx = max(
                (
                    g.engine_group_idx
                    for g in entry.cache_context.kv_layer_groups_manager.kernel_groups
                ),
                default=-1,
            )
            if len(group_to_cache) <= max_eg_idx:
                raise ValueError(
                    f"group_to_cache covers {len(group_to_cache)} engine "
                    f"group(s) but the registered model has engine groups up "
                    f"to index {max_eg_idx}."
                )

        # Normalize rope windows (wire turns tuples into lists); validate now
        # so a bad registration fails loudly instead of mid-retrieve.
        norm_rot: "list[tuple[int, int] | None]" = []
        for eg_idx, rot_entry in enumerate(group_rot or []):
            if rot_entry is None or len(rot_entry) == 0:
                # None (direct call) / [] (wire encoding): skip this group.
                norm_rot.append(None)
                continue
            if len(rot_entry) != 2 or int(rot_entry[0]) < 0 or int(rot_entry[1]) <= 0:
                raise ValueError(
                    f"group_rot[{eg_idx}] = {rot_entry!r}: expected "
                    "(offset >= 0, width > 0) or None."
                )
            norm_rot.append((int(rot_entry[0]), int(rot_entry[1])))

        cos_sin_caches: list[torch.Tensor] = []
        for cache_idx, cache_ipc in enumerate(cos_sin_caches_ipc):
            cos_sin_cache = cache_ipc.to_tensor()
            # YaRN/longrope bake an mscale m into the cache (cos²+sin²=m²≠1);
            # CB re-RoPE assumes a pure rotation, so an un-normalized m
            # injects an m² error per K element.
            _c32 = cos_sin_cache.to(torch.float32)
            _half = _c32.shape[1] // 2
            _m = float((_c32[:, :_half] ** 2 + _c32[:, _half:] ** 2).mean().sqrt())
            if abs(_m - 1.0) >= 1e-3:
                logger.info(
                    "CB re-RoPE: cache %d: stripping rope-cache mscale=%.4f "
                    "(m²=%.4f → K inflation if uncorrected) → unit magnitude",
                    cache_idx,
                    _m,
                    _m * _m,
                )
                cos_sin_cache = (_c32 / _m).to(cos_sin_cache.dtype)
            cos_sin_caches.append(cos_sin_cache)

        self._cb_rope_state[instance_id] = _CBRopeState(
            head_size=head_size,
            is_neox_style=is_neox_style,
            cos_sin_caches=cos_sin_caches,
            group_to_cache=list(group_to_cache),
            group_rot=norm_rot,
        )

        logger.info(
            "Registered CB rope state for instance %d "
            "(%d cache(s), shapes=%s dtype=%s, head_size=%d, is_neox=%s, "
            "group_map=%s, group_rot=%s)",
            instance_id,
            len(cos_sin_caches),
            [tuple(c.shape) for c in cos_sin_caches],
            cos_sin_caches[0].dtype if cos_sin_caches else "n/a (NoPE)",
            head_size,
            is_neox_style,
            "uniform" if not group_to_cache else str(group_to_cache),
            "legacy" if not norm_rot else str(norm_rot),
        )

        # Pre-warm plan invariants + slot staging off the retrieve critical
        # path.
        try:
            entry = self._transfer_module.get_and_touch_context_entry(instance_id)
            ctx = entry.cache_context if entry is not None else None
            if ctx is not None:
                self._cb_slot_buffers(
                    ctx, ctx.kv_layer_groups_manager.num_kernel_groups, 1 << 16
                )
                rope_state = self._cb_rope_state[instance_id]
                max_batch = ctx.max_batch_size
                if self._cb_plan_invariants.get(ctx) is None:
                    resolved = self._resolve_cb_plan_invariants(
                        ctx, rope_state, max_batch
                    )
                    if resolved is not None:
                        self._cb_plan_invariants[ctx] = (
                            rope_state,
                            max_batch,
                            resolved,
                        )
        except Exception:
            logger.debug("CB plan pre-warm skipped", exc_info=True)

    def cb_unregister_rope(self, instance_id: int) -> None:
        """Drop the instance's CB rope state; the paged KV cache stays intact."""
        self._cb_rope_state.pop(instance_id, None)
        if self._transfer_module.get_and_touch_context_entry(instance_id) is None:
            logger.warning(
                "cb_unregister_rope: instance %d not registered", instance_id
            )
            return
        logger.info("Unregistered CB rope state for instance %d", instance_id)
