# SPDX-License-Identifier: Apache-2.0
"""RBLN (Rebellions NPU) platform primitives.

Registers :class:`RblnDeviceSpec` with the device-detection registry so
LMCache resolves ``torch.rbln`` as an accelerator instead of falling back
to the CPU stub.  ``torch.rbln`` is contributed by the ``torch_rbln``
package through a torch backend entry point, so it is visible on a bare
``import torch`` -- no explicit ``import torch_rbln`` is required here.

Scope: **engine-driven** multiprocess (MP) transfer only. ``torch.rbln``
provides ``Stream`` / ``Event`` types for in-process ordering. LMCache has no
RBLN tensor IPC wrapper or event IPC backend, which the LMCache-driven path
needs to publish KV buffers across processes.  The spec therefore reports
:meth:`RblnDeviceSpec.is_lmcache_driven_available` as ``False`` and leaves
``ipc_wrapper_cls`` / ``event_ipc_backend`` at their ``None`` defaults, so
requesting ``mp_transfer_mode=lmcache_driven`` fails fast with a clear
error rather than crashing deeper in the transfer path.

See ``docs/design/v1/platform/devices/rbln/README.md`` for the full contract.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

# First Party
from lmcache.utils import EngineType
from lmcache.v1.platform.base.device_spec import DeviceSpec

if TYPE_CHECKING:
    # Third Party
    import torch

    # First Party
    from lmcache.v1.platform.base.device_ops import DeviceOps
    from lmcache.v1.platform.base.pin_memory import PinMemoryBackend


def warm_up_kv_transfer(
    kv_caches: Mapping[str, torch.Tensor], engine_type: EngineType = EngineType.VLLM
) -> None:
    """Prepare transfers of an engine's KV caches, building what they need here.

    For engines driving LMCache's engine-driven MP connector on RBLN. Call it
    once, on the thread that runs the model, after the KV caches handed to
    ``register_kv_caches`` are final and before the first request; vLLM-RBLN
    does so after its model warm-up, before it sizes its KV cache. Every
    cache layout gets its device staging allocated here, and HND caches their
    swap programs built and kept (see
    :meth:`RblnDeviceOps.prepare_kv_transfer`), so transfers never allocate
    staging or build a program. Without it, every transfer fails before
    copying: stores are dropped and restores miss. Calling it again for the
    same caches does nothing.

    Caches are grouped by shape and dtype, so each group has one layout, and
    each group is prepared as one KV cache group the way a single-group
    engine transfers it. Caches of one shape that LMCache splits into several
    groups (a hybrid model's groups of one layout) are not prepared per group.

    Args:
        kv_caches: Layer name to KV tensor, as passed to ``register_kv_caches``.
        engine_type: The engine whose layout the caches use, as passed to the
            engine-driven registration.

    Raises:
        ValueError: If a group's layout is one RBLN cannot transfer; its
            transfers would fail the same way.
        RuntimeError: If allocating staging or building a swap fails,
            including for HND caches that are not bf16 with a head size a
            multiple of 64, which torch-rbln builds no swap program for.
    """
    # Imported here, not at module scope: ``lmcache.v1.common.utils`` reads
    # ``lmcache.device_ops``, which device detection is still resolving
    # through this module while ``lmcache`` initializes on an RBLN host.
    # First Party
    from lmcache.v1.common.utils import normalize_kv_and_discover_format
    from lmcache.v1.platform import resolve_device_ops
    from lmcache.v1.platform.devices.rbln.device_ops import RblnDeviceOps

    groups: dict[tuple[tuple[int, ...], torch.dtype], list[torch.Tensor]] = {}
    for tensor in kv_caches.values():
        groups.setdefault((tuple(tensor.shape), tensor.dtype), []).append(tensor)
    ops = cast("RblnDeviceOps", resolve_device_ops("rbln"))
    for tensors in groups.values():
        engine_kv_format, normalized = normalize_kv_and_discover_format(
            tensors, engine_type
        )
        ops.prepare_kv_transfer(normalized, engine_kv_format)


# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class RblnDeviceSpec(DeviceSpec):
    """RBLN device specification for the detection registry."""

    @property
    def device_type(self) -> str:
        return "rbln"

    @property
    def torch_module_name(self) -> str:
        return "rbln"

    @property
    def ops_cls(self) -> type[DeviceOps]:
        # First Party
        from lmcache.v1.platform.devices.rbln.device_ops import RblnDeviceOps

        return RblnDeviceOps

    @property
    def is_pin_supported(self) -> bool:
        """Decline LMCache allocator-side pinning on RBLN.

        RBLN uses engine-driven MP. The worker performs the copies and
        registers its own SHM mapping through pin_memory. The server keeps
        an eager SHM L1 pool and leaves Device-DAX arenas unregistered.
        """
        return False

    @property
    def pin_memory_backend(self) -> type[PinMemoryBackend]:
        """Return the torch-rbln host registration backend."""
        # First Party
        from lmcache.v1.platform.devices.rbln.pin_memory import RblnPinMemoryBackend

        return RblnPinMemoryBackend

    def is_available(self) -> bool:
        """Check RBLN availability without importing ``lmcache.__init__``.

        ``torch.rbln.is_available()`` raises (rather than returning
        ``False``) when the runtime cannot register a physical NPU -- for
        example when another process already holds the device or a stale
        allocation survives.  Detection runs on every LMCache start,
        including on hosts with no free NPU, so the exception is swallowed
        and reported as "unavailable"; letting it escape would abort import
        for every co-tenant process on the box.

        Returns:
            bool: ``True`` when ``torch.rbln`` is present and reports at
            least one usable device, ``False`` otherwise.
        """
        try:
            # Third Party
            import torch

            return hasattr(torch, "rbln") and torch.rbln.is_available()
        except Exception:
            return False

    def is_lmcache_driven_available(self) -> bool:
        """Report that RBLN cannot ship KV tensors as IPC handles.

        LMCache has no RBLN event IPC backend or tensor IPC wrapper for the
        ordered cross-process publication this path depends on.  Returning
        ``False`` keeps
        ``mp_transfer_mode=lmcache_driven`` failing at its documented
        validation point instead of at an attribute lookup later on.

        Returns:
            bool: Always ``False``.
        """
        return False
