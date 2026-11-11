# SPDX-License-Identifier: Apache-2.0
"""Ops-only binding surface for ``NpuDeviceOps.ensure_native``.

Mirrors upstream's ``lmcache.cuda_ops`` contract: ops plus
``PageBufferShapeDesc`` only. The compiled ``lmcache.npu_ops`` additionally
exports the format enums and ``is_*`` predicates that upstream keeps on
``lmcache_native`` alone. Binding either through ``bind_native`` would
violate the device-ops facade contract or shadow the class-level NPU
overrides, so both are filtered out here.
"""

# First Party
from lmcache.v1.platform import torch_ops as _torch_ops
import lmcache.npu_ops as _c_ops

# Names upstream keeps on lmcache_native only; never on the ops facade.
_FACADE_EXCLUDES = frozenset(
    (
        "TransferDirection",
        "EngineKVFormat",
        "GPUKVFormat",
        "is_cross_layer",
        "is_kv_list",
        "is_layer_list",
        "is_mla",
    )
)

for _name in dir(_c_ops):
    if _name.startswith("_") or _name in _FACADE_EXCLUDES:
        continue
    _symbol = getattr(_c_ops, _name)
    if not (callable(_symbol) or isinstance(_symbol, type)):
        continue
    if _symbol is getattr(_torch_ops, _name, None):
        continue  # merged torch fallback; class-level NPU overrides win
    globals()[_name] = _symbol
