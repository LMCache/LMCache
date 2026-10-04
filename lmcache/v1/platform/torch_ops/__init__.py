# SPDX-License-Identifier: Apache-2.0
"""Device-agnostic torch baseline for the unified ``DeviceOps`` surface.

Migrated verbatim from the former ``lmcache.python_ops_fallback`` module.
Owns the torch/CPU implementation of every device op. Internal to the platform
package -- consumers go through :class:`DeviceOps`, never this module directly.
"""

# First Party
from lmcache.v1.platform.torch_ops._kv_format import *
from lmcache.v1.platform.torch_ops._tensor_from_ptr import *
from lmcache.v1.platform.torch_ops.cachegen_kernels import *
from lmcache.v1.platform.torch_ops.completion_recorder import *
from lmcache.v1.platform.torch_ops.event_recorder import *
from lmcache.v1.platform.torch_ops.mem_alloc import *
from lmcache.v1.platform.torch_ops.mem_kernels import *
from lmcache.v1.platform.torch_ops.mp_mem_kernels import *
from lmcache.v1.platform.torch_ops.pos_kernels import *
from lmcache.v1.platform.torch_ops.utils import *
