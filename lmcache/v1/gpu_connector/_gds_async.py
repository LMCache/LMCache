# SPDX-License-Identifier: Apache-2.0
"""Platform dispatch for the GDS async backend.

Re-exports the GDSContext-facing surface from the cuFile wrapper on NVIDIA
(:mod:`lmcache.v1.gpu_connector._cufile_async`) or the hipFile wrapper on AMD
ROCm (:mod:`lmcache.v1.gpu_connector._hipfile_async`). Both modules expose an
identical API -- :class:`AsyncHandle`, :class:`Submission`, and the
``register_*`` / ``deregister_*`` / ``close_driver`` functions -- so
:mod:`lmcache.v1.gpu_connector.gds_context` imports this shim as ``ca`` and is
platform-agnostic.

Selection is by ``torch.version.hip``: a ROCm torch build reports a non-None
HIP version. Importing this shim does not dlopen any GPU IO driver; both
backends bind ``libcufile``/``libhipfile`` lazily on first use.
"""

# Third Party
import torch

if torch.version.hip is not None:
    # First Party
    from lmcache.v1.gpu_connector._hipfile_async import (
        AsyncHandle as AsyncHandle,
        Submission as Submission,
        close_driver as close_driver,
        deregister_buffer as deregister_buffer,
        deregister_handle as deregister_handle,
        deregister_stream as deregister_stream,
        register_buffer as register_buffer,
        register_handle as register_handle,
        register_stream as register_stream,
    )
else:
    # First Party
    from lmcache.v1.gpu_connector._cufile_async import (
        AsyncHandle as AsyncHandle,
        Submission as Submission,
        close_driver as close_driver,
        deregister_buffer as deregister_buffer,
        deregister_handle as deregister_handle,
        deregister_stream as deregister_stream,
        register_buffer as register_buffer,
        register_handle as register_handle,
        register_stream as register_stream,
    )
