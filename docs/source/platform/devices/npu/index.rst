NPU
===

Multiprocess event IPC
----------------------

The ``lmcache_driven`` transfer mode requires NPU interprocess events. The
presence of ``torch.npu.Event`` and ``Event.from_ipc_handle`` alone does not
guarantee that the installed torch_npu, CANN and driver support them.

During KV-cache registration, LMCache checks the event API and attempts to
construct ``Event(interprocess=True)`` on the target device. If construction
fails, registration raises ``RuntimeError`` with the target device and the
original runtime error. No ordinary-event fallback is used.

This capability check does not record or export the temporary event, or verify
cross-process handle import and synchronization.
