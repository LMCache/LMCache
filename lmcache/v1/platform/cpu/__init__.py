# SPDX-License-Identifier: Apache-2.0
"""CPU-specific platform primitives.

:class:`~lmcache.v1.platform.cpu.shm.CpuShmTensorWrapper` carries a
``device_type`` ClassVar and a ``wrap`` factory classmethod, which the
universal registry picks up at run-time.
"""
