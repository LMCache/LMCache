# SPDX-License-Identifier: Apache-2.0
"""GDS L1 platform check + throughput tool.

Runnable via ``lmcache tool gds-check``. Reports whether the host
actually supports the GDS fast path (nvidia-fs loaded, fstype
supported, cuFile alignment satisfied), verifies a byte-for-byte
round-trip through the same code path as the GDS L1 backend, and
measures store / retrieve throughput so operators can compare
hardware before turning on ``--gds-l1-path`` in production.
"""
