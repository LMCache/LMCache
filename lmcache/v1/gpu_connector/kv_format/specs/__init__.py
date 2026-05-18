# SPDX-License-Identifier: Apache-2.0
"""Leaf :class:`KVFormatSpec` modules.

Each module here declares one or more concrete formats. Modules are
discovered and imported lazily by
:func:`lmcache.v1.gpu_connector.kv_format.registry.ensure_loaded`;
adding a new format only requires creating a new file in this
directory — no other code needs to change.
"""
