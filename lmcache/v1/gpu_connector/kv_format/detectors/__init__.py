# SPDX-License-Identifier: Apache-2.0
"""Per-engine :class:`EngineDetector` modules.

Each module declares one detector. Modules are discovered and imported
lazily by
:func:`lmcache.v1.gpu_connector.kv_format.registry.ensure_loaded`;
adding support for a new engine only requires creating a new file in
this directory — no other code needs to change.
"""
