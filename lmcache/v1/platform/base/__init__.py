# SPDX-License-Identifier: Apache-2.0
"""Platform abstraction base classes.

This sub-package groups the abstract base classes shared by every
accelerator backend (``cuda``, ``cpu``, ``musa``, ``hpu``, ``xpu``,
...). Concrete backends subclass these types in built-in sub-packages or
external device-plugin wheels; discovery lives in :mod:`lmcache.v1.platform`.
"""
