# SPDX-License-Identifier: Apache-2.0
"""Opt-in server modules shipped with LMCache.

Modules in this package are deliberately kept outside the built-in discovery
root (``lmcache.v1.multiprocess.modules``), so the server never composes them
on its own. An operator loads one by naming its Python module explicitly::

    lmcache server --server-module \\
        '{"module_path": "lmcache.v1.multiprocess.plugin_modules.<name>"}'

Each module follows the same ``DiscoverableModule`` contract as a built-in
(see ``lmcache.v1.multiprocess.engine_module``) and joins the same dependency
graph; see ``docs/source/developer_guide/extending_lmcache/server_modules.rst``.

Nothing under ``lmcache.v1.multiprocess.modules`` may import this package at
module scope: ``discover_modules()`` accepts re-exported classes, so such an
import would turn the module back into an always-built built-in.
"""
