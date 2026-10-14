# SPDX-License-Identifier: Apache-2.0
"""Async GDS implementations, discovered without importing them.

``base.py`` defines the shared interfaces. Each other public module exports a
concrete ``Backend(GDSBackend)`` class; its filename is its configuration name.
Keep optional imports and native library loading inside implementation methods,
so default selection stays driver-free. Private modules are shared helpers.
"""
