# SPDX-License-Identifier: Apache-2.0
"""Async GDS implementations, discovered without importing them.

Each public module's filename is its configuration name. It exports a concrete
``Backend(GDSBackend)`` class. Keep optional imports and native library loading
inside the implementation's methods, so default selection stays driver-free.
Private modules (names starting with ``_``) are shared helpers, not backends.
"""
