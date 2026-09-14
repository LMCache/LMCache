# SPDX-License-Identifier: Apache-2.0
"""Coordinator-owned shared Device-DAX L1 support for MP servers.

This package is the MP-server side of the shared Device-DAX M0 path: it
maps the host-local Device-DAX view of the shared region, talks to the
standalone Memory Coordinator (:mod:`lmcache.v1.memory_coordinator`) for
every allocation and lifetime decision, and applies the platform visibility
ABI around each exact payload range.
"""
