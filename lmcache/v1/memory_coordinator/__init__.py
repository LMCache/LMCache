# SPDX-License-Identifier: Apache-2.0
"""Standalone shared Device-DAX Memory Coordinator (M0).

This package implements the strong allocation and object-lifetime authority
for one shared Device-DAX/CXL region. It is deliberately independent from the
MP Coordinator (`lmcache.v1.mp_coordinator`), which remains an
eventual-consistency fleet directory and management plane: the Memory
Coordinator sits on the serving hot path and owns strong state, and the two
must never be merged.

The M0 transport is a small FastAPI service (:mod:`.app`) plus an ``httpx``
client (:mod:`.client`). KV payload bytes
never traverse this package: requests and responses carry only keys, layout
descriptions, offsets, lengths, generations, epochs, and opaque tokens.
"""
