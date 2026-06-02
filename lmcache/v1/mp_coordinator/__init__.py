# SPDX-License-Identifier: Apache-2.0
"""MP coordinator for LMCache multi-process (mp) servers.

A standalone process that mp servers across nodes register with, so they can be
coordinated as a fleet: state reconcile (e.g. quota persistence and broadcast
on join), blend-lookup routing across model replicas, and KV-cache operations
(pin, prefetch, ...).

Those capabilities are implemented as controllers plugged into a shared
dispatch seam. The package currently ships the framework -- ZMQ transport, the
controller seam, an instance registry, and lifecycle hooks -- plus the
registration controller. Further controllers are added as new modules that plug
into the same seam without framework changes.
"""
