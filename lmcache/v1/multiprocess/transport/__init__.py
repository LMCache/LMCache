# SPDX-License-Identifier: Apache-2.0
"""Transport layer for LMCache mp-mode gRPC.

Only the gRPC generated stubs live under this package now.  The
client/server runtime is implemented directly on top of those stubs in
``lmcache.v1.multiprocess.mq``; there is no protocol-agnostic
transport abstraction anymore.
"""
