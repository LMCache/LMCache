# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for the gRPC transport implementation."""

# Standard
from importlib import import_module
import sys

_impl = import_module("lmcache.v1.multiprocess.transport.grpc_impl.grpc")
sys.modules[__name__] = _impl
