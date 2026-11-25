# SPDX-License-Identifier: Apache-2.0
"""Compatibility import for shared outbound gRPC authentication."""

# First Party
from lmcache.v1.transport.grpc.auth import (
    GrpcClientAuthenticationInterceptor as GrpcClientAuthenticationInterceptor,
)
