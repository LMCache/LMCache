# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for shared-transport protobuf generation."""

# First Party
from lmcache.v1.transport.grpc._proto_gen._generate import generate as generate
from lmcache.v1.transport.grpc._proto_gen._generate import main

if __name__ == "__main__":
    main()
