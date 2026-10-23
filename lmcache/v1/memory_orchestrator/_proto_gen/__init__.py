# SPDX-License-Identifier: Apache-2.0
"""Generated protobuf and gRPC modules for the memory orchestrator.

``../protos/memory_orchestrator.proto`` is the source of truth. Package builds
generate ``memory_orchestrator_pb2.py``, ``memory_orchestrator_pb2.pyi``, and
``memory_orchestrator_pb2_grpc.py`` in this package. The message stub is
tracked so static analysis of the servicer and client also works before a
package build. The schema imports the existing P2P key and layout types;
regenerate both packages in dependency order with::

    pip install -r requirements/proto.txt
    python -m lmcache.v1.multiprocess.transport.grpc_impl._proto_gen._generate
"""
