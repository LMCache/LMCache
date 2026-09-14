# SPDX-License-Identifier: Apache-2.0
"""Generated protobuf and gRPC modules for the multiprocess transport.

The ``../protos/*.proto`` schemas are the source of truth. Package builds
generate ``*_pb2.py``, ``*_pb2.pyi``, and ``*_pb2_grpc.py`` modules in this
package. The type stubs used directly by custom adapters are tracked so static
analysis also works before a package build. Regenerate all bindings after
changing a schema with::

    pip install -r requirements/proto.txt
    python -m lmcache.v1.multiprocess.transport.grpc_impl._proto_gen._generate
"""
