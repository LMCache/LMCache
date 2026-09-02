# SPDX-License-Identifier: Apache-2.0
"""Generate Python gRPC bindings from the service proto files."""

# Standard
from pathlib import Path

# Third Party
from grpc_tools import protoc


def generate() -> None:
    """Generate protobuf and gRPC Python modules next to their proto sources.

    Raises:
        RuntimeError: If ``grpc_tools.protoc`` cannot compile the proto files.
    """
    root = Path(__file__).resolve().parents[6]
    proto_dir = Path(__file__).resolve().parent
    proto_files = sorted(
        str(path.relative_to(root)) for path in proto_dir.glob("*.proto")
    )
    for pattern in ("*_pb2.py", "*_pb2_grpc.py"):
        for generated_file in proto_dir.glob(pattern):
            generated_file.unlink()
    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{root}",
            f"--python_out={root}",
            f"--grpc_python_out={root}",
            *proto_files,
        ]
    )
    if result != 0:
        raise RuntimeError(f"grpc_tools.protoc failed with exit code {result}")


if __name__ == "__main__":
    generate()
