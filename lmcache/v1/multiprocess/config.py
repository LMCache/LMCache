# SPDX-License-Identifier: Apache-2.0

"""
Configuration for the multiprocess (ZMQ) server and HTTP frontend.
"""

# Standard
from dataclasses import dataclass
import argparse


@dataclass
class MPServerConfig:
    """Configuration for the ZMQ-based multiprocess cache server."""

    host: str = "localhost"
    """ZMQ server host."""

    port: int = 5555
    """ZMQ server port."""

    chunk_size: int = 256
    """Chunk size for KV cache operations."""

    max_workers: int = 1
    """Maximum number of worker threads for ZMQ server."""

    hash_algorithm: str = "blake3"
    """Hash algorithm for token-based operations (builtin, sha256_cbor, blake3)."""


DEFAULT_MP_SERVER_CONFIG = MPServerConfig()


@dataclass
class HTTPFrontendConfig:
    """Configuration for the HTTP frontend (uvicorn/FastAPI)."""

    http_host: str = "0.0.0.0"
    """HTTP server host."""

    http_port: int = 8000
    """HTTP server port."""


DEFAULT_HTTP_FRONTEND_CONFIG = HTTPFrontendConfig()


def add_mp_server_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """
    Add MP server configuration arguments to an existing parser.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        The same parser with MP server arguments added.
    """
    mp_group = parser.add_argument_group(
        "MP Server", "Configuration for the ZMQ multiprocess cache server"
    )
    mp_group.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the ZMQ server. Default is localhost.",
    )
    mp_group.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Port to bind the ZMQ server. Default is 5555.",
    )
    mp_group.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size for KV cache operations. Default is 256.",
    )
    mp_group.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of worker threads. Default is 1.",
    )
    mp_group.add_argument(
        "--hash-algorithm",
        type=str,
        default="blake3",
        help="Hash algorithm for token-based operations "
        "(builtin, sha256_cbor, blake3). Default is blake3.",
    )
    return parser


def parse_args_to_mp_server_config(
    args: argparse.Namespace,
) -> MPServerConfig:
    """
    Convert parsed command line arguments to an MPServerConfig.

    Args:
        args: Parsed arguments from the argument parser.

    Returns:
        MPServerConfig: The configuration object.
    """
    return MPServerConfig(
        host=args.host,
        port=args.port,
        chunk_size=args.chunk_size,
        max_workers=args.max_workers,
        hash_algorithm=args.hash_algorithm,
    )


def add_http_frontend_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """
    Add HTTP frontend configuration arguments to an existing parser.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        The same parser with HTTP frontend arguments added.
    """
    http_group = parser.add_argument_group(
        "HTTP Frontend", "Configuration for the HTTP frontend server"
    )
    http_group.add_argument(
        "--http-host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the HTTP server. Default is 0.0.0.0.",
    )
    http_group.add_argument(
        "--http-port",
        type=int,
        default=8000,
        help="Port to bind the HTTP server. Default is 8000.",
    )
    return parser


def parse_args_to_http_frontend_config(
    args: argparse.Namespace,
) -> HTTPFrontendConfig:
    """
    Convert parsed command line arguments to an HTTPFrontendConfig.

    Args:
        args: Parsed arguments from the argument parser.

    Returns:
        HTTPFrontendConfig: The configuration object.
    """
    return HTTPFrontendConfig(
        http_host=args.http_host,
        http_port=args.http_port,
    )
