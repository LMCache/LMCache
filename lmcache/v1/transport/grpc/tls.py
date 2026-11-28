# SPDX-License-Identifier: Apache-2.0
"""File-based TLS credentials for the multiprocess gRPC transport."""

# Standard
from dataclasses import dataclass
from pathlib import Path

# Third Party
import grpc


def _read_pem(path: str | Path, field_name: str) -> bytes:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read gRPC TLS {field_name}: {path}") from exc
    if not data.strip():
        raise ValueError(f"gRPC TLS {field_name} is empty: {path}")
    return data


@dataclass(frozen=True)
class GrpcServerTLSConfig:
    """Enable TLS for a Python gRPC server with PEM credentials.

    Args:
        cert_file: Server certificate chain.
        key_file: Server private key.
        ca_file: Optional CA bundle for verifying client certificates.
        require_client_cert: Require a certificate from each client (mTLS).

    Raises:
        ValueError: If the server certificate or key is unspecified, or mTLS
            is requested without a CA bundle.
    """

    cert_file: str | Path = ""
    key_file: str | Path = ""
    ca_file: str | Path | None = None
    require_client_cert: bool = False

    def __post_init__(self) -> None:
        if not self.cert_file or not self.key_file:
            raise ValueError("gRPC server TLS requires both cert_file and key_file")
        if self.require_client_cert and not self.ca_file:
            raise ValueError("gRPC server TLS requires ca_file for require_client_cert")

    def server_credentials(self) -> grpc.ServerCredentials:
        """Load the configured PEM files as gRPC server credentials.

        Returns:
            Credentials for binding a secure gRPC port.

        Raises:
            ValueError: If a configured file is unreadable or empty.
        """
        key = _read_pem(self.key_file, "key_file")
        cert = _read_pem(self.cert_file, "cert_file")
        roots = _read_pem(self.ca_file, "ca_file") if self.ca_file is not None else None
        return grpc.ssl_server_credentials(
            [(key, cert)],
            root_certificates=roots,
            require_client_auth=self.require_client_cert,
        )


@dataclass(frozen=True)
class GrpcClientTLSConfig:
    """Configure TLS for a multiprocess gRPC client.

    Args:
        enabled: Use a secure channel when true.
        ca_file: Optional PEM CA bundle. Defaults to gRPC's trusted roots.
        cert_file: Optional PEM client certificate chain for mTLS.
        key_file: Optional PEM client private key for mTLS.
        server_name_override: Name to verify in place of the dialed host.

    Raises:
        ValueError: If only one client identity file is specified, a supplied
            path or name is empty, or TLS options are supplied while disabled.
    """

    enabled: bool = False
    ca_file: str | Path | None = None
    cert_file: str | Path | None = None
    key_file: str | Path | None = None
    server_name_override: str | None = None

    def __post_init__(self) -> None:
        if (self.cert_file is None) != (self.key_file is None):
            raise ValueError("gRPC client TLS requires both cert_file and key_file")
        options = (
            self.ca_file,
            self.cert_file,
            self.key_file,
            self.server_name_override,
        )
        if any(option == "" for option in options):
            raise ValueError(
                "gRPC client TLS file paths and server name must not be empty"
            )
        if not self.enabled and any(option is not None for option in options):
            raise ValueError("gRPC client TLS options require enabled=True")

    def channel_credentials(self) -> grpc.ChannelCredentials:
        """Load the configured PEM files as gRPC channel credentials.

        Returns:
            Credentials for opening a TLS channel.

        Raises:
            ValueError: If a configured file is unreadable or empty.
        """
        roots = _read_pem(self.ca_file, "ca_file") if self.ca_file is not None else None
        key = (
            _read_pem(self.key_file, "key_file") if self.key_file is not None else None
        )
        cert = (
            _read_pem(self.cert_file, "cert_file")
            if self.cert_file is not None
            else None
        )
        return grpc.ssl_channel_credentials(
            root_certificates=roots, private_key=key, certificate_chain=cert
        )
