# SPDX-License-Identifier: Apache-2.0
"""CPU loopback tests for TLS on the multiprocess gRPC transport."""

# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, closing, contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Literal, cast
import sys
import threading

# Third Party
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
import grpc
import pytest

# First Party
from lmcache.v1.client.transport.grpc_impl.client import GrpcMultiprocessClient
from lmcache.v1.common.custom_types import KVEventBatch
from lmcache.v1.common.security.api_key import (
    API_KEY_ENV_VAR,
    API_KEY_FILE_ENV_VAR,
    LEGACY_HTTP_API_KEY_ENV_VAR,
    LEGACY_HTTP_API_KEY_FILE_ENV_VAR,
)
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.engine_module import EngineModule
from lmcache.v1.multiprocess.request_handler import request_handler
from lmcache.v1.multiprocess.transport.grpc_impl.server_builder import (
    build_grpc_request_server,
)
from lmcache.v1.transport.errors import MPRequestError
from lmcache.v1.transport.futures import MessagingStream
from lmcache.v1.transport.grpc.tls import GrpcClientTLSConfig, GrpcServerTLSConfig
from lmcache.v1.transport.request_client import RequestClient

_BATCH = KVEventBatch(True, 1, 0, False, [])
_SERVER_NAME = "localhost"


@dataclass(frozen=True)
class _Certificates:
    server: GrpcServerTLSConfig
    client: GrpcClientTLSConfig
    untrusted_ca: Path
    client_cert: Path
    client_key: Path


class _TestModule:
    @request_handler()
    def noop(self) -> str:
        """Return a fixed unary response."""
        return "ok"

    @request_handler()
    def subscribe_kv_events(
        self, instance_id: int, model_name: str, cursor: int, max_events: int
    ) -> MessagingStream[KVEventBatch]:
        """Return a finite stream containing one event batch."""
        del instance_id, model_name, cursor, max_events
        batches = iter([_BATCH])

        def read(closed: threading.Event) -> KVEventBatch:
            del closed
            return next(batches)

        return MessagingStream(read, lambda: None)


def _certificate_builder(
    name: str, key: rsa.RSAPrivateKey, issuer: x509.Name
) -> x509.CertificateBuilder:
    """Build the shared subject, key and validity fields for a test certificate."""
    now = datetime.now(UTC)
    return (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=1))
    )


def _create_ca(name: str) -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
    """Create a self-signed test CA and its signing key."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    cert = (
        _certificate_builder(name, key, subject)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(key, hashes.SHA256())
    )
    return cert, key


def _write_identity(
    path: Path,
    name: str,
    ca: x509.Certificate,
    ca_key: rsa.RSAPrivateKey,
    usage: x509.ObjectIdentifier,
) -> tuple[Path, Path]:
    """Sign an endpoint identity and write its certificate and key as PEM."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    builder = _certificate_builder(name, key, ca.subject).add_extension(
        x509.ExtendedKeyUsage([usage]), critical=False
    )
    if usage == ExtendedKeyUsageOID.SERVER_AUTH:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName(name)]), critical=False
        )
    certificate = builder.sign(ca_key, hashes.SHA256())
    cert_path = path / f"{name}.crt"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path = path / f"{name}.key"
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


@pytest.fixture(scope="module")
def certificates(tmp_path_factory: pytest.TempPathFactory) -> _Certificates:
    """Generate one trusted CA, one unrelated CA, and two signed identities."""
    path = tmp_path_factory.mktemp("grpc-tls")
    ca, ca_key = _create_ca("LMCache test CA")
    other_ca, _ = _create_ca("LMCache unrelated test CA")
    ca_path = path / "ca.crt"
    ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    untrusted_ca = path / "untrusted.crt"
    untrusted_ca.write_bytes(other_ca.public_bytes(serialization.Encoding.PEM))
    server_cert, server_key = _write_identity(
        path, _SERVER_NAME, ca, ca_key, ExtendedKeyUsageOID.SERVER_AUTH
    )
    client_cert, client_key = _write_identity(
        path, "worker", ca, ca_key, ExtendedKeyUsageOID.CLIENT_AUTH
    )
    return _Certificates(
        GrpcServerTLSConfig(cert_file=server_cert, key_file=server_key),
        GrpcClientTLSConfig(enabled=True, ca_file=ca_path),
        untrusted_ca,
        client_cert,
        client_key,
    )


@pytest.fixture(params=["sync", "aio"])
def server_mode(request: pytest.FixtureRequest) -> Literal["sync", "aio"]:
    """Exercise TLS on each Python gRPC server implementation."""
    return request.param


@pytest.fixture(autouse=True)
def clear_api_key_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use only the request API-key configuration specified in each test."""
    for variable in (
        API_KEY_ENV_VAR,
        API_KEY_FILE_ENV_VAR,
        LEGACY_HTTP_API_KEY_ENV_VAR,
        LEGACY_HTTP_API_KEY_FILE_ENV_VAR,
    ):
        monkeypatch.delenv(variable, raising=False)


@contextmanager
def _running_server(
    mode: Literal["sync", "aio", "auto"],
    tls: GrpcServerTLSConfig,
    *,
    api_key_file: Path | None = None,
) -> Iterator[int]:
    """Start a loopback MP server and yield its ephemeral port."""
    server = build_grpc_request_server(
        [cast(EngineModule, _TestModule())],
        MPServerConfig(
            host="127.0.0.1",
            port=0,
            grpc_server_mode=mode,
            grpc_tls=tls,
            api_key_files=[str(api_key_file)] if api_key_file is not None else [],
        ),
    )
    try:
        server.start()
        yield server.bound_port
    finally:
        server.close(wait=True)


def _read_first_batch(client: RequestClient) -> KVEventBatch:
    """Bound the read deadline, including when a handshake or RPC regresses."""
    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        closing(client.subscribe_kv_events(1, "model", 0, 1)) as stream,
    ):
        return executor.submit(next, stream).result(timeout=5)


def _connect(
    port: int,
    tls: GrpcClientTLSConfig,
    *,
    host: str = _SERVER_NAME,
    api_key: str | None = None,
) -> AbstractContextManager[RequestClient]:
    """Connect to the loopback server and close its channel at context exit."""
    return closing(
        GrpcMultiprocessClient(  # type: ignore[abstract]
            f"grpc://{host}:{port}", api_key=api_key, tls_config=tls
        )
    )


def test_tls_round_trip(
    server_mode: Literal["sync", "aio"], certificates: _Certificates
) -> None:
    """A trusted TLS client can use both RPC shapes and plaintext is rejected."""
    with _running_server(server_mode, certificates.server) as port:
        with _connect(port, certificates.client) as client:
            assert client.noop().result(timeout=5) == "ok"
            assert _read_first_batch(client) == _BATCH

        with grpc.insecure_channel(
            f"127.0.0.1:{port}", options=(("grpc.enable_http_proxy", 0),)
        ) as channel:
            noop = channel.unary_unary("/lmcache.mp.ManagementService/Noop")
            with pytest.raises(grpc.RpcError) as rejected:
                noop(b"", timeout=5)
            assert rejected.value.code() is grpc.StatusCode.UNAVAILABLE


def test_tls_with_api_key_authentication(
    server_mode: Literal["sync", "aio"], certificates: _Certificates, tmp_path: Path
) -> None:
    """The TLS channel sends credentials for unary and streaming calls."""
    api_key_file = tmp_path / "api-key"
    api_key_file.write_text("tls-test-key\n", encoding="utf-8")
    with _running_server(
        server_mode, certificates.server, api_key_file=api_key_file
    ) as port:
        with _connect(port, certificates.client, api_key="tls-test-key") as client:
            assert client.noop().result(timeout=5) == "ok"
            assert _read_first_batch(client) == _BATCH

        with _connect(port, certificates.client, api_key="wrong-key") as bad_client:
            with pytest.raises(MPRequestError) as unary:
                bad_client.noop().result(timeout=5)
            with pytest.raises(MPRequestError) as streaming:
                _read_first_batch(bad_client)
            assert unary.value.code == streaming.value.code == "UNAUTHENTICATED"


@pytest.mark.parametrize("failure", ["untrusted_ca", "hostname_mismatch"])
def test_tls_rejects_unverified_servers(
    server_mode: Literal["sync", "aio"],
    certificates: _Certificates,
    failure: str,
) -> None:
    """The client verifies server identity, including when a name is overridden."""
    trusted = replace(certificates.client, server_name_override=_SERVER_NAME)
    with _running_server(server_mode, certificates.server) as port:
        with _connect(port, trusted, host="127.0.0.1") as client:
            assert client.noop().result(timeout=5) == "ok"

        unverified = (
            replace(trusted, ca_file=certificates.untrusted_ca)
            if failure == "untrusted_ca"
            else replace(trusted, server_name_override=None)
        )
        with _connect(port, unverified, host="127.0.0.1") as rejected_client:
            # Wait-for-ready RPCs stay pending after a rejected handshake.
            with pytest.raises(TimeoutError):
                rejected_client.noop().result(timeout=1)


def test_mtls_requires_client_identity(
    server_mode: Literal["sync", "aio"], certificates: _Certificates
) -> None:
    """mTLS accepts a signed client certificate and requires one to connect."""
    server_tls = replace(
        certificates.server,
        ca_file=certificates.client.ca_file,
        require_client_cert=True,
    )
    valid = replace(
        certificates.client,
        cert_file=certificates.client_cert,
        key_file=certificates.client_key,
    )
    with _running_server(server_mode, server_tls) as port:
        with _connect(port, valid) as client:
            assert client.noop().result(timeout=5) == "ok"

        with _connect(port, certificates.client) as rejected_client:
            with pytest.raises(TimeoutError):
                rejected_client.noop().result(timeout=1)


def test_auto_with_tls_prefers_a_tls_capable_server(
    certificates: _Certificates, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto can serve TLS even when the native extension appears installed."""
    native = ModuleType("pygrpc_native")
    native.__spec__ = ModuleSpec("pygrpc_native", loader=None)
    monkeypatch.setitem(sys.modules, native.__name__, native)
    with (
        _running_server("auto", certificates.server) as port,
        _connect(port, certificates.client) as client,
    ):
        assert client.noop().result(timeout=5) == "ok"


def test_native_with_tls_is_rejected(certificates: _Certificates) -> None:
    """An explicit native request with TLS fails before binding a socket."""
    config = MPServerConfig(
        host="127.0.0.1",
        port=0,
        grpc_server_mode="native",
        grpc_tls=certificates.server,
    )
    with pytest.raises(ValueError, match=r"(?i)native.*tls|tls.*native"):
        build_grpc_request_server([], config)


def test_zmq_with_tls_is_rejected(certificates: _Certificates) -> None:
    """A ZMQ endpoint cannot silently ignore an MP TLS configuration."""
    with pytest.raises(ValueError, match=r"grpc_tls.*grpc"):
        MPServerConfig(transport="zmq", grpc_tls=certificates.server)


@pytest.mark.parametrize("field", ["cert_file", "key_file"])
def test_server_tls_rejects_missing_pem_file(
    certificates: _Certificates, tmp_path: Path, field: str
) -> None:
    """A missing server identity file produces a startup configuration error."""
    server_tls = replace(certificates.server, **{field: tmp_path / "missing.pem"})
    with pytest.raises(ValueError, match=field), _running_server("sync", server_tls):
        pass


def test_client_tls_rejects_missing_ca_file(
    certificates: _Certificates, tmp_path: Path
) -> None:
    """An unreadable CA bundle produces a configuration error immediately."""
    client_tls = replace(certificates.client, ca_file=tmp_path / "missing-ca.pem")
    with pytest.raises(ValueError, match="ca_file"), _connect(1, client_tls):
        pass


@pytest.mark.parametrize(
    ("cert_file", "key_file"), [("", ""), ("server.pem", ""), ("", "server.key")]
)
def test_server_tls_requires_cert_and_key(cert_file: str, key_file: str) -> None:
    """A TLS server configuration needs both server identity paths."""
    with pytest.raises(ValueError, match="cert_file|key_file"):
        GrpcServerTLSConfig(cert_file=cert_file, key_file=key_file)


def test_mtls_requires_a_ca(certificates: _Certificates) -> None:
    """Client certificate enforcement requires an explicit trust bundle."""
    with pytest.raises(ValueError, match="ca_file"):
        replace(certificates.server, require_client_cert=True)


@pytest.mark.parametrize(
    ("cert_file", "key_file"), [("client.pem", None), (None, "client.key")]
)
def test_client_tls_rejects_partial_identity(
    cert_file: str | None, key_file: str | None
) -> None:
    """A client supplies neither identity path or both of them."""
    with pytest.raises(ValueError, match="cert_file|key_file"):
        GrpcClientTLSConfig(enabled=True, cert_file=cert_file, key_file=key_file)


def test_disabled_client_tls_rejects_tls_options() -> None:
    """Explicit credentials cannot silently downgrade to a plaintext client."""
    with pytest.raises(ValueError, match="enabled|enable"):
        GrpcClientTLSConfig(ca_file="ca.pem")
