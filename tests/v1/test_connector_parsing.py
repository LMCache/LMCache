# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.connector import CreateConnector, parse_remote_url


def test_create_connector_rejects_missing_scheme() -> None:
    """CreateConnector should reject URLs without a scheme separator."""
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(ValueError) as excinfo:
            CreateConnector("localhost:6379", loop, local_cpu_backend=None)

        assert "missing scheme" in str(excinfo.value)
    finally:
        loop.close()


def test_parse_remote_url_rejects_missing_host() -> None:
    """parse_remote_url should fail when the host component is missing."""
    _assert_parse_remote_url_missing_component(
        "redis://:6379", expected_substring="missing host"
    )


def test_parse_remote_url_rejects_missing_port() -> None:
    """parse_remote_url should fail when the port component is missing."""
    _assert_parse_remote_url_missing_component(
        "redis://localhost", expected_substring="missing port"
    )


def test_parse_remote_url_parses_basic_lm_url_components() -> None:
    """parse_remote_url should return the expected fields for a basic LM URL."""
    parsed = parse_remote_url("lm://localhost:65000")

    assert parsed.host == "localhost"
    assert parsed.port == 65000
    assert parsed.path == ""
    assert parsed.username is None
    assert parsed.password is None
    assert parsed.query_params == {}


def test_parse_remote_url_parses_filesystem_style_path() -> None:
    """parse_remote_url should preserve the path for filesystem-style URLs."""
    parsed = parse_remote_url("fs://host:0/tmp/lmcache")

    assert parsed.host == "host"
    assert parsed.port == 0
    assert parsed.path == "/tmp/lmcache"
    assert parsed.username is None
    assert parsed.password is None
    assert parsed.query_params == {}


def test_parse_remote_url_parses_credentials() -> None:
    """parse_remote_url should capture username and password when present."""
    parsed = parse_remote_url("redis://user:password@localhost:6379/0")

    assert parsed.host == "localhost"
    assert parsed.port == 6379
    assert parsed.path == "/0"
    assert parsed.username == "user"
    assert parsed.password == "password"
    assert parsed.query_params == {}


def test_parse_remote_url_parses_query_params_with_parse_qs_semantics() -> None:
    """parse_remote_url should expose query params in parse_qs-compatible form."""
    parsed = parse_remote_url(
        "infinistore://127.0.0.1:12345?device=mlx5_0&verify=true"
    )

    assert parsed.host == "127.0.0.1"
    assert parsed.port == 12345
    assert parsed.path == ""
    assert parsed.username is None
    assert parsed.password is None
    assert parsed.query_params == {"device": ["mlx5_0"], "verify": ["true"]}


def _assert_parse_remote_url_missing_component(
    url: str, *, expected_substring: str
) -> None:
    with pytest.raises(AssertionError) as excinfo:
        parse_remote_url(url)

    assert expected_substring in str(excinfo.value)
