# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the version-gated ``client_info_tag`` support in
``ValkeyConnector`` / ``_ThreadWorkerPool``.

The connector tags LMCache's GLIDE connections as
``GlidePySync(lmcache:<version>)`` via GLIDE's ``client_info_tag`` client
config option, but only when the installed ``valkey-glide`` build supports
it (>= 2.5.0, valkey-io/valkey-glide#6389). Older builds would raise
``TypeError`` on the unknown kwarg, so the connector feature-detects support
from the config constructor signature and applies the tag behind that gate.

These tests inject a fake ``glide_sync`` module so the real
``_ThreadWorkerPool._get_client`` config-building path is exercised without
``glide_sync`` or a real Valkey server.
"""

# Standard
from unittest.mock import patch
import sys
import types

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.connector.valkey_connector import (
    CLIENT_INFO_TAG_MIN_GLIDE_VERSION,
    _glide_config_supports_client_info_tag,
    _lmcache_client_info_tag,
    _ThreadWorkerPool,
)


def _make_fake_glide_sync(supports_tag: bool) -> types.ModuleType:
    """Build a fake ``glide_sync`` module for ``_get_client``.

    The config classes accept ``client_info_tag`` only when
    ``supports_tag`` is True, mirroring pre- and post-2.5.0 glide builds.
    Each config records the kwargs it was constructed with, and each client
    records the last config passed to ``create`` so tests can assert what
    the connector threaded through.
    """
    mod = types.ModuleType("glide_sync")

    class NodeAddress:
        def __init__(self, host, port):
            self.host = host
            self.port = port

    class ServerCredentials:
        def __init__(self, username, password):
            self.username = username
            self.password = password

    class AdvancedGlideClientConfiguration:
        def __init__(self, connection_timeout=None):
            self.connection_timeout = connection_timeout

    class AdvancedGlideClusterClientConfiguration:
        def __init__(self, connection_timeout=None):
            self.connection_timeout = connection_timeout

    if supports_tag:

        class GlideClientConfiguration:
            def __init__(
                self,
                addresses,
                request_timeout=None,
                use_tls=False,
                advanced_config=None,
                credentials=None,
                database_id=None,
                client_info_tag=None,
            ):
                self.kwargs = {
                    "database_id": database_id,
                    "client_info_tag": client_info_tag,
                }

        class GlideClusterClientConfiguration:
            def __init__(
                self,
                addresses,
                request_timeout=None,
                use_tls=False,
                advanced_config=None,
                credentials=None,
                client_info_tag=None,
            ):
                self.kwargs = {"client_info_tag": client_info_tag}

    else:

        class GlideClientConfiguration:
            def __init__(
                self,
                addresses,
                request_timeout=None,
                use_tls=False,
                advanced_config=None,
                credentials=None,
                database_id=None,
            ):
                self.kwargs = {"database_id": database_id}

        class GlideClusterClientConfiguration:
            def __init__(
                self,
                addresses,
                request_timeout=None,
                use_tls=False,
                advanced_config=None,
                credentials=None,
            ):
                self.kwargs = {}

    class _FakeClientBase:
        last_config = None

        def __init__(self, config):
            self.config = config

        def get(self, key, buffer=None):  # signature drives has_buffer_get
            return None

        def close(self):
            pass

        @classmethod
        def create(cls, config):
            cls.last_config = config
            return cls(config)

    class GlideClient(_FakeClientBase):
        last_config = None

    class GlideClusterClient(_FakeClientBase):
        last_config = None

    mod.NodeAddress = NodeAddress
    mod.ServerCredentials = ServerCredentials
    mod.AdvancedGlideClientConfiguration = AdvancedGlideClientConfiguration
    mod.AdvancedGlideClusterClientConfiguration = (
        AdvancedGlideClusterClientConfiguration
    )
    mod.GlideClientConfiguration = GlideClientConfiguration
    mod.GlideClusterClientConfiguration = GlideClusterClientConfiguration
    mod.GlideClient = GlideClient
    mod.GlideClusterClient = GlideClusterClient
    return mod


# ── _lmcache_client_info_tag ────────────────────────────────────────────


def test_client_info_tag_value_format():
    """The tag is ``lmcache:<version>`` and contains no whitespace (a
    whitespace tag would be rejected by GLIDE's config validation)."""
    tag = _lmcache_client_info_tag()
    assert tag.startswith("lmcache:")
    assert not any(ch.isspace() for ch in tag)


def test_client_info_tag_whitespace_fallback(monkeypatch):
    """If the resolved version string somehow contains whitespace, fall back
    to a bare ``lmcache`` tag rather than emit an invalid tag."""
    # First Party
    import lmcache

    monkeypatch.setattr(lmcache, "__version__", "1.0 dev build", raising=False)
    assert _lmcache_client_info_tag() == "lmcache"


# ── _glide_config_supports_client_info_tag ──────────────────────────────


def test_supports_detection_true_and_false():
    """Support is feature-detected from the constructor signature."""

    class WithTag:
        def __init__(self, addresses, client_info_tag=None):
            pass

    class WithoutTag:
        def __init__(self, addresses):
            pass

    assert _glide_config_supports_client_info_tag(WithTag) is True
    assert _glide_config_supports_client_info_tag(WithoutTag) is False


def test_supports_detection_uninspectable_returns_false():
    """A type whose signature cannot be introspected is treated as
    unsupported rather than raising."""
    # ``object`` has no introspectable __init__ signature.
    assert _glide_config_supports_client_info_tag(object) is False


def test_min_version_constant():
    """The documented minimum glide version is 2.5.0."""
    assert CLIENT_INFO_TAG_MIN_GLIDE_VERSION == "2.5.0"


# ── _get_client config passthrough (real _ThreadWorkerPool) ─────────────


@pytest.mark.parametrize("cluster_mode", [False, True])
def test_get_client_sets_tag_when_supported(cluster_mode):
    """When the glide config supports it, the connector threads the LMCache
    tag through to the client config (standalone and cluster)."""
    fake = _make_fake_glide_sync(supports_tag=True)
    with patch.dict(sys.modules, {"glide_sync": fake}):
        pool = _ThreadWorkerPool(
            host="h",
            port=1,
            num_workers=1,
            username="",
            password="",
            cluster_mode=cluster_mode,
        )
        try:
            client_cls = fake.GlideClusterClient if cluster_mode else fake.GlideClient
            config = client_cls.last_config
            assert config is not None
            assert config.kwargs["client_info_tag"] == _lmcache_client_info_tag()
        finally:
            pool.close()


@pytest.mark.parametrize("cluster_mode", [False, True])
def test_get_client_omits_tag_when_unsupported(cluster_mode):
    """On an older glide build (no ``client_info_tag`` param) the connector
    must NOT pass the kwarg — doing so would raise ``TypeError``."""
    fake = _make_fake_glide_sync(supports_tag=False)
    with patch.dict(sys.modules, {"glide_sync": fake}):
        pool = _ThreadWorkerPool(
            host="h",
            port=1,
            num_workers=1,
            username="",
            password="",
            cluster_mode=cluster_mode,
        )
        try:
            client_cls = fake.GlideClusterClient if cluster_mode else fake.GlideClient
            config = client_cls.last_config
            assert config is not None
            assert "client_info_tag" not in config.kwargs
        finally:
            pool.close()


def test_get_client_standalone_still_passes_database_id_with_tag():
    """The tag is additive: standalone still forwards database_id alongside
    the new client_info_tag."""
    fake = _make_fake_glide_sync(supports_tag=True)
    with patch.dict(sys.modules, {"glide_sync": fake}):
        pool = _ThreadWorkerPool(
            host="h",
            port=1,
            num_workers=1,
            username="",
            password="",
            cluster_mode=False,
            database_id=7,
        )
        try:
            config = fake.GlideClient.last_config
            assert config.kwargs["database_id"] == 7
            assert config.kwargs["client_info_tag"] == _lmcache_client_info_tag()
        finally:
            pool.close()
