# SPDX-License-Identifier: Apache-2.0
"""Compatibility tests for the legacy ``lmcache.lmcache_frontend`` imports.

The fleet frontend moved to :mod:`lmcache.v1.server.frontend`; the old
import path remains a compatibility facade until it is separately
deprecated. These tests pin the facade: every legacy import must resolve
to the same module object as the canonical path.
"""


def test_legacy_app_module_is_canonical():
    """``lmcache.lmcache_frontend.app`` is the canonical app module."""
    # First Party
    from lmcache.lmcache_frontend import app as legacy_app
    from lmcache.v1.server.frontend import app as canonical_app

    assert legacy_app is canonical_app
    assert callable(legacy_app.main)


def test_legacy_heartbeat_module_is_canonical():
    """``lmcache.lmcache_frontend.heartbeat`` is the canonical module."""
    # First Party
    from lmcache.lmcache_frontend import heartbeat as legacy_heartbeat
    from lmcache.v1.server.frontend import heartbeat as canonical_heartbeat
    from lmcache.v1.server.frontend.heartbeat import HeartbeatService

    assert legacy_heartbeat is canonical_heartbeat
    assert legacy_heartbeat.HeartbeatService is HeartbeatService


def test_legacy_heartbeat_service_import():
    """``HeartbeatService`` imports from the legacy submodule path."""
    # First Party
    from lmcache.lmcache_frontend.heartbeat import HeartbeatService
    from lmcache.v1.server.frontend.heartbeat import (
        HeartbeatService as CanonicalHeartbeatService,
    )

    assert HeartbeatService is CanonicalHeartbeatService


def test_legacy_mp_plugin_is_canonical():
    """The legacy MP plugin path resolves to the canonical module."""
    # First Party
    from lmcache.lmcache_frontend.lmcache_mp_plugin import (
        lmcache_mp_frontend_plugin as legacy_plugin,
    )
    from lmcache.v1.server.frontend.lmcache_mp_plugin import (
        lmcache_mp_frontend_plugin as canonical_plugin,
    )

    assert legacy_plugin is canonical_plugin
    assert callable(legacy_plugin.build_argv)
