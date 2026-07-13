# SPDX-License-Identifier: Apache-2.0
"""Anonymous usage telemetry for LMCache.

Phone-home usage statistics, described in this package's ``README.md``.

This root package holds only what both deployment modes share: the wire
schema (:mod:`.messages` — the single source of truth for what LMCache
can phone home), identity and opt-out (:mod:`.identity`), transport
(:mod:`.transport`), the no-throw guard (:mod:`.guard`), environment
probing (:mod:`.env_probe`), and the one-shot reporter base
(:mod:`.base`).

Mode-specific reporters are fully separated: MP-server reporting lives in
:mod:`.mp` (payloads POST under the ``/mp/`` endpoint prefix);
single-process reporting lives in :mod:`.non_mp` and is scheduled for
removal together with that code path.

Every entry point called from serving code is wrapped with
:func:`lmcache.usage_telemetry.guard.swallow_telemetry_errors`: a failure
anywhere in telemetry can never affect caching or serving functionality.

Users can opt out at any time; see :func:`is_usage_tracking_enabled`.
"""

# First Party
from lmcache.usage_telemetry.base import UsageContextBase
from lmcache.usage_telemetry.env_probe import collect_env_message
from lmcache.usage_telemetry.identity import (
    UsageIdentity,
    get_usage_identity,
    is_usage_tracking_enabled,
)
from lmcache.usage_telemetry.messages import (
    USAGE_SCHEMA_VERSION,
    CacheLifespanMessage,
    ContinuousContextMessage,
    DeploymentMode,
    EngineMessage,
    EnvMessage,
    MetadataMessage,
    MPServerMessage,
    UsageMessage,
)
from lmcache.usage_telemetry.transport import UsageMessageSender

__all__ = [
    "USAGE_SCHEMA_VERSION",
    "CacheLifespanMessage",
    "ContinuousContextMessage",
    "DeploymentMode",
    "EngineMessage",
    "EnvMessage",
    "MPServerMessage",
    "MetadataMessage",
    "UsageContextBase",
    "UsageIdentity",
    "UsageMessage",
    "UsageMessageSender",
    "collect_env_message",
    "get_usage_identity",
    "is_usage_tracking_enabled",
]
