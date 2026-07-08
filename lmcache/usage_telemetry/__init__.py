# SPDX-License-Identifier: Apache-2.0
"""Anonymous usage telemetry for LMCache.

Phone-home usage statistics, described in this package's ``README.md``.
Two reporting paths:

- One-shot context reporting (:mod:`.one_shot`): a snapshot of the
  environment and engine configuration sent once at startup.
- Continuous reporting (:mod:`.continuous`): interval counters
  (hit/stored tokens) and a cache-lifespan histogram flushed periodically.

Every outgoing payload is stamped with a :class:`UsageIdentity` (per-process
``session_id`` plus persistent ``machine_id``) so the stats backend can join
continuous messages with the one-shot context that describes the deployment.

Users can opt out at any time; see :func:`is_usage_tracking_enabled`.
"""

# First Party
from lmcache.usage_telemetry.continuous import (
    CacheLifespanMessage,
    ContinuousContextMessage,
    ContinuousUsageContext,
)
from lmcache.usage_telemetry.env_probe import EnvMessage, collect_env_message
from lmcache.usage_telemetry.identity import (
    UsageIdentity,
    get_usage_identity,
    is_usage_tracking_enabled,
)
from lmcache.usage_telemetry.mp import (
    InitializeMPUsageContext,
    MPServerMessage,
    MPUsageContext,
)
from lmcache.usage_telemetry.one_shot import (
    EngineMessage,
    InitializeUsageContext,
    MetadataMessage,
    UsageContext,
    UsageContextBase,
)
from lmcache.usage_telemetry.transport import (
    USAGE_SCHEMA_VERSION,
    UsageMessageSender,
)

__all__ = [
    "USAGE_SCHEMA_VERSION",
    "CacheLifespanMessage",
    "ContinuousContextMessage",
    "ContinuousUsageContext",
    "EngineMessage",
    "EnvMessage",
    "InitializeMPUsageContext",
    "InitializeUsageContext",
    "MPServerMessage",
    "MPUsageContext",
    "MetadataMessage",
    "UsageContext",
    "UsageContextBase",
    "UsageIdentity",
    "UsageMessageSender",
    "collect_env_message",
    "get_usage_identity",
    "is_usage_tracking_enabled",
]
