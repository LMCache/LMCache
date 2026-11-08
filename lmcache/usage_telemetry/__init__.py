# SPDX-License-Identifier: Apache-2.0
"""Anonymous usage telemetry for LMCache.

Phone-home usage statistics, described in this package's ``README.md``.

- The wire schema of every message lives in :mod:`.messages` — the single
  source of truth for what LMCache can phone home. The InProcess engine's
  messages live in :mod:`lmcache.inprocess.usage_telemetry.messages`, in the
  external LMCache_inprocess package.
- Context reporting (:mod:`.base`, :mod:`.mp`) sends a
  snapshot of the environment and configuration at startup. The
  single-process reporter lives in the external LMCache_inprocess package,
  in :mod:`lmcache.inprocess.usage_telemetry.context` and
  :mod:`lmcache.inprocess.usage_telemetry.continuous`.

Every outgoing payload is stamped with a :class:`UsageIdentity` (per-process
``session_id`` plus persistent ``machine_id``) so the stats backend can join
continuous messages with the one-shot context that describes the deployment.

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
    ContinuousContextMessage,
    DeploymentMode,
    EnvMessage,
    MPServerMessage,
    UsageMessage,
)
from lmcache.usage_telemetry.mp import InitializeMPUsageContext, MPUsageContext
from lmcache.usage_telemetry.transport import UsageMessageSender

__all__ = [
    "USAGE_SCHEMA_VERSION",
    "ContinuousContextMessage",
    "DeploymentMode",
    "EnvMessage",
    "InitializeMPUsageContext",
    "MPServerMessage",
    "MPUsageContext",
    "UsageContextBase",
    "UsageIdentity",
    "UsageMessage",
    "UsageMessageSender",
    "collect_env_message",
    "get_usage_identity",
    "is_usage_tracking_enabled",
]
