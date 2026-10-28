# SPDX-License-Identifier: Apache-2.0

"""Optional shared-token authentication for the PD allocation plane.

PD disaggregation exposes a ROUTER socket that accepts allocation and
cancellation messages from any peer that can reach it.  Without a check
on the receiving side, an unrelated peer can hold buffer reservations,
register keys it never wrote, or release another request's keys.  This
module implements the shared-secret half of that check: senders stamp a
token derived from configuration onto ``AllocRequest`` / ``CancelNotif``
and the receiver compares it in constant time before touching any state.

Authentication is opt-in.  When no token is configured the comparison
is skipped, so existing deployments keep working unchanged.
"""

# Future
from __future__ import annotations

# Standard
import hmac
import os
from typing import Any, Optional

PD_AUTH_TOKEN_ENV_VAR = "LMCACHE_PD_AUTH_TOKEN"
PD_AUTH_TOKEN_CONFIG_KEY = "pd_auth_token"


def get_pd_auth_token(config: Any) -> Optional[bytes]:
    """Resolve the PD shared token from configuration or the environment.

    ``extra_config["pd_auth_token"]`` wins over ``LMCACHE_PD_AUTH_TOKEN``
    so a deployment can pin the token per process without exporting an
    environment variable.  An empty or missing value disables
    authentication.

    Args:
        config: The ``LMCacheEngineConfig`` (or any object exposing an
            ``extra_config`` mapping).

    Returns:
        The token as UTF-8 bytes, or ``None`` when authentication is off.
    """
    raw: Any = None
    extra_config = getattr(config, "extra_config", None)
    if extra_config:
        raw = extra_config.get(PD_AUTH_TOKEN_CONFIG_KEY)

    if raw is None:
        raw = os.environ.get(PD_AUTH_TOKEN_ENV_VAR)

    if raw is None:
        return None

    token = raw if isinstance(raw, bytes) else str(raw).encode("utf-8")
    return token or None


def pd_auth_token_matches(expected: Optional[bytes], provided: bytes) -> bool:
    """Check a peer-supplied token against the configured one.

    Args:
        expected: The configured token, or ``None`` when authentication is
            disabled.
        provided: The token carried by the incoming message.

    Returns:
        ``True`` when authentication is disabled or the tokens are equal.
    """
    if expected is None:
        return True
    if not provided:
        return False
    return hmac.compare_digest(expected, provided)
