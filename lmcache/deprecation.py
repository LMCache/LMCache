# SPDX-License-Identifier: Apache-2.0
# Standard
import os

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

IN_PROCESS_DEPRECATION_MESSAGE = (
    "LMCache in-process KV-cache mode is deprecated. For new deployments, use "
    "the standalone 'lmcache server' with the matching MP connector. Existing "
    "in-process configuration continues to work. Migration guide: "
    "https://docs.lmcache.ai/legacy/migration_to_mp.html"
)

_warned_pid: int | None = None


def warn_in_process_mode_deprecated() -> None:
    """Log the in-process migration warning once per operating-system process."""
    global _warned_pid
    pid = os.getpid()
    if _warned_pid == pid:
        return
    _warned_pid = pid
    logger.warning(IN_PROCESS_DEPRECATION_MESSAGE)
