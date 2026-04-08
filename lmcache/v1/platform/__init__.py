# SPDX-License-Identifier: Apache-2.0
"""Cross-platform abstraction layer for LMCache.

This package centralizes all platform-specific logic so that
business-logic modules never need to check
``torch.cuda.is_available()`` or ``hasattr(os, "eventfd")``
directly.

Public API::

    from lmcache.v1.platform import (
        HAS_CUDA,
        HAS_EVENTFD,
        EventNotifier,
        MemoryPinner,
        consume_fd,
        create_event_notifier,
        create_memory_pinner,
        cuda_init,
        current_device_id,
        lmc_ops,
        safe_device,
        synchronize,
    )
"""

# First Party
from lmcache.v1.platform.capabilities import (  # noqa: F401
    HAS_CUDA,
    HAS_EVENTFD,
)
from lmcache.v1.platform.cuda_utils import (  # noqa: F401
    cuda_init,
    current_device_id,
    safe_device,
    synchronize,
)
from lmcache.v1.platform.event_notifier import (  # noqa: F401
    EventNotifier,
    consume_fd,
    create_event_notifier,
)
from lmcache.v1.platform.memory_pinner import (  # noqa: F401
    MemoryPinner,
    create_memory_pinner,
)
from lmcache.v1.platform.ops import lmc_ops  # noqa: F401
