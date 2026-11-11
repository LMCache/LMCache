# SPDX-License-Identifier: Apache-2.0
"""Transfer channel implementations.

Importing this package imports each implementation module, which in turn
self-registers its factory via ``register_transfer_channel_factory``.
"""

# First Party
from lmcache.v1.distributed.transfer_channel.impl import (  # noqa: F401
    hixl_impl,
    nixl_impl,
)

try:
    # First Party
    from lmcache.v1.distributed.transfer_channel.impl import (  # noqa: F401
        mooncake_te_impl,
    )
except ModuleNotFoundError as e:
    if e.name != "mooncake":
        raise
