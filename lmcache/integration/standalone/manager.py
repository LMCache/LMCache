# SPDX-License-Identifier: Apache-2.0
"""
StandaloneLMCacheManager: A specialized manager for LMCache standalone mode.

This class extends LMCacheManager to handle standalone mode specifically,
removing vLLM dependencies and simplifying the initialization logic.
"""

# Standard
from typing import TYPE_CHECKING, Any, Optional

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.v1.manager import LMCacheManager

if TYPE_CHECKING:
    # Fir
    pass

logger = init_logger(__name__)


class StandaloneLMCacheManager(LMCacheManager):
    """
    LMCacheManager specialized for standalone mode.

    This class handles the standalone mode without vLLM dependencies,
    providing a cleaner and more focused implementation.
    """

    def __init__(
        self,
        config: Any,
        metadata: LMCacheEngineMetadata,
        northbound: Optional[Any] = None,
    ):
        """
        Initialize StandaloneLMCacheManager.

        Args:
            config: LMCache engine configuration
            metadata: Pre-constructed LMCacheEngineMetadata
                (from StandaloneMetadataBuilder)
            northbound: Reference to northbound adapter
                (LMCacheStandaloneStarter) for API server
        """
        # Call parent __init__ - it will create all components based on metadata
        super().__init__(
            config=config,
            metadata=metadata,
            northbound=northbound,
        )
