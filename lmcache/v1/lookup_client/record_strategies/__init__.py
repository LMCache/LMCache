# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Dict, Type
import importlib
import inspect
import pkgutil

# First Party
from lmcache.logging import init_logger
from lmcache.v1.lookup_client.record_strategies.base import RecordStrategy

logger = init_logger(__name__)


def _discover_strategies() -> Dict[str, Type[RecordStrategy]]:
    """Auto-discover all RecordStrategy implementations in the current package."""
    strategies = {}

    # Import current package
    # First Party
    from lmcache.v1.lookup_client import record_strategies

    # Iterate through all modules in the package
    for importer, modname, ispkg in pkgutil.iter_modules(
        record_strategies.__path__, record_strategies.__name__ + "."
    ):
        try:
            # Import the module
            module = importlib.import_module(modname)

            # Find all classes that inherit from RecordStrategy
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, RecordStrategy)
                    and obj is not RecordStrategy
                    and hasattr(obj, "name")
                ):
                    try:
                        strategy_name = obj.name()
                        strategies[strategy_name] = obj
                    except Exception:
                        # Skip classes where name() method fails
                        continue
        except Exception:
            # Skip modules that can't be imported
            continue

    return strategies


# Cache discovered strategies
_strategies_cache = None


def _get_strategies() -> Dict[str, Type[RecordStrategy]]:
    """Get cached strategies, discovering them if necessary."""
    global _strategies_cache
    if _strategies_cache is None:
        _strategies_cache = _discover_strategies()
    return _strategies_cache


def create_record_strategy(
    strategy_name: str,
    chunk_size: int,
    config,
) -> RecordStrategy:
    """
    Factory function to create record strategy instances.

    Args:
        strategy_name: Type of strategy (name returned by strategy's name() method)
        chunk_size: Size of each chunk
        config: LMCacheEngineConfig instance

    Returns:
        RecordStrategy instance

    Raises:
        ValueError: If strategy_type is not supported
    """
    strategies = _get_strategies()

    if strategy_name not in strategies:
        available = list(strategies.keys())
        raise ValueError(
            f"Unknown record strategy name: {strategy_name}. "
            f"Available strategies: {available}"
        )

    strategy_class = strategies[strategy_name]
    logger.info("Creating record strategy %s within %s", strategy_name, strategies)
    return strategy_class(config, chunk_size)  # type: ignore[arg-type]


def list_record_strategies() -> list[str]:
    """List all available record strategy names."""
    return list(_get_strategies().keys())
