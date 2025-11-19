# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Dict, Type
import importlib
import inspect
import pkgutil

# First Party
from lmcache.logging import init_logger
from lmcache.v1.lookup_client.record_strategies.base import (
    AsyncRecorder,
    RecordStrategy,
)

logger = init_logger(__name__)


def _discover_strategies() -> Dict[str, Type[RecordStrategy]]:
    strategies = {}
    # First Party
    from lmcache.v1.lookup_client import record_strategies

    for importer, modname, ispkg in pkgutil.iter_modules(
        record_strategies.__path__, record_strategies.__name__ + "."
    ):
        try:
            module = importlib.import_module(modname)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, RecordStrategy) and obj is not RecordStrategy:
                    # Use module name as strategy name
                    strategy_name = modname.split(".")[-1]
                    strategies[strategy_name] = obj
        except Exception:
            continue
    return strategies


_strategies_cache = None


def _get_strategies() -> Dict[str, Type[RecordStrategy]]:
    global _strategies_cache
    if _strategies_cache is None:
        _strategies_cache = _discover_strategies()
    return _strategies_cache


def create_record_strategy(config) -> RecordStrategy:
    strategies = _get_strategies()
    strategy_name = config.chunk_statistics_strategy
    chunk_size = config.chunk_size
    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}"
        )
    return strategies[strategy_name](config, chunk_size)  # type: ignore[call-arg]


def list_record_strategies() -> list[str]:
    return list(_get_strategies().keys())


__all__ = [
    "AsyncRecorder",
    "RecordStrategy",
    "create_record_strategy",
    "list_record_strategies",
]


def __getattr__(name):
    strategies = _get_strategies()
    for strategy_class in strategies.values():
        if strategy_class.__name__ == name:
            return strategy_class
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
