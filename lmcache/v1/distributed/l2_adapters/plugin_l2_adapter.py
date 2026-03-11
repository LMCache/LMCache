# SPDX-License-Identifier: Apache-2.0
"""
Plugin L2 adapter -- dynamically loads an external adapter
class from a user-supplied Python module.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any
import importlib

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.l2_adapters.base import (
        L2AdapterInterface,
    )

# First Party
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_factory,
    register_l2_adapter_type,
)

# Config class


class PluginL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for a plugin L2 adapter.

    Dynamically loads an adapter class from a user-supplied
    Python module at creation time.

    Fields:
    - module_path: Dotted Python import path of the module
        containing the adapter class.
    - class_name: Name of the class inside *module_path*
        that implements ``L2AdapterInterface``.
    - adapter_params: Arbitrary dict of keyword arguments
        forwarded to the adapter class constructor.
    """

    def __init__(
        self,
        module_path: str,
        class_name: str,
        adapter_params: dict[str, Any] | None = None,
    ):
        self.module_path = module_path
        self.class_name = class_name
        self.adapter_params = adapter_params or {}

    @classmethod
    def from_dict(cls, d: dict) -> "PluginL2AdapterConfig":
        module_path = d.get("module_path")
        if not isinstance(module_path, str) or not module_path:
            raise ValueError("module_path must be a non-empty string")

        class_name = d.get("class_name")
        if not isinstance(class_name, str) or not class_name:
            raise ValueError("class_name must be a non-empty string")

        adapter_params = d.get("adapter_params", {})
        if not isinstance(adapter_params, dict):
            raise ValueError("adapter_params must be a dict")

        return cls(
            module_path=module_path,
            class_name=class_name,
            adapter_params=adapter_params,
        )

    @classmethod
    def help(cls) -> str:
        return (
            "Plugin L2 adapter config fields:\n"
            "- module_path (str): dotted import path of "
            "the module containing the adapter class "
            "(required)\n"
            "- class_name (str): name of the adapter "
            "class inside the module (required)\n"
            "- adapter_params (dict): keyword arguments "
            "forwarded to the adapter constructor "
            "(optional, default {})\n"
            "\n"
            "Example JSON:\n"
            '{"type": "plugin", '
            '"module_path": "my_plugin.l2", '
            '"class_name": "MyL2Adapter", '
            '"adapter_params": {"host": "localhost"}}'
        )


# Factory function


def _create_plugin_adapter(
    config: PluginL2AdapterConfig,
    **kwargs: object,
) -> "L2AdapterInterface":
    """Dynamically load and create a plugin L2 adapter."""
    # First Party
    from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface as _L2AI

    try:
        module = importlib.import_module(config.module_path)
    except ImportError as e:
        raise ImportError(
            "Could not import module '%s': %s" % (config.module_path, e)
        ) from e

    try:
        adapter_cls = getattr(module, config.class_name)
    except AttributeError as e:
        raise AttributeError(
            "Module '%s' has no class '%s': %s"
            % (config.module_path, config.class_name, e)
        ) from e

    if not (isinstance(adapter_cls, type) and issubclass(adapter_cls, _L2AI)):
        raise TypeError(
            "%s.%s is not a subclass of "
            "L2AdapterInterface" % (config.module_path, config.class_name)
        )

    return adapter_cls(config.adapter_params, **kwargs)  # type: ignore[call-arg]


# Self-register config type and adapter factory
register_l2_adapter_type("plugin", PluginL2AdapterConfig)
register_l2_adapter_factory("plugin", _create_plugin_adapter)
