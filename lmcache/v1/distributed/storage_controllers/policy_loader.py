# SPDX-License-Identifier: Apache-2.0
"""Explicit loading for operator-installed MP storage policy classes."""

# Standard
from typing import TypeVar
import importlib
import inspect

PolicyT = TypeVar("PolicyT")


def load_external_policy(selection: str, policy_type: type[PolicyT]) -> PolicyT:
    """Construct an explicitly selected external policy.

    Args:
        selection: Import path in ``package.module:Class`` form.
        policy_type: Required policy base class.

    Returns:
        A concrete policy constructed without arguments, like registry policies.

    Raises:
        ValueError: The path is malformed or does not name a concrete subclass.
        ImportError: The selected module cannot be imported.
        AttributeError: The selected module does not define the named attribute.
        Exception: The policy constructor raises; its exception is propagated.

    Note:
        Imports happen only for an explicit selection. Plugins are trusted code;
        neither imports nor construction are sandboxed.
    """
    module_path, separator, class_name = selection.partition(":")
    if not separator or not module_path or not class_name or ":" in class_name:
        raise ValueError(
            f"Invalid policy {selection!r}: expected 'package.module:Class'"
        )
    policy_cls = getattr(importlib.import_module(module_path), class_name)
    if (
        not isinstance(policy_cls, type)
        or not issubclass(policy_cls, policy_type)
        or inspect.isabstract(policy_cls)
    ):
        raise ValueError(
            f"Policy {selection!r} must be a concrete {policy_type.__name__} subclass"
        )
    return policy_cls()
