# SPDX-License-Identifier: Apache-2.0
"""
Test to ensure wrapper connectors (AuditConnector and InstrumentedRemoteConnector)
implement all methods defined in the base RemoteConnector class.
"""

# Standard
from typing import Dict, List, Set
import inspect

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.connector.audit_connector import AuditConnector
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)


def get_all_methods_from_base(base_class) -> Set[str]:
    """
    Get all public methods defined in the base class (excluding inherited from object).
    """
    methods = set()
    for name in dir(base_class):
        # Skip private and special methods
        if name.startswith("_"):
            continue
        attr = getattr(base_class, name)
        if callable(attr):
            methods.add(name)
    return methods


def get_methods_implemented_in_class(cls) -> Set[str]:
    """
    Get methods that are actually implemented in the class itself (not just inherited).
    This checks if the method is defined in the class's own __dict__
    or its direct parents (excluding the base class we're checking against).

    For classes that dynamically generate methods (like AuditConnector), we check
    the class __dict__ directly without instantiation.
    """
    implemented = set()

    # Check the class's own __dict__ for methods
    # This works for both regular methods and dynamically generated ones
    for name in cls.__dict__:
        if name.startswith("_"):
            continue
        attr = cls.__dict__[name]
        # Check if it's callable (function, method, etc.)
        if callable(attr):
            implemented.add(name)

    # Also check using getattr to catch any dynamically added methods
    # that might not be in __dict__ but are accessible
    for name in dir(cls):
        if name.startswith("_"):
            continue
        if name in implemented:
            continue  # Already found
        try:
            attr = getattr(cls, name)
            if callable(attr):
                # Verify it's not inherited from RemoteConnector
                # by checking if it exists in the class's MRO
                # (excluding RemoteConnector)
                for base in cls.__mro__:
                    if base is RemoteConnector:
                        break
                    if name in base.__dict__:
                        implemented.add(name)
                        break
        except AttributeError:
            pass

    return implemented


def get_abstract_methods(cls) -> Set[str]:
    """
    Get all abstract methods from a class.
    """
    abstract_methods = set()
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if getattr(method, "__isabstractmethod__", False):
            abstract_methods.add(name)
    return abstract_methods


def check_method_signatures(base_class, wrapper_class, wrapper_name: str) -> List[Dict]:
    """
    Check if method signatures in wrapper class match the base class.
    Returns a list of mismatches.
    """
    base_methods = get_all_methods_from_base(base_class)
    signature_mismatches = []

    for method_name in base_methods:
        base_method = getattr(base_class, method_name)
        wrapper_method = getattr(wrapper_class, method_name, None)

        if wrapper_method is None:
            continue

        try:
            base_sig = inspect.signature(base_method)
            wrapper_sig = inspect.signature(wrapper_method)

            # Compare parameter names (excluding 'self')
            base_params = [p for p in base_sig.parameters.keys() if p != "self"]
            wrapper_params = [p for p in wrapper_sig.parameters.keys() if p != "self"]

            if base_params != wrapper_params:
                signature_mismatches.append(
                    {
                        "method": method_name,
                        "base_params": base_params,
                        "wrapper_params": wrapper_params,
                    }
                )
        except (ValueError, TypeError):
            # Some methods might not have inspectable signatures
            pass

    return signature_mismatches


class TestConnectorCompleteness:
    """Test that wrapper connectors implement all base connector methods"""

    def test_audit_connector_completeness(self):
        """
        Comprehensive test to verify AuditConnector implements all methods
        from RemoteConnector with correct signatures.
        """
        # 1. Get all methods from base class
        base_methods = get_all_methods_from_base(RemoteConnector)

        # 2. Get methods actually implemented in AuditConnector
        audit_implemented = get_methods_implemented_in_class(AuditConnector)

        # 3. Check which base methods are missing in the implementation
        missing_methods = base_methods - audit_implemented

        assert len(missing_methods) == 0, (
            f"AuditConnector is missing {len(missing_methods)} methods from "
            f"RemoteConnector: {sorted(missing_methods)}\n"
            f"Base methods: {sorted(base_methods)}\n"
            f"Implemented methods: {sorted(audit_implemented)}"
        )

        # 4. Check all abstract methods are implemented
        abstract_methods = get_abstract_methods(RemoteConnector)
        audit_missing_abstract = []
        for method_name in abstract_methods:
            # Check if the method is actually implemented in AuditConnector
            if method_name not in audit_implemented:
                audit_missing_abstract.append(method_name)

        assert len(audit_missing_abstract) == 0, (
            f"AuditConnector has not implemented {len(audit_missing_abstract)} "
            f"abstract methods: {sorted(audit_missing_abstract)}"
        )

        # 5. Check method signatures match
        signature_mismatches = check_method_signatures(
            RemoteConnector, AuditConnector, "AuditConnector"
        )

        assert len(signature_mismatches) == 0, (
            f"AuditConnector has {len(signature_mismatches)} method signature "
            f"mismatches:\n"
            + "\n".join(
                f"  - {m['method']}: base={m['base_params']}, "
                f"audit={m['wrapper_params']}"
                for m in signature_mismatches
            )
        )

    def test_instrumented_connector_completeness(self):
        """
        Comprehensive test to verify InstrumentedRemoteConnector implements all methods
        from RemoteConnector with correct signatures.
        """
        # 1. Get all methods from base class
        base_methods = get_all_methods_from_base(RemoteConnector)

        # 2. Get methods actually implemented in InstrumentedRemoteConnector
        instrumented_implemented = get_methods_implemented_in_class(
            InstrumentedRemoteConnector
        )

        # 3. Check which base methods are missing in the implementation
        missing_methods = base_methods - instrumented_implemented

        assert len(missing_methods) == 0, (
            f"InstrumentedRemoteConnector is missing {len(missing_methods)} methods "
            f"from RemoteConnector: {sorted(missing_methods)}\n"
            f"Base methods: {sorted(base_methods)}\n"
            f"Implemented methods: {sorted(instrumented_implemented)}"
        )

        # 4. Check all abstract methods are implemented
        abstract_methods = get_abstract_methods(RemoteConnector)
        instrumented_missing_abstract = []
        for method_name in abstract_methods:
            # Check if the method is actually implemented in InstrumentedRemoteConnector
            if method_name not in instrumented_implemented:
                instrumented_missing_abstract.append(method_name)

        assert len(instrumented_missing_abstract) == 0, (
            f"InstrumentedRemoteConnector has not implemented "
            f"{len(instrumented_missing_abstract)} abstract methods: "
            f"{sorted(instrumented_missing_abstract)}"
        )

        # 5. Check method signatures match
        signature_mismatches = check_method_signatures(
            RemoteConnector, InstrumentedRemoteConnector, "InstrumentedRemoteConnector"
        )

        assert len(signature_mismatches) == 0, (
            f"InstrumentedRemoteConnector has {len(signature_mismatches)} method "
            f"signature mismatches:\n"
            + "\n".join(
                f"  - {m['method']}: base={m['base_params']}, "
                f"instrumented={m['wrapper_params']}"
                for m in signature_mismatches
            )
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
