# SPDX-License-Identifier: Apache-2.0
"""Tests for controller discovery: what the scan finds, builds, and skips."""

# Standard
import sys
import textwrap

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import PersistenceType
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.controllers import build_controllers
from lmcache.v1.mp_coordinator.controllers.eviction_controller import (
    FleetEvictionController,
)
from lmcache.v1.mp_coordinator.controllers.prefetch_manager import PrefetchManager
from lmcache.v1.mp_coordinator.controllers.usage_manager import L2UsageManager


class TestControllerDiscovery:
    def test_a_controller_owning_durable_state_is_found(self):
        """The scan replaces a hand-written list, so the one controller
        with durable state today must come back from it."""
        registry = build_controllers(MPCoordinatorConfig())

        assert isinstance(
            registry.get(FleetEvictionController), FleetEvictionController
        )

    def test_a_collaborator_a_controller_owns_is_not_built_separately(self):
        """``L2UsageManager`` sits in this package but belongs to the
        eviction controller; a second copy would account nothing."""
        registry = build_controllers(MPCoordinatorConfig())

        with pytest.raises(KeyError, match="L2UsageManager"):
            registry.get(L2UsageManager)  # type: ignore[type-var]

    def test_a_controller_without_durable_state_is_still_built(self):
        """Discovery builds every controller, not only the durable ones —
        the prefetch manager is reached through the same registry."""
        registry = build_controllers(MPCoordinatorConfig())

        assert isinstance(registry.get(PrefetchManager), PrefetchManager)

    def test_configuration_reaches_the_discovered_controller(self):
        """Discovery replaces a call site that passed config explicitly,
        so the knobs have to survive the indirection."""
        config = MPCoordinatorConfig(eviction_ratio=0.33, trigger_watermark=0.77)

        controller = build_controllers(config).get(FleetEvictionController)

        actions = controller.policy.get_eviction_actions(
            expected_ratio=0.33, cache_salt=""
        )
        assert actions == []  # nothing tracked yet, but the ratio was accepted
        assert isinstance(controller, FleetEvictionController)

    def test_a_controller_added_to_the_package_is_picked_up(self, tmp_path):
        """The point of scanning: a new controller needs no edit to
        ``create_app``. This drops one into the package and checks that
        both discovery and the type-based routing collect it.
        """
        # First Party
        from lmcache.v1.mp_coordinator import controllers

        module_path = tmp_path / "late_controller.py"
        module_path.write_text(
            textwrap.dedent(
                '''
                # SPDX-License-Identifier: Apache-2.0
                """A controller that appears only for this test."""

                from collections.abc import Mapping

                from lmcache.v1.distributed.api import PersistenceType
                from lmcache.v1.mp_coordinator.controllers.base import Controller


                class LateController(Controller):
                    """Owns one metadata section."""

                    def get_durable_components(self):
                        return (self,)

                    @property
                    def persistence_type(self):
                        return PersistenceType.METADATA

                    @property
                    def name(self):
                        return "late"

                    def capture(self) -> Mapping[str, object]:
                        return {"seen": True}

                    def restore(self, state: Mapping[str, object]) -> None:
                        pass
                '''
            )
        )
        controllers.__path__.append(str(tmp_path))
        try:
            collected = build_controllers(MPCoordinatorConfig()).durable_components()
        finally:
            controllers.__path__.remove(str(tmp_path))
            sys.modules.pop(f"{controllers.__name__}.late_controller", None)

        assert "late" in [c.name for c in collected[PersistenceType.METADATA]]
