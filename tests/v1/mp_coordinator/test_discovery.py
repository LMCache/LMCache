# SPDX-License-Identifier: Apache-2.0
"""Tests for discovery: what each scan finds, builds, and skips."""

# Standard
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.mp_coordinator import views as views_package
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.controllers import build_controllers
from lmcache.v1.mp_coordinator.controllers.base import Controller
from lmcache.v1.mp_coordinator.controllers.eviction_controller import (
    FleetEvictionController,
)
from lmcache.v1.mp_coordinator.controllers.prefetch_manager import PrefetchManager
from lmcache.v1.mp_coordinator.discovery import Registry
from lmcache.v1.mp_coordinator.persistence.durable_component import PersistenceType
from lmcache.v1.mp_coordinator.views import build_views
from lmcache.v1.mp_coordinator.views.base import View
from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory
from lmcache.v1.mp_coordinator.views.usage_manager import CacheUsageManager

_LATE_VIEW = '''
# SPDX-License-Identifier: Apache-2.0
"""A view that exists only for this test."""

from collections.abc import Mapping

from lmcache.v1.mp_coordinator.persistence.durable_component import PersistenceType
from lmcache.v1.mp_coordinator.views.base import View


class LateView(View):
    """Owns one checkpoint section."""

    def get_durable_components(self):
        return (self,)

    @property
    def persistence_type(self):
        return PersistenceType.CHECKPOINT

    @property
    def name(self):
        return "late"

    def capture(self) -> Mapping[str, object]:
        return {"seen": True}

    def restore(self, state: Mapping[str, object]) -> None:
        pass
'''


class TestWhatIsFound:
    def test_the_views_are_the_fleet_read_models(self):
        assert {type(v).__name__ for v in build_views(MPCoordinatorConfig()).all()} == {
            "CacheUsageManager",
            "KeyDirectory",
            "ServerConfigRegistry",
        }

    def test_the_controllers_are_the_things_that_act(self):
        views = build_views(MPCoordinatorConfig())

        built = build_controllers(MPCoordinatorConfig(), views).all()

        assert {type(c).__name__ for c in built} == {
            "FleetEvictionController",
            "PrefetchManager",
        }

    @pytest.mark.parametrize(
        ("base", "build"),
        [(View, "views"), (Controller, "controllers")],
    )
    def test_a_marker_base_is_not_itself_a_member(self, base, build):
        """Each marker lives in the package it marks and subclasses itself
        by definition; building it would put a useless instance in the
        registry and a nameless section in the checkpoint."""
        views = build_views(MPCoordinatorConfig())
        registry = (
            views
            if build == "views"
            else build_controllers(MPCoordinatorConfig(), views)
        )

        with pytest.raises(KeyError, match=base.__name__):
            registry.get(base)

    def test_a_view_added_to_the_package_is_picked_up(self, tmp_path):
        """The point of scanning: a new view needs no edit to
        ``create_app``, and the state it advertises is routed for it."""
        (tmp_path / "late_view.py").write_text(_LATE_VIEW)
        views_package.__path__.append(str(tmp_path))
        try:
            durable = build_views(MPCoordinatorConfig()).durable_components()
        finally:
            views_package.__path__.remove(str(tmp_path))
            sys.modules.pop(f"{views_package.__name__}.late_view", None)

        assert "late" in [c.name for c in durable[PersistenceType.CHECKPOINT]]


class TestConfigurationAndDependencies:
    def test_a_view_configures_itself(self):
        """Blend lookup is the directory's own business: it matches on the
        chunk size the directory keys on."""
        config = MPCoordinatorConfig(enable_blend_lookup=True, chunk_size=256)

        directory = build_views(config).get(KeyDirectory)

        assert directory.blend_stats().table_size > 0

    def test_a_controller_reads_the_registry_view_not_its_own_copy(self):
        """The eviction plan is only correct against the bytes the fleet
        actually reported; a private copy would report zero."""
        views = build_views(MPCoordinatorConfig())
        controllers = build_controllers(MPCoordinatorConfig(), views)

        controllers.get(FleetEvictionController)

        # Nothing built a second usage view along the way.
        assert views.get(CacheUsageManager) is views.get(CacheUsageManager)

    def test_each_member_is_built_once(self):
        views = build_views(MPCoordinatorConfig())

        assert views.get(KeyDirectory) is views.get(KeyDirectory)

    def test_a_dependency_cycle_is_reported_not_recursed(self):
        """Left alone this is a RecursionError deep inside discovery, with
        nothing naming the classes involved."""

        class _First(View):
            @classmethod
            def from_config(cls, config, views):
                views.get(_Second)
                return cls()

        class _Second(View):
            @classmethod
            def from_config(cls, config, views):
                views.get(_First)
                return cls()

        registry: Registry[View] = Registry(
            [_First, _Second],
            build=lambda t, r: t.from_config(MPCoordinatorConfig(), r),
        )

        with pytest.raises(ValueError, match="depends on itself"):
            registry.get(_First)


class TestRegistry:
    def test_an_undiscovered_class_says_so(self):
        class _Absent(View):
            pass

        registry: Registry[View] = Registry([], build=lambda t, r: t())

        with pytest.raises(KeyError, match="_Absent"):
            registry.get(_Absent)

    def test_a_duplicate_class_is_refused(self):
        """Callers address members by type, so a duplicate has no
        unambiguous answer."""
        twice = [PrefetchManager, PrefetchManager]

        with pytest.raises(ValueError, match="duplicate"):
            Registry(twice, build=lambda t, r: t())
