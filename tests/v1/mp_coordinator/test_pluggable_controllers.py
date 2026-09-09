# SPDX-License-Identifier: Apache-2.0
"""A controller the coordinator has never heard of, doing everything.

``create_app`` names no controller, so nothing here imports one either:
each test drops a controller in at runtime and checks the coordinator
runs its loop, mounts its endpoints, and finds it when it ships outside
the tree entirely.
"""

# Standard
import sys

# Third Party
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.mp_coordinator import controllers as controllers_package
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.controllers import build_controllers
from lmcache.v1.mp_coordinator.controllers.base import ControllerRuntime
from lmcache.v1.mp_coordinator.controllers.prefetch_manager import PrefetchManager
from lmcache.v1.mp_coordinator.views import build_views

_OUT_OF_TREE_ROUTES = '''
# SPDX-License-Identifier: Apache-2.0
"""The endpoints of a controller that ships outside lmcache."""

from fastapi import APIRouter


def build_router(controller):
    router = APIRouter()

    @router.get("/acme/widget")
    async def widget() -> dict:
        return {"source": controller.source}

    return (router,)
'''

_OUT_OF_TREE = '''
# SPDX-License-Identifier: Apache-2.0
"""A controller that ships outside lmcache entirely."""

from lmcache.v1.mp_coordinator.controllers.base import Controller

from acme_controllers.widget.http_api import build_router


class WidgetController(Controller):
    def __init__(self):
        self.source = "out of tree"

    def get_routers(self):
        return build_router(self)
'''

_FAILING_CONTROLLER = '''
# SPDX-License-Identifier: Apache-2.0
"""A controller whose startup fails, to check nothing is left running."""

from contextlib import asynccontextmanager

from lmcache.v1.mp_coordinator.controllers.base import Controller


class ZzzFailingController(Controller):
    """Named to sort last, so a well-behaved controller starts first."""

    @asynccontextmanager
    async def run(self, runtime):
        raise RuntimeError("this controller cannot start")
        yield
'''

_SETTLING_CONTROLLER = '''
# SPDX-License-Identifier: Apache-2.0
"""A controller that records when it settles and when it is captured."""

import asyncio
from contextlib import asynccontextmanager

from lmcache.v1.mp_coordinator.controllers.base import Controller
from lmcache.v1.mp_coordinator.persistence.durable_component import PersistenceType

ORDER = []


class SettlingController(Controller):
    """Appends to ORDER so a test can see teardown against capture."""

    @asynccontextmanager
    async def run(self, runtime):
        try:
            yield
        finally:
            ORDER.append("tearing-down")
            # Long enough that a live checkpoint timer would fire here.
            await asyncio.sleep(0.2)
            # A draining controller is still dispatching, so the client
            # it was handed must outlive its teardown.
            ORDER.append("closed-client" if runtime.http_client.is_closed
                         else "settled")

    def get_durable_components(self):
        return (self,)

    @property
    def persistence_type(self):
        return PersistenceType.CHECKPOINT

    @property
    def name(self):
        return "settling"

    def capture(self):
        ORDER.append("captured")
        return {}

    def restore(self, state):
        pass
'''

_LATE_CONTROLLER = '''
# SPDX-License-Identifier: Apache-2.0
"""A controller that exists only for this test."""

from contextlib import asynccontextmanager

from fastapi import APIRouter

from lmcache.v1.mp_coordinator.controllers.base import Controller

CALLS = []


class LateController(Controller):
    """Records what the lifespan does to it, and the runtime it got."""

    def __init__(self):
        self.greeting = "hello from a controller nothing imports"

    @asynccontextmanager
    async def run(self, runtime):
        CALLS.append(("start", runtime.http_client))
        try:
            yield
        finally:
            CALLS.append(("stop", None))

    def get_routers(self):
        router = APIRouter()

        @router.get("/late/greeting")
        async def greeting() -> dict:
            # Closes over the controller: no registry lookup by class.
            return {"greeting": self.greeting}

        return (router,)
'''


@pytest.fixture
def late_controller(tmp_path):
    """Drop a controller into the package for one test, then take it out.

    Yields the module name, so a test reads what the controller recorded
    without importing something that does not exist at collection time.
    """
    (tmp_path / "late_controller.py").write_text(_LATE_CONTROLLER)
    module_name = f"{controllers_package.__name__}.late_controller"
    controllers_package.__path__.append(str(tmp_path))
    try:
        yield module_name
    finally:
        controllers_package.__path__.remove(str(tmp_path))
        sys.modules.pop(module_name, None)


def _config() -> MPCoordinatorConfig:
    return MPCoordinatorConfig(health_check_interval=0.0, eviction_check_interval=0.0)


def test_a_controller_with_a_loop_is_started_and_stopped(late_controller):
    """The point of the protocol: a controller nothing imports still runs."""
    with TestClient(create_app(_config())):
        calls = sys.modules[late_controller].CALLS
        assert [name for name, _ in calls] == ["start"]
        assert calls[0][1] is not None, "start got no client"

    assert [name for name, _ in sys.modules[late_controller].CALLS] == [
        "start",
        "stop",
    ]


def test_the_client_is_the_one_the_app_uses(late_controller):
    """A controller must dispatch on the app's client, not one of its own:
    the lifespan closes that client, and an outbound call on a closed one
    fails at shutdown."""
    app = create_app(_config())
    with TestClient(app):
        assert sys.modules[late_controller].CALLS[0][1] is app.state.outbound_client


@pytest.mark.asyncio
async def test_a_controller_with_no_background_work_writes_nothing():
    """Prefetch answers requests and nothing else. Lifetime is part of
    every controller's interface, so the default has to be a safe no-op
    rather than something each such controller reimplements."""
    prefetch = build_controllers(_config(), build_views(_config())).get(PrefetchManager)

    async with prefetch.run(ControllerRuntime(http_client=None)):
        pass


def test_a_controller_mounts_its_own_endpoints(late_controller):
    """The route arrives with the thing it operates on, and the handler
    reaches that thing directly rather than resolving a class name."""
    with TestClient(create_app(_config())) as client:
        response = client.get("/late/greeting")

    assert response.status_code == 200
    assert response.json() == {"greeting": "hello from a controller nothing imports"}


def test_a_controller_shipped_outside_the_tree_is_loaded_by_name(tmp_path, monkeypatch):
    """The vLLM-style path: nothing is dropped into lmcache's directories,
    the operator names an importable package and the coordinator scans it."""
    package = tmp_path / "acme_controllers"
    # A directory, not a file: a controller with routes of its own is two
    # modules, and neither is named anywhere.
    widget = package / "widget"
    widget.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (widget / "__init__.py").write_text("")
    (widget / "http_api.py").write_text(_OUT_OF_TREE_ROUTES)
    (widget / "controller.py").write_text(_OUT_OF_TREE)
    monkeypatch.syspath_prepend(str(tmp_path))

    config = MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
        extra_config={"controller_packages": ["acme_controllers"]},
    )
    try:
        with TestClient(create_app(config)) as client:
            response = client.get("/acme/widget")
    finally:
        for name in list(sys.modules):
            if name.startswith("acme_controllers"):
                del sys.modules[name]

    assert response.status_code == 200
    assert response.json() == {"source": "out of tree"}


@pytest.mark.parametrize("value", ["acme_controllers", 42, {"acme": 1}, ["acme", 7]])
def test_a_malformed_package_list_is_refused(value):
    """The names arrive as JSON on the command line, so their shape is the
    operator's to get wrong. A bare string is the likely slip -- it would
    otherwise iterate into one package per character."""
    config = MPCoordinatorConfig(extra_config={"controller_packages": value})

    with pytest.raises(ValueError, match="controller_packages"):
        build_controllers(config, build_views(config))


def test_a_package_that_does_not_import_is_reported():
    """An operator asked for it by name, so a silent skip would look like
    a controller that loaded and did nothing."""
    config = MPCoordinatorConfig(
        extra_config={"controller_packages": ["no_such_package_here"]}
    )

    with pytest.raises(ModuleNotFoundError, match="no_such_package_here"):
        build_controllers(config, build_views(config))


def test_a_controller_failing_to_start_does_not_take_the_others_down(
    late_controller, tmp_path, monkeypatch
):
    """A broken controller -- likely a third-party one, since any package
    can supply them -- must not cost the coordinator the endpoints that
    belong to no controller, nor the controllers that did start."""
    # A directory of its own: ``late_controller`` has put ``tmp_path`` on
    # the controllers package's ``__path__``, so a package written there
    # would be found twice -- once by the scan, once by name.
    external = tmp_path / "external"
    package = external / "acme_failing"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "boom.py").write_text(_FAILING_CONTROLLER)
    monkeypatch.syspath_prepend(str(external))

    config = MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
        extra_config={"controller_packages": ["acme_failing"]},
    )
    try:
        with TestClient(create_app(config)) as client:
            # An endpoint owned by no controller, and one owned by the
            # controller that did start.
            fleet = client.get("/instances")
            greeting = client.get("/late/greeting")
        calls = [name for name, _ in sys.modules[late_controller].CALLS]
    finally:
        for name in list(sys.modules):
            if name.startswith("acme_failing"):
                del sys.modules[name]

    assert fleet.status_code == 200
    assert greeting.status_code == 200
    # The healthy one ran for the app's lifetime and was torn down.
    assert calls == ["start", "stop"]


def test_the_final_checkpoint_captures_what_a_controller_settled_on(
    tmp_path, monkeypatch
):
    """Teardown order is load-bearing three times over: a timer still
    firing would snapshot a controller mid-teardown, a controller still
    sweeping when the last checkpoint is taken would have a mid-sweep
    state written, and one draining after the client closed could not
    finish the dispatch it launched."""
    package = tmp_path / "acme_settling"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "c.py").write_text(_SETTLING_CONTROLLER)
    monkeypatch.syspath_prepend(str(tmp_path))

    config = MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
        # A live timer, so the test can see whether it is still firing
        # while the controller tears down.
        checkpoint_interval=0.01,
        checkpoint_path=str(tmp_path / "ckpt"),
        extra_config={"controller_packages": ["acme_settling"]},
    )
    try:
        with TestClient(create_app(config)):
            pass
        order = sys.modules["acme_settling.c"].ORDER
    finally:
        for name in list(sys.modules):
            if name.startswith("acme_settling"):
                del sys.modules[name]

    # The timer is cancelled before teardown begins, so nothing is
    # captured while it runs. The final write comes after it, not during.
    during_teardown = order[order.index("tearing-down") : order.index("settled")]
    assert "captured" not in during_teardown, order
    assert order[-2:] == ["settled", "captured"], order
