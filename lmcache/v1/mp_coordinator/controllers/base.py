# SPDX-License-Identifier: Apache-2.0
"""What makes a class in this package a controller.

Discovery builds every ``Controller`` it finds here, so the marker is
what separates a controller from the collaborators one owns -- the usage
view, for instance, belongs to the eviction controller and must not be
built a second time.
"""

# Standard
from collections.abc import Sequence
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.mp_coordinator.persistence.store import DurableComponent

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


class Controller:
    """A collaborator the coordinator builds once at startup.

    Subclass to have discovery construct it (see ``build_controllers``).
    """

    @classmethod
    def from_config(cls, config: "MPCoordinatorConfig") -> "Controller":
        """Build this controller from the coordinator's configuration.

        Defaults to ignoring it, so only a controller that reads
        configuration writes this hook.

        Args:
            config: The coordinator configuration.
        """
        return cls()

    def get_durable_components(self) -> Sequence[DurableComponent]:
        """Return the state this controller needs to outlive the process.

        Empty by default, so a controller holding nothing durable says
        nothing about persistence at all. Each component it does return
        carries the ``persistence_type`` that decides where it is stored.
        """
        return ()
