# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum, auto
import asyncio


class EventType(Enum):
    LOADING = auto()


class EventManager:
    def __init__(self) -> None:
        self.events = {}
        for event_type in EventType:
            self.events[event_type] = {}

        # TODO(Jiayi): verify thread safety if needed

    def add_event(
        self,
        event_type: EventType,
        event_id: str,
        future: asyncio.Future,
    ) -> None:
        sub_events_dict = self.events.get(event_type, None)
        assert sub_events_dict is not None, (
            f"Invalid event type {event_type} in EventManager."
        )
        sub_events_dict[event_id] = future

    def pop_event(
        self,
        event_type: EventType,
        event_id: str,
    ) -> asyncio.Future:
        sub_events_dict = self.events.get(event_type, None)
        assert sub_events_dict is not None, (
            f"Invalid event type {event_type} in EventManager."
        )
        return sub_events_dict.pop(event_id, [])
