# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum, auto
import asyncio
import threading


class EventType(Enum):
    LOADING = auto()


class EventManager:
    def __init__(self) -> None:
        self.events: dict[EventType, dict[str, asyncio.Future]] = {}
        for event_type in EventType:
            self.events[event_type] = {}

        self.lock = threading.Lock()

    def add_event(
        self,
        event_type: EventType,
        event_id: str,
        future: asyncio.Future,
    ) -> None:
        with self.lock:
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
        with self.lock:
            sub_events_dict = self.events.get(event_type, None)
            assert sub_events_dict is not None, (
                f"Invalid event type {event_type} in EventManager."
            )
            event = sub_events_dict.pop(event_id, None)
            assert event is not None, (
                f"Event {event_id} of type {event_type} not found in EventManager."
            )
            return event
