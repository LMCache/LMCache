# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum, auto
import asyncio
import threading


class EventType(Enum):
    LOADING = auto()


class EventStatus(Enum):
    ONGOING = auto()
    DONE = auto()
    NOT_FOUND = auto()


class EventManager:
    """
    A thread-safe event manager to manage asynchronous events.
    Each event is identified by its type and a unique id.
    """

    def __init__(self) -> None:
        self.events: dict[EventType, dict[str, tuple[EventStatus, asyncio.Future]]] = {}
        for event_type in EventType:
            self.events[event_type] = {}

        self.lock = threading.Lock()

    def add_event(
        self,
        event_type: EventType,
        event_id: str,
        future: asyncio.Future,
    ) -> None:
        """
        Add an event with the given type and id.
        """
        with self.lock:
            sub_events_dict = self.events.get(event_type, None)
            assert sub_events_dict is not None, (
                f"Invalid event type {event_type} in EventManager."
            )
            sub_events_dict[event_id] = (EventStatus.ONGOING, future)

    def pop_event(
        self,
        event_type: EventType,
        event_id: str,
    ) -> asyncio.Future:
        """
        Pop and return the event with the given type and id.
        """
        with self.lock:
            sub_events_dict = self.events.get(event_type, None)
            assert sub_events_dict is not None, (
                f"Invalid event type {event_type} in EventManager."
            )
            assert event_id in sub_events_dict, (
                f"Event {event_id} of type {event_type} not found in EventManager."
            )
            status, event = sub_events_dict.pop(event_id)
            assert status == EventStatus.DONE, (
                f"Event {event_id} of type {event_type} is not done."
            )
            assert event is not None, (
                f"Event {event_id} of type {event_type} not found in EventManager."
            )
            return event

    def update_event_status(
        self,
        event_type: EventType,
        event_id: str,
        status: EventStatus,
    ) -> None:
        """
        Update the status of the event with the given type and id.
        """
        with self.lock:
            sub_events_dict = self.events.get(event_type, None)
            assert sub_events_dict is not None, (
                f"Invalid event type {event_type} in EventManager."
            )
            if event_id in sub_events_dict:
                _, event = sub_events_dict[event_id]
                sub_events_dict[event_id] = (status, event)
            else:
                raise KeyError(f"Event {event_id} of type {event_type} not found.")

    def get_event_status(
        self,
        event_type: EventType,
        event_id: str,
    ) -> EventStatus:
        """
        Get the status of the event with the given type and id.
        """
        with self.lock:
            sub_events_dict = self.events.get(event_type, None)
            assert sub_events_dict is not None, (
                f"Invalid event type {event_type} in EventManager."
            )
            if event_id in sub_events_dict:
                status, _ = sub_events_dict[event_id]
                return status
            else:
                return EventStatus.NOT_FOUND
