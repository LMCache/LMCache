# SPDX-License-Identifier: Apache-2.0
"""Request-owned read leases for the private TP1 retained-view path."""

from contextlib import ExitStack
import threading


class RetainedReadLocks:
    def __init__(self):
        self._lock = threading.Lock()
        self._readers = {}

    def reserve_prefix(self, store, session_id, keys):
        """Acquire only a contiguous prefix. A miss ends the lookup."""
        leases = {}
        with self._lock:
            try:
                for key in keys:
                    if key in leases:
                        raise ValueError("duplicate retained chunk key")
                    try:
                        lease = store.acquire(key)
                        view = lease.__enter__()
                    except KeyError:
                        break
                    leases[key] = (lease, view)
                if session_id in self._readers:
                    self._release(self._readers.pop(session_id))
                if leases:
                    self._readers[session_id] = leases
                return len(leases)
            except BaseException:
                self._release(leases)
                raise

    def claim(self, session_id, keys):
        """Transfer leases to the synchronous consumer before GPU work starts."""
        stack = ExitStack()
        with self._lock:
            leases = self._readers.get(session_id, {})
            if len(set(keys)) != len(keys) or any(k not in leases for k in keys):
                raise KeyError("retained RETRIEVE without matching LOOKUP read locks")
            views = []
            for key in keys:
                cm, view = leases.pop(key)
                stack.push(cm)
                views.append(view)
            if not leases:
                self._readers.pop(session_id, None)
        return stack, views

    @staticmethod
    def _release(leases):
        with ExitStack() as stack:
            for cm, _ in leases.values():
                stack.push(cm)

    def cancel(self, session_id, keys=None):
        with self._lock:
            leases = self._readers.get(session_id, {})
            selected = (
                dict(leases)
                if keys is None
                else {k: leases[k] for k in keys if k in leases}
            )
            for k in selected:
                del leases[k]
            if not leases:
                self._readers.pop(session_id, None)
        self._release(selected)

    def close(self):
        with self._lock:
            readers, self._readers = self._readers, {}
        for leases in readers.values():
            self._release(leases)
