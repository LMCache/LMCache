# SPDX-License-Identifier: Apache-2.0
"""Track one recurrent checkpoint transfer across its device workers."""


class RecurrentCopyProgress:
    """Track whether a read reservation is safe to release and publish.

    The caller serializes access with its reservation lock.

    Args:
        ranks: Worker ranks that must copy the complete checkpoint set.
    """

    def __init__(self, ranks: set[int]) -> None:
        self._pending = ranks.copy()
        self._running: set[int] = set()
        self._failed = False
        self._result_taken = False

    def start(self, rank: int) -> bool:
        """Claim a pending copy for rank, returning whether it may start."""
        if rank not in self._pending:
            return False
        self._pending.remove(rank)
        self._running.add(rank)
        return True

    def finish(self, rank: int, succeeded: bool) -> None:
        """Record the outcome for rank after its device stream is fenced."""
        if rank in self._running:
            self._running.remove(rank)
            self._failed |= not succeeded

    def cancel(self) -> None:
        """Stop unstarted copies. Running copies still require a fence."""
        self._pending.clear()
        self._failed = True

    def take_result(self) -> bool | None:
        """Return the publish decision once, when releasing memory is safe.

        Returns:
            True if every copy succeeded, False if canceled or any copy
            failed, or None while copies are pending or already settled.
        """
        if self._pending or self._running or self._result_taken:
            return None
        self._result_taken = True
        return not self._failed
