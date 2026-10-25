# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Dict, Optional

# Third Party
import msgspec


class AsyncLookupMsg(msgspec.Struct, tag=True):  # type: ignore
    """Base class for async lookup messages"""

    def describe(self) -> str:
        return ""


class LookupRequestMsg(AsyncLookupMsg):
    """Async lookup request message from scheduler to worker"""

    lookup_id: str
    hashes: list[int]
    offsets: list[int]
    request_configs: Optional[Dict[str, str]] = None

    def describe(self) -> str:
        return (
            f"Async lookup request for lookup_id={self.lookup_id} "
            f"with {len(self.hashes)} hashes"
        )


class LookupResponseMsg(AsyncLookupMsg):
    """Async lookup response message from worker to scheduler.

    An empty ``kind`` is the legacy single response: the tokens are already
    in host memory. ``kind="lookup"`` is only the metadata hit length.
    ``kind="reading"`` means host GET has started and carries no length.
    ``kind="preloaded"`` means the body is in host memory.
    """

    lookup_id: str
    num_hit_tokens: int
    kind: str = ""

    def describe(self) -> str:
        return (
            f"Async lookup response for lookup_id={self.lookup_id} "
            f"with {self.num_hit_tokens} hit tokens"
        )


class LookupCleanupMsg(AsyncLookupMsg):
    """Cleanup message from scheduler to worker to release memory objects"""

    lookup_id: str

    def describe(self) -> str:
        return f"Cleanup memory for lookup_id={self.lookup_id}"
