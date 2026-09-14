# SPDX-License-Identifier: Apache-2.0
"""Typed httpx client for the Memory Coordinator.

The client latches the ``region_epoch`` it observes at connect time and
sends it with every request. When the coordinator restarts (new epoch), the
server rejects the stale epoch with 409 and the client *fences itself*:
every subsequent operation raises :class:`StaleEpochError` until an
explicit, operator-coordinated reset/restart constructs a new client. This
is deliberate — an MP server holding views into the old region layout must
not silently adopt a new epoch.
"""

# Standard
import threading

# Third Party
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_coordinator.api import (
    EncodedObjectKey,
    EpochResponse,
    InvalidReservationError,
    LookupHit,
    LookupRequest,
    LookupResponse,
    MemoryCoordinatorError,
    OutOfSpaceError,
    RegionContract,
    ReservationBatchRequest,
    ReservationRef,
    StaleEpochError,
    StatusResponse,
    WriteGrant,
    WriteReserveItem,
    WriteReserveRequest,
    WriteReserveResponse,
)
from lmcache.v1.memory_coordinator.config import read_token_file

logger = init_logger(__name__)

_ERROR_TYPES: dict[str, type[MemoryCoordinatorError]] = {
    "stale_epoch": StaleEpochError,
    "invalid_reservation": InvalidReservationError,
    "out_of_space": OutOfSpaceError,
}


class MemoryCoordinatorHttpClient:
    """Synchronous HTTP client that latches one coordinator epoch.

    Args:
        endpoint: Base URL of the coordinator, e.g. ``http://host:9400``.
        token_file: Absolute path to the bearer-token file.
        timeout: Per-request timeout in seconds.

    Raises:
        ValueError: The endpoint or token file is invalid.
        MemoryCoordinatorError: The coordinator rejected the first request.
        httpx.HTTPError: The coordinator is unreachable.
    """

    def __init__(
        self,
        endpoint: str,
        token_file: str,
        timeout: float = 10.0,
    ) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("coordinator endpoint must be an http(s) URL")
        token = read_token_file(token_file)
        self._http = httpx.Client(
            base_url=endpoint.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
        self._lock = threading.RLock()
        self._fenced = False
        self._closed = False
        try:
            contract = RegionContract.model_validate(self._get("/v1/region"))
        except BaseException:
            self._closed = True
            self._http.close()
            raise
        self._contract = contract
        logger.info(
            "Memory coordinator client connected: region_id=%s epoch=%s",
            contract.region_id,
            contract.region_epoch,
        )

    def region_contract(self) -> RegionContract:
        """Return the contract latched at connect time (epoch included)."""
        return self._contract

    def reserve_writes(
        self,
        items: list[WriteReserveItem],
    ) -> list[WriteGrant | None]:
        """Reserve every absent key in one capacity-atomic batch.

        Args:
            items: Keys and layouts to reserve.

        Returns:
            One entry per item; ``None`` for keys that already exist.

        Raises:
            OutOfSpaceError: The complete batch does not fit.
            StaleEpochError: The coordinator epoch changed; this client is
                now fenced.
        """
        body = WriteReserveRequest(
            region_epoch=self._contract.region_epoch,
            items=items,
        )
        payload = self._post("/v1/writes/reserve", body.model_dump())
        response = WriteReserveResponse.model_validate(payload)
        self._check_response_epoch(response.region_epoch)
        return list(response.grants)

    def finish_writes(self, reservations: list[ReservationRef]) -> None:
        """Atomically publish a write batch (all become ``VALID``)."""
        self._reservation_batch("/v1/writes/finish", reservations)

    def abort_writes(self, reservations: list[ReservationRef]) -> None:
        """Drop ``WRITING`` metadata for a failed batch."""
        self._reservation_batch("/v1/writes/abort", reservations)

    def lookup(
        self,
        keys: list[EncodedObjectKey],
    ) -> list[LookupHit | None]:
        """Return every ``VALID`` hit in one batch; misses return ``None``."""
        body = LookupRequest(
            region_epoch=self._contract.region_epoch,
            keys=keys,
        )
        payload = self._post("/v1/lookup", body.model_dump())
        response = LookupResponse.model_validate(payload)
        self._check_response_epoch(response.region_epoch)
        return list(response.hits)

    def status(self) -> StatusResponse:
        """Return the coordinator's constant-size status."""
        response = StatusResponse.model_validate(self._get("/v1/status"))
        self._check_response_epoch(response.region.region_epoch)
        return response

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(used_bytes, capacity_bytes)`` from coordinator status."""
        status = self.status()
        return status.used_bytes, status.region.capacity_bytes

    def close(self) -> None:
        """Release the transport; further operations fail."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._http.close()

    def _reservation_batch(
        self,
        path: str,
        reservations: list[ReservationRef],
    ) -> None:
        # Empty batches are still sent so every operation checks the epoch.
        body = ReservationBatchRequest(
            region_epoch=self._contract.region_epoch,
            reservations=reservations,
        )
        response = EpochResponse.model_validate(self._post(path, body.model_dump()))
        self._check_response_epoch(response.region_epoch)

    def _ensure_usable(self) -> None:
        if self._closed:
            raise MemoryCoordinatorError("memory coordinator client is closed")
        if self._fenced:
            raise StaleEpochError(
                "memory coordinator client is fenced by an epoch change; "
                "a coordinated reset/restart is required"
            )

    def _check_response_epoch(self, region_epoch: str) -> None:
        if region_epoch != self._contract.region_epoch:
            with self._lock:
                self._fenced = True
            raise StaleEpochError(
                "memory coordinator returned a different region epoch"
            )

    def _get(self, path: str) -> dict[str, object]:
        with self._lock:
            self._ensure_usable()
            response = self._http.get(path)
        return self._decode(response)

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        with self._lock:
            self._ensure_usable()
            try:
                response = self._http.post(path, json=payload)
            except httpx.HTTPError:
                # The server may have committed before the connection failed.
                self._fenced = True
                raise
            if response.status_code >= 500 and response.status_code != 507:
                self._fenced = True
        return self._decode(response)

    def _decode(self, response: httpx.Response) -> dict[str, object]:
        if response.status_code < 400:
            decoded: dict[str, object] = response.json()
            return decoded
        error = ""
        detail = ""
        try:
            body = response.json()
            error = str(body.get("error", ""))
            detail = str(body.get("detail", ""))
        except ValueError:
            detail = response.text
        if error == "stale_epoch":
            with self._lock:
                self._fenced = True
        error_type = _ERROR_TYPES.get(error, MemoryCoordinatorError)
        raise error_type(
            f"memory coordinator returned {response.status_code}: "
            f"{error or 'error'} {detail}".strip()
        )
