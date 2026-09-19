# SPDX-License-Identifier: Apache-2.0
"""HTTP metadata client; epoch changes and ambiguous writes fence further use."""

# Standard
import threading

# Third Party
import httpx

# First Party
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

_ERROR_TYPES: dict[str, type[MemoryCoordinatorError]] = {
    "stale_epoch": StaleEpochError,
    "invalid_reservation": InvalidReservationError,
    "out_of_space": OutOfSpaceError,
}


class MemoryCoordinatorHttpClient:
    """Connect to HTTP(S) ``endpoint`` with absolute ``token_file`` credentials.

    ``timeout`` is in seconds. Invalid settings raise ValueError; transport
    failures raise httpx.HTTPError; rejected operations raise
    MemoryCoordinatorError. Epoch changes or ambiguous POSTs permanently fence
    the client: subsequent operations raise StaleEpochError until a coordinated
    reset constructs a new client. A closed client raises MemoryCoordinatorError.
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
        try:
            self._contract = RegionContract.model_validate(
                self._request("GET", "/v1/region")
            )
        except BaseException:
            self._http.close()
            raise

    def region_contract(self) -> RegionContract:
        """Return the contract latched at connect time (epoch included)."""
        return self._contract

    def reserve_writes(
        self,
        items: list[WriteReserveItem],
    ) -> list[WriteGrant | None]:
        """Return one grant per key/layout in ``items``, None for existing keys.

        OutOfSpaceError rejects the whole batch; other errors follow the client
        contract. Reservation and capacity checks are atomic.
        """
        body = WriteReserveRequest(
            region_epoch=self._contract.region_epoch,
            items=items,
        )
        payload = self._request("POST", "/v1/writes/reserve", body.model_dump())
        response = WriteReserveResponse.model_validate(payload)
        self._check_response_epoch(response.region_epoch)
        return response.grants

    def finish_writes(self, reservations: list[ReservationRef]) -> None:
        """Publish ``reservations`` atomically; invalid tokens raise errors."""
        self._reservation_batch("/v1/writes/finish", reservations)

    def abort_writes(self, reservations: list[ReservationRef]) -> None:
        """Abort ``reservations`` without reclaiming; invalid tokens raise errors."""
        self._reservation_batch("/v1/writes/abort", reservations)

    def lookup(
        self,
        keys: list[EncodedObjectKey],
    ) -> list[LookupHit | None]:
        """Return VALID hits for ``keys`` (None for misses); client errors apply."""
        body = LookupRequest(
            region_epoch=self._contract.region_epoch,
            keys=keys,
        )
        payload = self._request("POST", "/v1/lookup", body.model_dump())
        response = LookupResponse.model_validate(payload)
        self._check_response_epoch(response.region_epoch)
        return response.hits

    def status(self) -> StatusResponse:
        """Return the coordinator's constant-size status."""
        response = StatusResponse.model_validate(self._request("GET", "/v1/status"))
        self._check_response_epoch(response.region.region_epoch)
        return response

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(used_bytes, capacity_bytes)`` from coordinator status."""
        status = self.status()
        return status.used_bytes, status.region.capacity_bytes

    def close(self) -> None:
        """Release the transport; further operations fail."""
        with self._lock:
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
        response = EpochResponse.model_validate(
            self._request("POST", path, body.model_dump())
        )
        self._check_response_epoch(response.region_epoch)

    def _ensure_usable(self) -> None:
        if self._http.is_closed:
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

    def _request(
        self, method: str, path: str, payload: dict[str, object] | None = None
    ) -> dict[str, object]:
        with self._lock:
            self._ensure_usable()
            try:
                response = self._http.request(method, path, json=payload)
            except httpx.HTTPError:
                # The server may have committed before the connection failed.
                self._fenced |= method == "POST"
                raise
            if (
                method == "POST"
                and response.status_code >= 500
                and response.status_code != 507
            ):
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
