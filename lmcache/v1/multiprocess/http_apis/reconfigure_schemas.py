# SPDX-License-Identifier: Apache-2.0
"""Shared wire vocabulary for the ``/reconfigure`` endpoint family.

Human-friendly size parsing, the family's error-response helpers, and engine
lookup, shared by the family's route modules so every tier follows one wire
contract.

The family error contract implemented by these helpers:

- request-schema violations -> ``422 {"detail": [...]}`` (the same outer
  response envelope FastAPI's own validation produces);
- value errors on well-formed fields (size strings) -> ``400 {"error": ...}``;
- domain errors -> the status carried by the raised reconfigure error
  (404 lookup miss, 409 state conflict, ...), body ``{"error": ...}`` unless
  the error supplies its own payload.
"""

# Standard
from collections.abc import Mapping
from decimal import Decimal
from typing import Protocol, TypeAlias

# Third Party
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import StrictInt, ValidationError

SIZE_ERROR = "size must be a positive integer byte count or a string like '100GiB'"
# Strict on the wire: JSON booleans and floats are schema violations (422),
# never silently coerced into byte counts.
SizeRequest: TypeAlias = StrictInt | str

_MAX_SIZE_STRING_LEN = 64
_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "kib": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "mib": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
    "gib": 1024**3,
    "t": 1024**4,
    "tb": 1024**4,
    "tib": 1024**4,
}


class _ReconfigureErrorLike(Protocol):
    """Shape shared by the tiers' HTTP-mappable reconfiguration errors."""

    @property
    def status_code(self) -> int:
        """Return the HTTP status carried by the domain error."""
        ...

    @property
    def payload(self) -> Mapping[str, object]:
        """Return the JSON object carried by the domain error."""
        ...


def parse_size_string(size: str) -> int:
    """Parse a human-friendly size string into a byte count.

    Args:
        size: A size such as ``"100GiB"``, ``"512mb"``, or ``"1048576"``.
            Decimal values (``"1.5GiB"``) are accepted; the unit suffix is
            case-insensitive and binary (1024-based).

    Returns:
        The size in bytes.

    Raises:
        ValueError: If the string cannot be parsed or is not positive.
    """
    text = size.strip()
    if not text or len(text) > _MAX_SIZE_STRING_LEN:
        raise ValueError(SIZE_ERROR)

    unit_start = len(text)
    while unit_start > 0 and text[unit_start - 1].isalpha():
        unit_start -= 1

    value_text = text[:unit_start].strip()
    unit = text[unit_start:].lower()
    if unit not in _SIZE_UNITS:
        raise ValueError(SIZE_ERROR)
    if "." in value_text:
        whole, fraction = value_text.split(".", 1)
        if not whole or not fraction:
            raise ValueError(SIZE_ERROR)
        if not whole.isdigit() or not fraction.isdigit():
            raise ValueError(SIZE_ERROR)
    elif not value_text.isdigit():
        raise ValueError(SIZE_ERROR)

    value = Decimal(value_text)
    if value <= 0:
        raise ValueError(SIZE_ERROR)
    return int(value * _SIZE_UNITS[unit])


def resolve_size_bytes(size: SizeRequest) -> int:
    """Resolve a wire-level size value into a positive byte count.

    Args:
        size: A positive integer byte count or a size string accepted by
            :func:`parse_size_string`.

    Returns:
        The size in bytes.

    Raises:
        ValueError: If the value is not a positive size.
    """
    if isinstance(size, bool):
        raise ValueError(SIZE_ERROR)
    resolved = size if isinstance(size, int) else parse_size_string(size)
    if resolved <= 0:
        raise ValueError(SIZE_ERROR)
    return resolved


def validation_error_response(exc: ValidationError) -> JSONResponse:
    """Return the family's 422 response for a request-schema violation.

    Args:
        exc: The validation error raised by a request model.

    Returns:
        A 422 response whose body is ``{"detail": [...]}`` -- the same outer
        response envelope FastAPI's own request validation produces.
    """
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


def reconfigure_error_response(exc: _ReconfigureErrorLike) -> JSONResponse:
    """Return the response for an HTTP-mappable reconfiguration error.

    Args:
        exc: A domain error carrying an HTTP ``status_code`` and a JSON
            ``payload``.

    Returns:
        A response with the error's status code and its payload as the body.
    """
    return JSONResponse(status_code=exc.status_code, content=exc.payload)


def get_engine_from_request(request: Request) -> object | None:
    """Return the app's engine, or ``None`` before engine initialization.

    Args:
        request: The FastAPI request whose app state holds the engine.

    Returns:
        The engine object, or ``None`` while the server is still starting up.
    """
    return getattr(request.app.state, "engine", None)
