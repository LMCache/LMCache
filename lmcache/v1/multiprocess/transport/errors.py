# SPDX-License-Identifier: Apache-2.0
"""Errors a request client raises when a request does not succeed."""


class MPServerUnavailableError(TimeoutError, ConnectionError):
    """The server went away before it answered a request.

    It is also a ``TimeoutError`` because over ZMQ the same failure shows up
    as a timeout, and callers catch it as one.

    Args:
        code: The gRPC status code name, such as ``"UNAVAILABLE"``.
        details: The status message.
    """

    def __init__(self, code: str, details: str) -> None:
        super().__init__(f"LMCache server unavailable ({code}): {details}")
        self.code = code
        self.details = details

    def __reduce__(self) -> tuple[type, tuple[str, str]]:
        return type(self), (self.code, self.details)


class MPRequestError(RuntimeError):
    """The server failed a request.

    Args:
        code: The gRPC status code name, such as ``"INTERNAL"``.
        details: The server's message.
    """

    def __init__(self, code: str, details: str) -> None:
        super().__init__(f"LMCache request failed ({code}): {details}")
        self.code = code
        self.details = details

    def __reduce__(self) -> tuple[type, tuple[str, str]]:
        return type(self), (self.code, self.details)
