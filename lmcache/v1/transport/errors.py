# SPDX-License-Identifier: Apache-2.0
"""Handler rejection contract shared by request transports."""


class InvalidRequestError(ValueError):
    """A validated request was refused without changing handler state."""
