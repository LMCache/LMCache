# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for the shared request failure contract."""

# First Party
from lmcache.v1.transport.errors import REQUEST_FAILURES as REQUEST_FAILURES
from lmcache.v1.transport.errors import MPRequestError as MPRequestError
from lmcache.v1.transport.errors import (
    MPServerUnavailableError as MPServerUnavailableError,
)
