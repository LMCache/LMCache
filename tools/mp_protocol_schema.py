#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print the current LMCache MP protocol schema as JSON."""

# Future
from __future__ import annotations

# Standard
import contextlib
import json
import sys


def main() -> None:
    with contextlib.redirect_stdout(sys.stderr):
        # First Party
        from lmcache.v1.multiprocess.protocol import get_protocol_schema

    print(json.dumps(get_protocol_schema(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
