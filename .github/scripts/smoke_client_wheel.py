# SPDX-License-Identifier: Apache-2.0
"""Check the installed client wheel outside the source checkout.

Run with the clean installation's interpreter and -I to prevent checkout imports.
"""

# Standard
from importlib import metadata
from pathlib import Path

# Third Party
import torch  # noqa: F401 -- native extensions require torch to be loaded first

# First Party
from lmcache.v1.client.transport.factory import RequestClientFactory
import lmcache
import lmcache.lmcache_native  # noqa: F401 -- verify the compiled extension loads


def main() -> None:
    """Verify wheel ownership and create an installed request client."""
    distribution = metadata.distribution("lmcache-client")
    init = distribution.locate_file("lmcache/__init__.py")
    assert Path(lmcache.__file__).samefile(str(init)), lmcache.__file__
    RequestClientFactory.create("127.0.0.1:1").close()
    print("Client wheel smoke passed:", distribution.version, lmcache.__file__)


if __name__ == "__main__":
    main()
