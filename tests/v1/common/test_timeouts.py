# SPDX-License-Identifier: Apache-2.0
"""Dependencies of ``lmcache.v1.common.timeouts``."""

# Standard
import subprocess
import sys
import textwrap


def test_import_does_not_load_mp_observability() -> None:
    script = textwrap.dedent(
        """
        import sys

        from lmcache.v1.common.timeouts import LMCacheTimeoutError

        LMCacheTimeoutError("t")
        prefix = "lmcache.v1.mp_observability"
        loaded = sorted(m for m in sys.modules if m.startswith(prefix))
        assert loaded == [], loaded
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
