# SPDX-License-Identifier: Apache-2.0
"""Dependencies of ``lmcache.v1.common.sliding_window``."""

# Standard
import json
import subprocess
import sys
import textwrap

_SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    import lmcache

    before = set(sys.modules)
    from lmcache.v1.common.sliding_window import recalculate_blocks_to_skip

    recalculate_blocks_to_skip(4, 2, 7)
    added = sorted(set(sys.modules) - before)
    print(json.dumps([m for m in added if m.startswith("lmcache.v1.multiprocess")]))
    """
)


def test_import_does_not_load_the_multiprocess_package() -> None:
    # `import lmcache` itself loads lmcache.v1.multiprocess.transfer_mode.
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []
