# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import os
import shutil
import subprocess
import sys


def test_source_only_lmcache_native_imports(tmp_path: Path) -> None:
    """Source-only trees should still import ``lmcache.lmcache_native``."""
    repo_root = Path(__file__).resolve().parents[2]
    staged_root = tmp_path / "source_only_pkg"
    shutil.copytree(
        repo_root / "lmcache",
        staged_root / "lmcache",
        ignore=shutil.ignore_patterns("*.so", "__pycache__"),
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(staged_root)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import lmcache.lmcache_native as lmcache_native; "
                "from lmcache.v1.gpu_connector.kv_format.detectors.base "
                "import EngineDetector; "
                "assert EngineDetector.__name__ == 'EngineDetector'; "
                "assert int(lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS) == 1; "
                "bitmap = lmcache_native.Bitmap(4, 2); "
                "assert bitmap.popcount() == 2; "
                "lmcache_native.PeriodicEventNotifier.create(interval_ms=5); "
                "assert lmcache_native.PeriodicEventNotifier.get() is not None; "
                "lmcache_native.PeriodicEventNotifier.shutdown(); "
                "print('ok')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
