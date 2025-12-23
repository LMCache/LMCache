# SPDX-License-Identifier: Apache-2.0
"""
LMCache - KV cache management for large language models.

This module automatically installs global exception handlers to capture
crash reports for debugging purposes.
"""

# Standard
import os
import sys
import traceback
import tempfile
import threading
from datetime import datetime

# Global flag to ensure crash test only runs once
_CRASH_TEST_EXECUTED = False

def _install_crash_handler():
    """
    Install global exception handlers for the main thread and background
    threads that save crash reports to a temporary directory.
    """
    # Main Thread Exception Handler
    _original_excepthook = sys.excepthook

    def lmcache_main_handler(exc_type, exc_value, exc_traceback):
        _write_crash_report(exc_type, exc_value, exc_traceback)
        _original_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = lmcache_main_handler

    # Background Thread Exception Handler
    if hasattr(threading, 'excepthook'):
        def lmcache_thread_handler(args):
            """
            Args attributes: exc_type, exc_value, exc_traceback, thread
            """
            _write_crash_report(args.exc_type, args.exc_value, args.exc_traceback)
            # Use the internal __excepthook__ to print to stderr as per default behavior
            sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)
    else:
        # Fallback for Python < 3.8:
        # Default behavior(print to stderr only
        pass

    threading.excepthook = lmcache_thread_handler


def _write_crash_report(exc_type=None, exc_value=None, exc_traceback=None, induced=False):
    """Writes detailed crash metadata to a file in the system temp directory."""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()

    # Identify which thread crashed
    thread_name = threading.current_thread().name

    crash_file = os.path.join(
        tempfile.gettempdir(),
        f'lmcache-report-{pid}.dump.{timestamp}'
    )

    try:
        # Attempt to get version info safely
        try:
            from lmcache.utils import get_version
            version_info = get_version()
        except Exception:
            version_info = "unknown"

        with open(crash_file, "w", encoding="utf-8") as f:
            f.write("LMCache Crash Report\n")
            f.write("=" * 80 + "\n")
            f.write(f"Time:           {now.isoformat()}\n")
            f.write(f"PID:            {pid}\n")
            f.write(f"Thread:         {thread_name}\n")
            f.write(f"LMCache Ver:    {version_info}\n")
            f.write("=" * 80 + "\n\n")

            if induced or exc_type is None:
                f.write("Exception Type: RuntimeError (Induced Test)\n")
                f.write("Message:        Testing LMCache crash handler\n")
            else:
                f.write(f"Exception Type: {exc_type.__name__}\n")
                f.write(f"Exception Message: {str(exc_value)}\n")
                f.write("-" * 80 + "\n")
                f.write("Full Traceback:\n")
                f.write("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))

            f.write("\n" + "=" * 80 + "\n")
            f.write("System Information:\n")
            f.write(f"Python Version: {sys.version}\n")
            f.write(f"Platform:       {sys.platform}\n")

            # Capture relevant environment variables for LLM context
            f.write("\nEnvironment:\n")
            for env_var in ["CUDA_VISIBLE_DEVICES", "LMCACHE_CONFIG_FILE"]:
                f.write(f"{env_var}: {os.environ.get(env_var, 'Not Set')}\n")

            # Try to add active engine instances
            try:
                from lmcache.cache_engine import LMCacheEngineBuilder
                instances = list(LMCacheEngineBuilder._instances.keys())
                f.write(f"\nActive LMCache Instances: {instances}\n")
            except Exception:
                pass

        sys.stderr.write(f"\n[LMCache] Crash dump created at: {crash_file}\n")

    except Exception as write_error:
        sys.stderr.write(f"[LMCache] Failed to write crash dump: {write_error}\n")


def _check_crash_test():
    """Checks for LMCACHE_CRASH_TEST environment variable to trigger a test crash."""
    global _CRASH_TEST_EXECUTED
    if _CRASH_TEST_EXECUTED:
        return
    _CRASH_TEST_EXECUTED = True

    crash_test = os.environ.get("LMCACHE_CRASH_TEST", "").lower()
    if crash_test in ("1", "true", "yes"):
        sys.stderr.write("\n⚠️  LMCACHE_CRASH_TEST detected: Triggering test report.\n")
        _write_crash_report(induced=True)


# Initialize handlers on import
_install_crash_handler()

# Run crash test if enabled
_check_crash_test()
