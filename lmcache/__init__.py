# SPDX-License-Identifier: Apache-2.0
"""
LMCache - KV cache management for large language models.

This module automatically installs a global exception handler to capture
crash reports for debugging purposes.
"""

# Standard
from datetime import datetime
import os
import sys
import traceback


def _install_crash_handler():
    """
    Install a global exception handler that saves crash reports to
    /tmp/lmcache-report-{pid}.dump.{timestamp} before re-raising.
    """
    # Save the original exception hook
    _original_excepthook = sys.excepthook

    def lmcache_crash_handler(exc_type, exc_value, exc_traceback):
        """
        Custom exception hook that saves exception details to a crash dump file.

        Args:
            exc_type: The type of the exception
            exc_value: The exception instance
            exc_traceback: The traceback object
        """
        # Generate crash report filename
        pid = os.getpid()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        crash_file = f"/tmp/lmcache-report-{pid}.dump.{timestamp}"

        # Try to save the crash report
        try:
            # Get version info if available
            try:
                # First Party
                from lmcache.utils import get_version

                version_info = get_version()
            except Exception:
                version_info = "unknown"

            # Format the exception
            exception_text = "".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )

            # Write crash report
            with open(crash_file, "w") as f:
                f.write("LMCache Crash Report\n")
                f.write("=" * 80 + "\n")
                f.write(f"Time: {datetime.now().isoformat()}\n")
                f.write(f"PID: {pid}\n")
                f.write(f"LMCache Version: {version_info}\n")
                f.write(f"Exception Type: {exc_type.__name__}\n")
                f.write(f"Exception Message: {str(exc_value)}\n")
                f.write("=" * 80 + "\n\n")
                f.write("Full Traceback:\n")
                f.write(exception_text)
                f.write("\n" + "=" * 80 + "\n")

                # Add system information
                f.write("\nSystem Information:\n")
                f.write(f"Python Version: {sys.version}\n")
                f.write(f"Platform: {sys.platform}\n")

                # Try to add LMCache engine state if available
                try:
                    # First Party
                    from lmcache.cache_engine import LMCacheEngineBuilder

                    instances = list(LMCacheEngineBuilder._instances.keys())
                    f.write(f"Active LMCache Instances: {instances}\n")
                except Exception:
                    pass

            # Print notification to stderr
            print(
                f"\n{'=' * 80}\n"
                f"[LMCache] Unhandled exception occurred!\n"
                f"[LMCache] Crash report saved to: {crash_file}\n"
                f"{'=' * 80}\n",
                file=sys.stderr,
            )

        except Exception as write_error:
            # If we fail to write the crash report, at least try to log it
            print(
                f"[LMCache] Failed to write crash report to {crash_file}: "
                f"{write_error}",
                file=sys.stderr,
            )

        # Call the original exception hook to maintain normal Python behavior
        # (this will print the traceback and exit if appropriate)
        _original_excepthook(exc_type, exc_value, exc_traceback)

    # Install the custom exception hook
    sys.excepthook = lmcache_crash_handler


# Install the crash handler when LMCache is imported
_install_crash_handler()


# Testing utility: Allow crash injection via environment variable
def _check_crash_test():
    """
    Check if crash testing is enabled via environment variable.

    Set LMCACHE_CRASH_TEST=1 to make LMCache crash intentionally.
    This is useful for testing the crash handler.
    """
    # Only run once even if lmcache is imported multiple times
    # Use environment variable to track execution across module reloads
    if os.environ.get("_LMCACHE_CRASH_TEST_EXECUTED") == "1":
        return

    os.environ["_LMCACHE_CRASH_TEST_EXECUTED"] = "1"

    crash_test = os.environ.get("LMCACHE_CRASH_TEST", "").lower()

    if crash_test in ("1", "true", "yes"):
        crash_message = os.environ.get(
            "LMCACHE_CRASH_MESSAGE",
            "Intentional crash triggered by LMCACHE_CRASH_TEST environment variable",
        )

        print(
            f"\n{'=' * 80}\n"
            f"⚠️  LMCACHE_CRASH_TEST is enabled!\n"
            f"LMCache will crash to test the crash handler.\n"
            f"{'=' * 80}\n",
            file=sys.stderr,
        )

        # Write crash dump BEFORE raising exception
        # This ensures the dump is created even if the exception is caught
        # by vLLM's multiprocessing error handler
        pid = os.getpid()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        crash_file = f"/tmp/lmcache-report-{pid}.dump.{timestamp}"

        try:
            # Get version info if available
            try:
                # First Party
                from lmcache.utils import get_version

                version_info = get_version()
            except Exception:
                version_info = "unknown"

            # Write crash dump
            with open(crash_file, "w") as f:
                f.write("LMCache Crash Report (Test Mode)\n")
                f.write("=" * 80 + "\n")
                f.write(f"Time: {datetime.now().isoformat()}\n")
                f.write(f"PID: {pid}\n")
                f.write(f"LMCache Version: {version_info}\n")
                f.write("Exception Type: RuntimeError\n")
                f.write(f"Exception Message: {crash_message}\n")
                f.write("=" * 80 + "\n\n")
                f.write(
                    "This crash was intentionally triggered by LMCACHE_CRASH_TEST=1\n\n"
                )
                f.write("=" * 80 + "\n")
                f.write("\nSystem Information:\n")
                f.write(f"Python Version: {sys.version}\n")
                f.write(f"Platform: {sys.platform}\n")

                # Try to add LMCache engine state
                try:
                    # First Party
                    from lmcache.cache_engine import LMCacheEngineBuilder

                    instances = list(LMCacheEngineBuilder._instances.keys())
                    f.write(f"Active LMCache Instances: {instances}\n")
                except Exception:
                    pass

            print(
                f"\n[LMCache] Crash dump created at: {crash_file}\n",
                file=sys.stderr,
            )

        except Exception as write_error:
            print(
                f"[LMCache] Failed to write crash dump: {write_error}",
                file=sys.stderr,
            )

        # Now raise the exception
        raise RuntimeError(
            f"{crash_message}\n\n"
            f"This crash was intentionally triggered for testing.\n"
            f"Crash report saved to: {crash_file}\n\n"
            f"To disable this, unset the environment variable:\n"
            f"  unset LMCACHE_CRASH_TEST\n"
        )


# Run crash test if enabled
_check_crash_test()
