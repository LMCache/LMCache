# SPDX-License-Identifier: Apache-2.0
"""Marker base for platform base classes.

A class under ``platform/base/`` is treated as a platform base class iff
it subclasses :class:`PlatformBase`.  This module intentionally contains
no imports or logic beyond the marker class so importing it cannot create
cycles or heavy side effects.
"""


class PlatformBase:
    """Empty marker base.

    A class under ``platform/base/`` is treated as a platform base class
    iff it subclasses :class:`PlatformBase`.  Carries no logic.
    """
