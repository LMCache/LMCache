# SPDX-License-Identifier: Apache-2.0

"""MP observability subscriber packages.

Import concrete subscribers from their concern-specific packages
(``subscribers.metrics``, ``subscribers.logging``, or
``subscribers.tracing``) so loading a logging-only path does not import
native-extension-backed metrics modules.
"""
