# SPDX-License-Identifier: Apache-2.0
"""Built-in device backend implementations for the platform abstraction.

Each direct subpackage defines a concrete ``DeviceSpec`` discovered by
``lmcache.v1.platform._device_detect``. Generic platform helpers and the torch
fallback live outside this namespace and are therefore excluded from built-in
backend discovery.
"""
