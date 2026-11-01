# SPDX-License-Identifier: Apache-2.0
"""Runtime for ``lmcache bench quality``: plans, arm runs, scoring, verdict.

Kept in a sub-package so that command discovery, which imports every
module directly under ``quality_bench``, imports only the lightweight
command wrappers.
"""
