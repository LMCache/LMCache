# SPDX-License-Identifier: Apache-2.0
"""
Guard tests that verify vLLM interfaces LMCache depends on still exist.

These are intentionally simple import-and-call checks. If vLLM renames or
removes an API we rely on, these tests will fail in CI and alert us before
the breakage reaches production.
"""

# Third Party


def test_get_kv_cache_layout_importable():
    """get_kv_cache_layout must exist and be callable."""
    # Third Party
    from vllm.v1.attention.backends.utils import get_kv_cache_layout

    result = get_kv_cache_layout()
    assert result is None or result in ("NHD", "HND"), (
        f"get_kv_cache_layout() returned unexpected value: {result!r}"
    )
