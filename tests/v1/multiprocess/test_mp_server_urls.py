# SPDX-License-Identifier: Apache-2.0
"""Tests for server-URL normalization in the multiprocess vLLM adapters."""


def test_normalize_server_urls_wraps_a_single_url_string():
    """A bare URL string must stay one URL, not become one URL per character."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        _normalize_server_urls,
    )

    url = "tcp://127.0.0.1:5555"
    assert _normalize_server_urls(url) == [url]


def test_normalize_server_urls_keeps_sequences():
    """Sequences of URLs are returned as an equivalent list."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        _normalize_server_urls,
    )

    urls = ["tcp://127.0.0.1:5555", "tcp://127.0.0.1:5556"]
    assert _normalize_server_urls(urls) == urls
    assert _normalize_server_urls(tuple(urls)) == urls
