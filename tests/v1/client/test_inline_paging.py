# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from typing import Any, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.client.transfer_context import inline
from lmcache.v1.client.transfer_context.inline import EngineDrivenContextInline
from lmcache.v1.common.custom_types import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
)
from lmcache.v1.transport.futures import MessagingFuture
from lmcache.v1.transport.tensor_wire import (
    decode_grouped_tensors,
    decode_tensors,
    encode_grouped_tensors,
    encode_tensors,
)

SHAPE = (4, 8)
CHUNK = [torch.full(SHAPE, float(i)) for i in range(5)]


class _Client:
    """Records pages and answers retrieve pages from ``chunks``."""

    def __init__(
        self,
        groups: list[tuple[list[int], list[torch.Tensor]]] | None,
        store_context: dict[str, Any] | None = None,
    ):
        self.groups = groups
        self.store_context = store_context or {}
        self.pages: list[tuple[int, int, bytes]] = []

    def prepare_store(self, _key, _iid) -> MessagingFuture:
        return _done(PrepareStoreResponse(context=self.store_context))

    def commit_store(self, _key, _iid, data, start, count) -> MessagingFuture:
        self.pages.append((start, count, data))
        return _done(True)

    def prepare_retrieve(self, _key, _iid, start, count) -> MessagingFuture:
        self.pages.append((start, count, b""))
        end = start + count if count else 5
        if self.groups is None:
            data = encode_tensors(CHUNK[start:end])
        else:
            data = encode_grouped_tensors(
                [
                    (
                        [i for i in ix if start <= i < end],
                        [t for i, t in zip(ix, ts, strict=True) if start <= i < end],
                    )
                    for ix, ts in self.groups
                ]
            )
        return _done(PrepareRetrieveResponse(success=True, data=data))


def _done(value: Any) -> MessagingFuture:
    future: MessagingFuture = MessagingFuture()
    future.set_result(value)
    return future


def _ctx(client: _Client) -> EngineDrivenContextInline:
    layout = SimpleNamespace(shapes=[torch.Size(SHAPE)], dtypes=[torch.float32])
    metadata = SimpleNamespace(layout_desc=layout)
    return EngineDrivenContextInline(cast(Any, metadata), cast(Any, client), 1.0)


@pytest.fixture(autouse=True)
def two_chunk_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inline, "INLINE_PAGE_BYTES", 2 * CHUNK[0].nbytes)


def test_store_is_sent_in_pages() -> None:
    client = _Client(None)
    assert _ctx(client).commit_store(cast(Any, None), 1, CHUNK)
    assert [(s, c) for s, c, _ in client.pages] == [(0, 2), (2, 2), (4, 0)]
    sent = [
        decode_tensors(
            data,
            expected_shape=torch.Size(SHAPE),
            expected_dtype=torch.float32,
            expected_count=n,
        )
        for (_, _, data), n in zip(client.pages, (2, 2, 1), strict=True)
    ]
    assert all(torch.equal(a, b) for a, b in zip(sum(sent, []), CHUNK, strict=True))

    # A grouped store pages by chunk position; the last page runs to the end
    # even when its trailing chunks were skipped.
    client = _Client(None)
    groups = [([0, 1, 2], CHUNK[:3]), ([2], CHUNK[2:3])]
    assert _ctx(client).commit_store(cast(Any, None), 1, groups)
    assert [(s, c) for s, c, _ in client.pages] == [(0, 2), (2, 0)]
    expected = [(torch.Size(SHAPE), torch.float32, 5)] * 2
    last = decode_grouped_tensors(client.pages[1][2], expected=expected)
    assert [ix for ix, _ in last] == [[2], [2]]


def _sent(client: _Client, counts: tuple[int, ...]) -> list[torch.Tensor]:
    """Decode the flat pages ``client`` received; ``counts`` per page."""
    return sum(
        (
            decode_tensors(
                data,
                expected_shape=torch.Size(SHAPE),
                expected_dtype=torch.float32,
                expected_count=n,
            )
            for (_, _, data), n in zip(client.pages, counts, strict=True)
        ),
        [],
    )


def test_prepare_store_returns_the_chunks_the_server_names() -> None:
    named = _ctx(_Client(None, {"chunk_indices": [1, 3]}))
    assert named.prepare_store(cast(Any, None), 1) == (None, [1, 3])
    # A server that names no chunks gets every chunk, as before.
    assert _ctx(_Client(None)).prepare_store(cast(Any, None), 1) is None


def test_store_sends_only_the_named_chunks_in_runs() -> None:
    """A flat page carries exactly the chunks of its range, so a gap in the
    named positions ends a page as the page size does."""
    client = _Client(None)
    chunks = CHUNK[:3]
    assert _ctx(client).commit_store(
        cast(Any, None), 1, chunks, chunk_indices=[1, 2, 4], num_chunks=5
    )
    assert [(s, c) for s, c, _ in client.pages] == [(1, 2), (4, 0)]
    assert all(
        torch.equal(a, b) for a, b in zip(_sent(client, (2, 1)), chunks, strict=True)
    )


def test_store_ends_with_an_empty_page_when_the_last_chunk_is_not_sent() -> None:
    """Only a page with count 0 ends the store, so when the last chunk is
    held an empty page at the end of the key closes it."""
    client = _Client(None)
    chunks = CHUNK[:3]
    assert _ctx(client).commit_store(
        cast(Any, None), 1, chunks, chunk_indices=[0, 1, 2], num_chunks=5
    )
    assert [(s, c) for s, c, _ in client.pages] == [(0, 2), (2, 1), (5, 0)]
    assert all(
        torch.equal(a, b) for a, b in zip(_sent(client, (2, 1, 0)), chunks, strict=True)
    )


def test_retrieve_reassembles_pages() -> None:
    client = _Client(None)
    chunks = _ctx(client).prepare_retrieve(cast(Any, None), 1, 5)
    assert [(s, c) for s, c, _ in client.pages] == [(0, 2), (2, 2), (4, 0)]
    assert chunks is not None
    assert all(torch.equal(a, b) for a, b in zip(chunks, CHUNK, strict=True))

    # Window starts size the pages: positions before group 1's window carry
    # one chunk, so they pack two per page; later ones carry two.
    groups = [([0, 1, 2, 3, 4], CHUNK), ([3, 4], CHUNK[3:])]
    client = _Client(groups)
    expected = [(torch.Size(SHAPE), torch.float32, 5)] * 2
    merged = _ctx(client).prepare_retrieve_multigroup(
        cast(Any, None), 1, expected, [0, 3]
    )
    assert [(s, c) for s, c, _ in client.pages] == [(0, 2), (2, 1), (3, 1), (4, 0)]
    assert merged is not None
    assert [ix for ix, _ in merged] == [[0, 1, 2, 3, 4], [3, 4]]
    assert all(
        torch.equal(a, b)
        for (_, got), (_, want) in zip(merged, groups, strict=True)
        for a, b in zip(got, want, strict=True)
    )
