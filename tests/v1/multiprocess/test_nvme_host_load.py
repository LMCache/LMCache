# SPDX-License-Identifier: Apache-2.0
"""The NVMe engine's host-thread load: when it applies, and that it returns
or raises only once every read of the batch has ended. The slab read is
faked, so these run without GDS or CUDA."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.object_group_transfer import nvme_dma
from lmcache.v1.multiprocess.object_group_transfer.nvme_dma import NvmeDmaEngine


@pytest.fixture
def log(monkeypatch) -> list:
    """Fake the slab read on a slab with two reader threads; reads are
    logged as ``("read", object, buffer)``."""
    entries: list = []
    monkeypatch.setattr(nvme_dma, "read_threads", lambda memory_obj: 2)
    monkeypatch.setattr(
        nvme_dma,
        "read_slab_object",
        lambda memory_obj, buffer, device: entries.append(("read", memory_obj, buffer)),
    )
    return entries


def test_load_returns_after_every_object_of_the_batch_was_read(log):
    NvmeDmaEngine().load_from_host(["o0", "o1", "o2"], ["b0", "b1", "b2"], "gpu")
    assert sorted(log) == [
        ("read", "o0", "b0"),
        ("read", "o1", "b1"),
        ("read", "o2", "b2"),
    ]


def test_failed_read_is_raised_after_the_other_reads_have_ended(log, monkeypatch):
    """The caller may reuse the regions once the call has raised, so no
    reader may still be writing them."""
    slow_read_started = threading.Event()
    release_slow_read = threading.Event()

    def read(memory_obj, buffer, device):
        if memory_obj == "bad":
            slow_read_started.wait(10)
            raise OSError("short slab read")
        slow_read_started.set()
        release_slow_read.wait(10)
        log.append(("read", memory_obj, buffer))

    monkeypatch.setattr(nvme_dma, "read_slab_object", read)
    threading.Timer(0.2, release_slow_read.set).start()
    with pytest.raises(OSError, match="short slab read"):
        NvmeDmaEngine().load_from_host(["slow", "bad"], ["b0", "b1"], "gpu")
    assert log == [("read", "slow", "b0")]


def _slab_object(read_threads: int, monkeypatch) -> MagicMock:
    """An object whose slab is configured with ``read_threads`` threads."""
    monkeypatch.setattr(
        nvme_dma,
        "get_l1_gds_context",
        lambda owner: SimpleNamespace(read_threads=owner),
    )
    obj = MagicMock()
    obj.get_l1_manager.return_value = read_threads
    return obj


def test_host_load_needs_reader_threads_on_every_objects_slab(monkeypatch):
    engine = NvmeDmaEngine()
    threaded = _slab_object(2, monkeypatch)
    stream_only = _slab_object(0, monkeypatch)
    assert engine.loads_from_host([threaded, threaded]) is True
    assert engine.loads_from_host([threaded, stream_only]) is False
