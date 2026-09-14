# SPDX-License-Identifier: Apache-2.0
"""The server re-records and re-exports one long-lived event per context.

This relies on two device-runtime properties, checked here on real hardware:
a peer may open the same handle repeatedly, and every import observes the
exporter's later recordings.
"""

# Standard
from multiprocessing.connection import Connection

# Third Party
import pytest
import torch
import torch.multiprocessing as mp

_SLEEP_CYCLES = 2_000_000_000  # ~1 s at 2 GHz


def _importer(handle: bytes, conn: Connection) -> None:
    device = torch.device("cuda", 0)
    first = torch.cuda.Event.from_ipc_handle(device, handle)
    second = torch.cuda.Event.from_ipc_handle(device, handle)
    first.synchronize()
    conn.send((first.query(), second.query()))
    conn.recv()  # exporter re-recorded behind a device sleep
    third = torch.cuda.Event.from_ipc_handle(device, handle)
    conn.send((first.query(), second.query(), third.query()))
    third.synchronize()
    conn.send((first.query(), third.query()))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_reexported_event_tracks_later_records_in_importer() -> None:
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    event = torch.cuda.Event(interprocess=True)

    def record_after_sleep() -> None:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(_SLEEP_CYCLES)
            event.record(stream)

    record_after_sleep()
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_importer, args=(event.ipc_handle(), child_conn))
    proc.start()
    try:
        assert parent_conn.recv() == (True, True)
        record_after_sleep()
        parent_conn.send("re-recorded")
        assert parent_conn.recv() == (False, False, False)
        assert parent_conn.recv() == (True, True)
    finally:
        proc.join(timeout=60)
        if proc.is_alive():
            proc.kill()
    assert proc.exitcode == 0
