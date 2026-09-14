# SPDX-License-Identifier: Apache-2.0
"""The server re-records and re-exports long-lived events per context.

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


def _exporter(conn: Connection) -> None:
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    event = torch.cuda.Event(interprocess=True)

    def record_after_sleep() -> None:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(_SLEEP_CYCLES)
            event.record(stream)

    record_after_sleep()
    conn.send(event.ipc_handle())
    conn.recv()  # importer has synchronized the first recording
    record_after_sleep()
    conn.send("re-recorded")
    conn.recv()  # importer is done


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_reexported_event_tracks_later_records_in_importer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CI sets CUDA_LAUNCH_BLOCKING=1, which would drain the exporter's sleep
    # inside the launch. The exporter is spawned, so clearing it here suffices.
    monkeypatch.delenv("CUDA_LAUNCH_BLOCKING", raising=False)
    device = torch.device("cuda", 0)
    ctx = mp.get_context("spawn")
    conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_exporter, args=(child_conn,))
    proc.start()
    try:
        handle = conn.recv()
        first = torch.cuda.Event.from_ipc_handle(device, handle)
        second = torch.cuda.Event.from_ipc_handle(device, handle)
        first.synchronize()
        assert (first.query(), second.query()) == (True, True)
        conn.send("synchronized")
        assert conn.recv() == "re-recorded"
        third = torch.cuda.Event.from_ipc_handle(device, handle)
        assert (first.query(), second.query(), third.query()) == (False, False, False)
        third.synchronize()
        assert (first.query(), third.query()) == (True, True)
        conn.send("done")
    finally:
        proc.join(timeout=60)
        if proc.is_alive():
            proc.kill()
    assert proc.exitcode == 0
