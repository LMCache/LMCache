# SPDX-License-Identifier: Apache-2.0
"""Two-process reproducer for the imported-event lifetime hazard.

Exporter: records an interprocess event behind a long sleep kernel, ships
the handle. Importer: opens the handle, makes its own stream wait on it,
queues work behind the wait, then either drops the import right away
(--drop, what the MP server did at handler return) or holds it until the
stream has drained (--hold, what the fix does).

Expected: --hold completes everywhere. --drop is where runtimes differ; a
runtime that frees the event's signal under the queued wait faults or
hangs here.

Usage: python imported_event_lifetime.py --drop|--hold [--cycles N]
"""

# Standard
import argparse
import gc
import sys
import time

# Third Party
import torch
import torch.multiprocessing as mp


def exporter(conn, cycles):
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    event = torch.cuda.Event(interprocess=True)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(cycles)  # long-running producer work
        event.record(stream)
    conn.send(event.ipc_handle())
    conn.recv()  # importer finished
    stream.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", action="store_true")
    ap.add_argument("--hold", action="store_true")
    ap.add_argument("--cycles", type=int, default=2_000_000_000)
    args = ap.parse_args()
    assert args.drop != args.hold, "pick one of --drop / --hold"

    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=exporter, args=(child, args.cycles))
    proc.start()
    handle = parent.recv()

    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    x = torch.ones(1 << 20, device="cuda")
    t0 = time.time()
    imported = torch.cuda.Event.from_ipc_handle(torch.device("cuda", 0), handle)
    with torch.cuda.stream(stream):
        stream.wait_event(imported)  # queued wait referencing the import
        y = x * 2  # consumer work behind the wait
    if args.drop:
        del imported  # what the server did at handler return
        gc.collect()
    with torch.cuda.stream(stream):
        z = y + 1  # more work behind the pending wait
    stream.synchronize()
    if args.hold:
        del imported
    waited = time.time() - t0
    ok = bool((z == 3).all())
    parent.send("done")
    proc.join(timeout=60)
    mode = "drop" if args.drop else "hold"
    print(
        f"mode={mode} waited={waited:.2f}s result_ok={ok} exporter_exit={proc.exitcode}"
    )
    sys.exit(0 if ok and proc.exitcode == 0 and waited > 0.5 else 1)


if __name__ == "__main__":
    main()
