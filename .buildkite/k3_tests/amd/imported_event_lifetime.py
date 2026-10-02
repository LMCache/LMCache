# SPDX-License-Identifier: Apache-2.0
"""Two-process reproducer for the imported-event lifetime hazard.

Exporter: records an interprocess event behind a producer kernel that runs
for several seconds, ships the handle. Importer: opens the handle, makes its
own stream wait on it, queues work behind the wait, then either drops the
import while the wait is still pending (--drop, what the MP server did at
handler return) or holds it until the stream has drained (--hold, what the
fix does).

The run is only meaningful if the wait really was pending at the drop point,
so the importer initializes its device context first and signals the exporter
to start the producer; the producer length is calibrated at runtime instead
of assuming a clock rate. The process exits non-zero if the wait was not
pending.

Expected: --hold completes everywhere. --drop is where runtimes differ; a
runtime that frees the event's signal under the queued wait faults or hangs
here, which shows up as a crash before the PROBE_RESULT line is printed.

Usage: python imported_event_lifetime.py --drop|--hold [--seconds N]
"""

# Standard
import argparse
import gc
import sys
import time

# Third Party
import torch
import torch.multiprocessing as mp


def _cycles_for(seconds: float) -> int:
    """Calibrate torch.cuda._sleep so the producer runs for about `seconds`."""
    probe_cycles = 100_000_000
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.cuda._sleep(probe_cycles)
    torch.cuda.synchronize()
    per_cycle = max(time.perf_counter() - t0, 1e-6) / probe_cycles
    return int(seconds / per_cycle)


def exporter(conn, seconds):
    torch.cuda.set_device(0)
    cycles = _cycles_for(seconds)
    conn.recv()  # importer is initialized and ready to wait
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
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()
    assert args.drop != args.hold, "pick one of --drop / --hold"
    mode = "drop" if args.drop else "hold"

    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=exporter, args=(child, args.seconds))
    proc.start()

    # Bring up this process's device context and load the kernels used below
    # before the producer starts. A runtime that loads kernel modules lazily
    # may synchronize the device on a kernel's first launch, which would
    # silently wait out the producer before the drop point.
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    x = torch.ones(1 << 20, device="cuda")
    with torch.cuda.stream(stream):
        warm = (x * 2) + 1
    torch.cuda.synchronize()
    assert bool((warm == 3).all())
    parent.send("ready")
    handle = parent.recv()

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
    t_drop = time.time()
    stream.synchronize()
    waited = time.time() - t0
    blocked = time.time() - t_drop
    if args.hold:
        del imported
    # The hazard needs the wait still queued when the import is dropped. The
    # producer runs for `seconds`, so a synchronize that returns quickly means
    # the wait had already been satisfied and the run proves nothing.
    pending = blocked >= min(1.0, args.seconds / 2)
    ok = bool((z == 3).all())
    parent.send("done")
    proc.join(timeout=60)
    print(
        f"PROBE_RESULT mode={mode} pending={pending} blocked={blocked:.2f}s "
        f"waited={waited:.2f}s result_ok={ok} exporter_exit={proc.exitcode}",
        flush=True,
    )
    sys.exit(0 if ok and pending and proc.exitcode == 0 else 1)


if __name__ == "__main__":
    main()
