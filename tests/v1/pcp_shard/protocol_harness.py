# SPDX-License-Identifier: Apache-2.0
"""Multi-process (gloo) check of the shard exchange/broadcast protocol in
lmcache.v1.pcp_shard, with torch only (no engine).

Run: python protocol_harness.py <world_size> <rounds>
Every round: the same random chunk list on all ranks (sometimes a different
one on one rank), random per-rank fetch failures. Checks on every rank: the
agreed prefix equals the reference (first chunk whose owner failed), every
chunk below it arrives with its owner's bytes, and the next round still works
(the collective sequence never got out of step).
"""

# Standard
from datetime import timedelta
import os
import random
import sys
import traceback

# Third Party
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# First Party
from lmcache.v1 import pcp_shard

CHUNK = 8


def payload(round_id: int, j: int, n_bytes: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(round_id * 100003 + j)
    return torch.randint(0, 256, (n_bytes,), dtype=torch.uint8, generator=g)


def bcast(t, src):
    dist.broadcast(t, src)


def bcast_obj(obj, src):
    box = [obj]
    dist.broadcast_object_list(box, src)
    return box[0]


def run(rank: int, world: int, rounds: int) -> None:
    shared = random.Random(42)  # same stream on all ranks
    for rnd in range(rounds):
        n_tokens = shared.randint(0, 30 * CHUNK)
        n_chunks = (n_tokens + CHUNK - 1) // CHUNK
        bounds = [(s, min(s + CHUNK, n_tokens)) for s in range(0, n_tokens, CHUNK)]
        hashes = [shared.getrandbits(63) for _ in bounds]
        # per-rank fetch failure: rank r fails at one of its owned chunks
        fail_at = {}
        for r in range(world):
            owned = [j for j in range(n_chunks) if j % world == r]
            if owned and shared.random() < 0.4:
                fail_at[r] = shared.choice(owned)
        raise_rank = shared.randrange(world) if shared.random() < 0.1 else None
        mismatch_rank = shared.randrange(world) if shared.random() < 0.1 else None

        my_bounds, my_hashes = bounds, hashes
        if rank == mismatch_rank and bounds:
            my_hashes = hashes[:-1] + [hashes[-1] ^ 1]

        owners = [pcp_shard.chunk_owner(s, CHUNK, world) for s, _ in my_bounds]
        owned = pcp_shard.owned_positions([s for s, _ in my_bounds], CHUNK, world, rank)
        local, metas = {}, {}
        first_fail = n_chunks
        if rank == raise_rank:
            first_fail = owned[0] if owned else n_chunks
        else:
            for j in owned:
                if fail_at.get(rank) == j:
                    first_fail = j
                    break
                s, e = my_bounds[j]
                nb = (e - s) * 4
                local[j] = payload(rnd, j, nb)
                metas[j] = {"nbytes": nb}
        fp = pcp_shard.fingerprint(my_bounds, my_hashes)
        msgs = pcp_shard.exchange(rank, world, (fp, first_fail, metas), bcast_obj)
        prefix = pcp_shard.agree_prefix(msgs, n_chunks, owners)

        # reference
        if mismatch_rank is not None and bounds:
            expect = 0
        else:
            fails = list(fail_at.values())
            if raise_rank is not None:
                ro = [j for j in range(n_chunks) if j % world == raise_rank]
                if ro:
                    fails.append(ro[0])
            expect = min([n_chunks] + fails)
        assert prefix == expect, (rnd, rank, prefix, expect)

        got = pcp_shard.broadcast_chunks(
            rank,
            prefix,
            owners,
            msgs,
            local,
            lambda meta: torch.empty(meta["nbytes"], dtype=torch.uint8),
            bcast,
        )
        assert sorted(got) == list(range(prefix))
        for j, t in got.items():
            s, e = bounds[j]
            assert torch.equal(t, payload(rnd, j, (e - s) * 4)), (rnd, j)


def _worker(rank, world, rounds, port, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    try:
        dist.init_process_group(
            "gloo", rank=rank, world_size=world, timeout=timedelta(seconds=60)
        )
        run(rank, world, rounds)
        q.put((rank, "ok", ""))
    except BaseException:
        q.put((rank, "fail", traceback.format_exc()))
    finally:
        sys.stdout.flush()
        q.close()
        q.join_thread()
        os._exit(0)


def main():
    world, rounds = int(sys.argv[1]), int(sys.argv[2])
    port = 31000 + random.Random(os.getpid()).randint(0, 2000)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(r, world, rounds, port, q))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    results = [q.get(timeout=240) for _ in range(world)]
    for p in procs:
        p.join(timeout=30)
    failed = [r for r in results if r[1] != "ok"]
    for rank, _, tb in sorted(failed):
        print(f"rank {rank} FAILED:\n{tb}", file=sys.stderr)
    print(f"world={world} rounds={rounds} ok={len(results) - len(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
