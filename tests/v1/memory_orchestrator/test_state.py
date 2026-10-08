# SPDX-License-Identifier: Apache-2.0
"""Tests for the memory orchestrator's region state machine (no gRPC)."""

# Standard
from collections.abc import Callable
from dataclasses import replace
from functools import partial

# Third Party
import pytest

# First Party
from lmcache.v1.memory_orchestrator.api import (
    CloseResult,
    Envelope,
    ReadGrantResult,
    ReadRequest,
    ReadStatus,
    RegisterResult,
    TokenStatus,
    WireLayout,
    WireObjectKey,
    WriteGrantResult,
    WriteRequest,
    WriteStatus,
)
from lmcache.v1.memory_orchestrator.state import RegionState, StateError

REGION = "pool-a"
EPOCH = 7
ALIGN = 4096
CAPACITY = 16 * ALIGN
FINGERPRINT = b"fp/v1"
MODE = "software_fenced"
MAX_BATCH = 8
LAYOUT = WireLayout(shapes=((2, 4, 16),), dtypes=("bfloat16",))

make_state = partial(
    RegionState,
    region_id=REGION,
    capacity_bytes=CAPACITY,
    alignment_bytes=ALIGN,
    layout_fingerprint=FINGERPRINT,
    visibility_mode=MODE,
    max_batch_entries=MAX_BATCH,
    region_epoch=EPOCH,
    reset_required=False,
)


def envelope(client_id: str = "a", incarnation: int = 1) -> Envelope:
    return Envelope(REGION, EPOCH, client_id, incarnation, request_id=1)


def register(state: RegionState, client_id: str = "a") -> Envelope:
    env = envelope(client_id)
    state.register_client(env, FINGERPRINT, CAPACITY, MODE)
    return env


def key(name: str) -> WireObjectKey:
    return WireObjectKey(
        chunk_hash=name.encode(),
        model_name="model",
        kv_rank=0,
        object_group_id=0,
        cache_salt="",
    )


def write(name: str, payload_bytes: int = ALIGN) -> WriteRequest:
    return WriteRequest(key=key(name), payload_bytes=payload_bytes, layout=LAYOUT)


def read(name: str, reader_count: int = 1) -> ReadRequest:
    return ReadRequest(key=key(name), reader_count=reader_count)


def commit(state: RegionState, env: Envelope, *names: str) -> None:
    grants = state.reserve_write(env, [write(name) for name in names])
    tokens = [grant.token for grant in grants if grant.token is not None]
    assert len(tokens) == len(names)
    assert state.finish_write(env, tokens) == [TokenStatus.OK] * len(names)


def client_calls(state: RegionState, env: Envelope) -> list[Callable[[], object]]:
    """Return every client call to ``state`` with ``env``, register_client first."""
    return [
        partial(state.register_client, env, FINGERPRINT, CAPACITY, MODE),
        partial(state.check_client, env),
        partial(state.close_client, env),
        partial(state.reserve_write, env, [write("x")]),
        partial(state.finish_write, env, [b"t"]),
        partial(state.abort_write, env, [b"t"]),
        partial(state.reserve_read, env, [read("x")]),
        partial(state.finish_read, env, [b"t"]),
    ]


def assert_refused(code: str, call: Callable[..., object], *args: object) -> StateError:
    with pytest.raises(StateError) as excinfo:
        call(*args)
    assert excinfo.value.code == code
    return excinfo.value


@pytest.fixture
def state() -> RegionState:
    return make_state()


@pytest.fixture
def env(state: RegionState) -> Envelope:
    return register(state)


class TestConstruction:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"alignment_bytes": 0},
            {"alignment_bytes": 3000},  # not a power of two
            {"capacity_bytes": 0},
            {"capacity_bytes": ALIGN + 1},  # not a multiple of the alignment
            {"visibility_mode": "eventual"},
            {"max_batch_entries": 0},
            {"region_epoch": 0},
        ],
    )
    def test_invalid_arguments_raise(self, overrides):
        with pytest.raises(ValueError):
            make_state(**overrides)


class TestWrites:
    def test_one_writer_per_key(self, state, env):
        first = state.reserve_write(env, [write("k")])[0]
        assert first.status is WriteStatus.WRITE_GRANTED
        assert first.handle is not None and first.token is not None
        second = state.reserve_write(register(state, "b"), [write("k")])
        assert second == [WriteGrantResult(WriteStatus.BUSY_WRITING, None, None)]
        assert state.usage().writing == 1

    def test_out_of_space_is_all_or_nothing_for_absent_keys(self, state, env):
        commit(state, env, "valid")
        busy = state.reserve_write(env, [write("busy")])[0]
        other = register(state, "b")
        before = state.usage()
        free = CAPACITY - before.allocated_bytes
        half = free // 2
        grants = state.reserve_write(
            other,
            [write("valid"), write("x", half), write("busy"), write("y", half + 1)],
        )
        assert grants == [
            WriteGrantResult(WriteStatus.EXISTS_VALID, None, None),
            WriteGrantResult(WriteStatus.OUT_OF_SPACE, None, None),
            WriteGrantResult(WriteStatus.BUSY_WRITING, None, None),
            WriteGrantResult(WriteStatus.OUT_OF_SPACE, None, None),
        ]
        assert state.usage() == before
        # The writer that holds "busy" is unaffected, and what fits is granted.
        assert state.finish_write(env, [busy.token]) == [TokenStatus.OK]
        fits = state.reserve_write(other, [write("x", free)])[0]
        assert fits.status is WriteStatus.WRITE_GRANTED
        assert state.usage().allocated_bytes == CAPACITY

    def test_allocation_is_monotonic_and_never_reuses_an_extent(self, state, env):
        grants = state.reserve_write(
            env, [write("a", 1), write("b", ALIGN), write("c", ALIGN + 1)]
        )
        handles = [grant.handle for grant in grants]
        assert [(h.offset, h.length, h.generation) for h in handles] == [
            (0, ALIGN, 1),
            (ALIGN, ALIGN, 1),
            (2 * ALIGN, 2 * ALIGN, 1),
        ]
        # An aborted extent stays consumed; its key gets a new one.
        assert state.abort_write(env, [grants[0].token]) == [TokenStatus.OK]
        assert state.reserve_read(env, [read("a")])[0].status is ReadStatus.MISS
        usage = state.usage()
        assert (usage.consumed, usage.writing, usage.valid) == (1, 2, 0)
        again = state.reserve_write(env, [write("a")])[0]
        assert again.status is WriteStatus.WRITE_GRANTED
        assert again.handle.offset == 4 * ALIGN
        assert again.token != grants[0].token
        assert state.usage().allocated_bytes == 5 * ALIGN

    @pytest.mark.parametrize(
        "entry",
        [
            write("k", 0),  # empty payload
            write("ok"),  # duplicate key in the batch
            WriteRequest(key("k"), ALIGN, WireLayout(shapes=(), dtypes=())),
            # one dtype per shape
            WriteRequest(
                key("k"), ALIGN, WireLayout(shapes=((1,), (2,)), dtypes=("uint8",))
            ),
        ],
    )
    def test_invalid_entries_reject_the_whole_batch(self, state, env, entry):
        batch = [write("ok"), entry]
        assert_refused("INVALID_ARGUMENT", state.reserve_write, env, batch)
        usage = state.usage()
        assert (usage.allocated_bytes, usage.writing) == (0, 0)

    def test_batch_above_max_batch_entries_is_rejected(self, state, env):
        entries = [write(f"k{i}") for i in range(MAX_BATCH + 1)]
        tokens = [b"t"] * (MAX_BATCH + 1)
        assert_refused("RESOURCE_EXHAUSTED", state.reserve_write, env, entries)
        assert_refused("RESOURCE_EXHAUSTED", state.finish_write, env, tokens)
        assert state.usage().allocated_bytes == 0


class TestReads:
    def test_reads_only_see_valid_objects(self, state, env):
        state.reserve_write(env, [write("writing")])
        grant = state.reserve_write(env, [write("valid", 100)])[0]
        state.finish_write(env, [grant.token])
        busy, miss, hit = state.reserve_read(
            env, [read("writing"), read("absent"), read("valid")]
        )
        assert busy == ReadGrantResult(ReadStatus.BUSY_WRITING, None, ())
        assert miss == ReadGrantResult(ReadStatus.MISS, None, ())
        assert (hit.status, hit.handle) == (ReadStatus.READ_GRANTED, grant.handle)
        assert (hit.layout, hit.payload_bytes, len(hit.leases)) == (LAYOUT, 100, 1)

    def test_lease_counting_with_reader_count(self, state, env):
        commit(state, env, "k")
        grant = state.reserve_read(env, [read("k", 3)])[0]
        assert len(set(grant.leases)) == 3
        assert state.usage().read_leases == 3
        assert state.finish_read(env, list(grant.leases[:2])) == [TokenStatus.OK] * 2
        assert state.usage().read_leases == 1
        assert state.finish_read(env, [grant.leases[0]]) == [TokenStatus.OK]
        assert state.usage().read_leases == 1
        assert state.finish_read(env, [grant.leases[2]]) == [TokenStatus.OK]
        assert state.usage().read_leases == 0

    @pytest.mark.parametrize("reader_count", [0, 1025])
    def test_reader_count_out_of_range_is_rejected(self, state, env, reader_count):
        commit(state, env, "k")
        batch = [read("k"), read("k", reader_count)]
        assert_refused("INVALID_ARGUMENT", state.reserve_read, env, batch)
        assert state.usage().read_leases == 0


class TestTokens:
    def test_a_token_completes_once_and_stays_ok_for_that_call(self, state, env):
        grants = state.reserve_write(env, [write("k"), write("x")])
        finished, aborted = (grant.token for grant in grants)
        # A write token is not a lease.
        assert state.finish_read(env, [finished]) == [TokenStatus.STALE_TOKEN]
        assert state.finish_write(env, [finished, finished]) == [TokenStatus.OK] * 2
        assert state.finish_write(env, [finished]) == [TokenStatus.OK]
        assert state.abort_write(env, [aborted]) == [TokenStatus.OK]
        assert state.abort_write(env, [aborted]) == [TokenStatus.OK]
        usage = state.usage()
        assert (usage.valid, usage.writing, usage.consumed) == (1, 0, 1)
        assert usage.valid_bytes == ALIGN
        # Completed under one call, a token is stale under the others.
        assert state.abort_write(env, [finished]) == [TokenStatus.STALE_TOKEN]
        assert state.finish_read(env, [finished]) == [TokenStatus.STALE_TOKEN]
        assert state.finish_write(env, [aborted]) == [TokenStatus.STALE_TOKEN]
        assert state.reserve_read(env, [read("k")])[0].status is ReadStatus.READ_GRANTED

    def test_other_clients_and_unknown_tokens_are_stale(self, state, env):
        commit(state, env, "valid")
        lease = state.reserve_read(env, [read("valid")])[0].leases[0]
        token = state.reserve_write(env, [write("k")])[0].token
        other = register(state, "b")
        stale = [TokenStatus.STALE_TOKEN] * 2
        assert state.finish_write(other, [token, b"unknown"]) == stale
        assert state.abort_write(other, [token, b"unknown"]) == stale
        assert state.finish_read(other, [lease, b"unknown"]) == stale
        assert state.reserve_read(env, [read("k")])[0].status is ReadStatus.BUSY_WRITING
        assert state.finish_write(env, [token]) == [TokenStatus.OK]
        assert state.finish_read(env, [lease]) == [TokenStatus.OK]


class TestEnvelopes:
    def test_refusals_change_no_state(self, state, env):
        commit(state, env, "valid")
        state.reserve_write(env, [write("busy")])
        before, clients = state.usage(), state.registered_clients()
        wrong_region = replace(env, region_id="pool-b")
        stale_epoch = replace(env, expected_region_epoch=EPOCH + 1)
        for bad in (wrong_region, stale_epoch):
            for call in client_calls(state, bad):
                assert_refused("FAILED_PRECONDITION", call)
        # A new client or incarnation may register, but nothing else.
        for bad in (envelope("stranger"), replace(env, client_incarnation=2)):
            for call in client_calls(state, bad)[1:]:
                assert_refused("FAILED_PRECONDITION", call)
        assert state.usage() == before
        assert state.registered_clients() == clients

    @pytest.mark.parametrize(
        "args",
        [
            (b"other", CAPACITY, MODE),
            (FINGERPRINT, CAPACITY, "coherent"),
            (FINGERPRINT, CAPACITY - 1, MODE),
        ],
    )
    def test_register_mismatch_is_refused(self, state, env, args):
        token = state.reserve_write(env, [write("k")])[0].token
        # Neither a new client nor a new incarnation of a registered one.
        for new in (envelope("b"), envelope(incarnation=2)):
            assert_refused("FAILED_PRECONDITION", state.register_client, new, *args)
        assert state.registered_clients() == 1
        assert state.finish_write(env, [token]) == [TokenStatus.OK]


class TestClients:
    def test_new_incarnation_retires_the_old_one(self, state, env):
        commit(state, env, "valid")
        lease = state.reserve_read(env, [read("valid", 2)])[0].leases[0]
        old = state.reserve_write(env, [write("busy")])[0]

        new_env = envelope(incarnation=2)
        result = state.register_client(new_env, FINGERPRINT, CAPACITY, MODE)
        assert result == RegisterResult(
            region_epoch=EPOCH, retired_writes=1, released_leases=2
        )
        usage = state.usage()
        assert (usage.writing, usage.consumed) == (0, 1)
        assert (usage.read_leases, usage.clients) == (0, 1)
        assert_refused("FAILED_PRECONDITION", state.finish_write, env, [old.token])
        assert state.finish_write(new_env, [old.token]) == [TokenStatus.STALE_TOKEN]
        assert state.finish_read(new_env, [lease]) == [TokenStatus.STALE_TOKEN]
        again = state.reserve_write(new_env, [write("busy")])[0]
        assert again.status is WriteStatus.WRITE_GRANTED
        assert again.handle.offset > old.handle.offset
        # Registering it again returns the first result and retires nothing.
        retry = state.register_client(new_env, FINGERPRINT, CAPACITY + ALIGN, MODE)
        assert retry == result
        assert state.finish_write(new_env, [again.token]) == [TokenStatus.OK]

    def test_close_client_retires_and_unregisters(self, state, env):
        commit(state, env, "valid")
        state.reserve_read(env, [read("valid")])
        state.reserve_write(env, [write("w1"), write("w2")])
        other = register(state, "b")

        assert state.close_client(env) == CloseResult(
            aborted_writes=2, released_leases=1
        )
        assert state.registered_clients() == 1
        usage = state.usage()
        assert (usage.writing, usage.valid) == (0, 1)
        assert (usage.consumed, usage.read_leases) == (2, 0)
        error = assert_refused("FAILED_PRECONDITION", state.close_client, env)
        assert error.message.startswith("client not registered")
        w1, valid = state.reserve_write(other, [write("w1"), write("valid")])
        assert w1.status is WriteStatus.WRITE_GRANTED
        assert valid.status is WriteStatus.EXISTS_VALID
        # The closed client may register again with a new incarnation.
        assert state.register_client(
            envelope(incarnation=3), FINGERPRINT, CAPACITY, MODE
        ) == RegisterResult(region_epoch=EPOCH, retired_writes=0, released_leases=0)


class TestResetRequired:
    def test_reset_required_refuses_every_client_call(self):
        state = make_state(reset_required=True)
        for call in client_calls(state, envelope()):
            error = assert_refused("FAILED_PRECONDITION", call)
            assert error.message.startswith("RESET_REQUIRED")
        assert state.contract().reset_required is True
        assert state.registered_clients() == 0
        assert state.usage().allocated_bytes == 0


class TestUsage:
    def test_usage_numbers(self, state, env):
        other = register(state, "b")
        commit(state, env, "v1", "v2")
        state.reserve_write(env, [write("w", 2 * ALIGN)])
        aborted = state.reserve_write(env, [write("x", 3 * ALIGN)])[0]
        state.abort_write(env, [aborted.token])
        state.reserve_read(other, [read("v1", 2), read("v2")])

        usage = state.usage()
        assert (usage.capacity_bytes, usage.allocated_bytes) == (CAPACITY, 7 * ALIGN)
        assert (usage.valid_bytes, usage.writing, usage.valid) == (2 * ALIGN, 1, 2)
        assert (usage.consumed, usage.read_leases, usage.clients) == (1, 3, 2)
        state.check_client(env)
