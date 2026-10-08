# SPDX-License-Identifier: Apache-2.0
"""Tests for the memory orchestrator's region state machine (no gRPC)."""

# Standard
from collections.abc import Callable
from dataclasses import replace

# Third Party
import pytest

# First Party
from lmcache.v1.memory_orchestrator.api import (
    CloseResult,
    Envelope,
    ReadRequest,
    ReadStatus,
    RegisterResult,
    TokenStatus,
    WireLayout,
    WireObjectKey,
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


def make_state(*, reset_required: bool = False) -> RegionState:
    return RegionState(
        REGION,
        CAPACITY,
        ALIGN,
        FINGERPRINT,
        MODE,
        MAX_BATCH,
        region_epoch=EPOCH,
        reset_required=reset_required,
    )


def envelope(client_id: str = "a", incarnation: int = 1) -> Envelope:
    return Envelope(
        region_id=REGION,
        expected_region_epoch=EPOCH,
        client_id=client_id,
        client_incarnation=incarnation,
        request_id=1,
    )


def register(
    state: RegionState, client_id: str = "a", incarnation: int = 1
) -> Envelope:
    env = envelope(client_id, incarnation)
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
            {"alignment_bytes": 3000},
            {"alignment_bytes": 0},
            {"capacity_bytes": 0},
            {"capacity_bytes": ALIGN + 1},
            {"visibility_mode": "eventual"},
            {"max_batch_entries": 0},
            {"region_epoch": 0},
        ],
    )
    def test_invalid_arguments_raise(self, overrides):
        kwargs = {
            "region_id": REGION,
            "capacity_bytes": CAPACITY,
            "alignment_bytes": ALIGN,
            "layout_fingerprint": FINGERPRINT,
            "visibility_mode": MODE,
            "max_batch_entries": MAX_BATCH,
            "region_epoch": EPOCH,
            "reset_required": False,
        }
        kwargs.update(overrides)
        with pytest.raises(ValueError):
            RegionState(**kwargs)


class TestWrites:
    def test_duplicate_writers_get_one_grant(self, state, env):
        other = register(state, "b")
        first = state.reserve_write(env, [write("k")])
        second = state.reserve_write(other, [write("k")])
        assert first[0].status is WriteStatus.WRITE_GRANTED
        assert first[0].handle is not None and first[0].token is not None
        assert second[0].status is WriteStatus.BUSY_WRITING
        assert second[0].handle is None and second[0].token is None
        assert state.usage().writing == 1

    def test_abort_consumes_extent_and_key_can_be_written_again(self, state, env):
        first = state.reserve_write(env, [write("k")])[0]
        assert state.abort_write(env, [first.token]) == [TokenStatus.OK]
        assert state.reserve_read(env, [read("k")])[0].status is ReadStatus.MISS
        usage = state.usage()
        assert (usage.consumed, usage.writing, usage.valid) == (1, 0, 0)

        second = state.reserve_write(env, [write("k")])[0]
        assert second.status is WriteStatus.WRITE_GRANTED
        assert second.handle.offset > first.handle.offset
        assert second.token != first.token
        assert state.usage().allocated_bytes == 2 * ALIGN

    def test_out_of_space_is_all_or_nothing_for_absent_keys(self, state, env):
        commit(state, env, "valid")
        busy = state.reserve_write(env, [write("busy")])[0]
        before = state.usage()
        free = CAPACITY - before.allocated_bytes
        other = register(state, "b")
        grants = state.reserve_write(
            other,
            [
                write("valid"),
                write("new-1", free // 2),
                write("busy"),
                write("new-2", free // 2 + 1),
            ],
        )
        assert [grant.status for grant in grants] == [
            WriteStatus.EXISTS_VALID,
            WriteStatus.OUT_OF_SPACE,
            WriteStatus.BUSY_WRITING,
            WriteStatus.OUT_OF_SPACE,
        ]
        assert all(grant.handle is None and grant.token is None for grant in grants)
        assert state.usage() == replace(before, clients=before.clients + 1)
        # The writer that holds "busy" is unaffected, and what fits is granted.
        assert state.finish_write(env, [busy.token]) == [TokenStatus.OK]
        fits = state.reserve_write(other, [write("new-1", free)])[0]
        assert fits.status is WriteStatus.WRITE_GRANTED
        assert state.usage().allocated_bytes == CAPACITY

    def test_extents_are_aligned_and_allocated_in_request_order(self, state, env):
        grants = state.reserve_write(
            env, [write("a", 1), write("b", ALIGN), write("c", ALIGN + 1)]
        )
        handles = [grant.handle for grant in grants]
        assert [(h.offset, h.length, h.generation) for h in handles] == [
            (0, ALIGN, 1),
            (ALIGN, ALIGN, 1),
            (2 * ALIGN, 2 * ALIGN, 1),
        ]
        assert state.usage().allocated_bytes == 4 * ALIGN

    @pytest.mark.parametrize(
        "entry",
        [
            write("k", 0),
            write("ok"),  # duplicate key in the batch
            WriteRequest(key("k"), ALIGN, WireLayout(shapes=(), dtypes=())),
            WriteRequest(
                key("k"), ALIGN, WireLayout(shapes=((1,), (2,)), dtypes=("uint8",))
            ),
        ],
    )
    def test_invalid_entries_reject_the_whole_batch(self, state, env, entry):
        assert_refused(
            "INVALID_ARGUMENT", state.reserve_write, env, [write("ok"), entry]
        )
        usage = state.usage()
        assert (usage.allocated_bytes, usage.writing) == (0, 0)

    def test_batch_above_max_batch_entries_is_rejected(self, state, env):
        entries = [write(f"k{i}") for i in range(MAX_BATCH + 1)]
        assert_refused("RESOURCE_EXHAUSTED", state.reserve_write, env, entries)
        assert_refused(
            "RESOURCE_EXHAUSTED", state.finish_write, env, [b"t"] * (MAX_BATCH + 1)
        )
        assert state.usage().allocated_bytes == 0


class TestReads:
    def test_reads_only_see_valid_objects(self, state, env):
        state.reserve_write(env, [write("writing")])
        grant = state.reserve_write(env, [write("valid", 100)])[0]
        state.finish_write(env, [grant.token])

        grants = state.reserve_read(
            env, [read("writing"), read("absent"), read("valid")]
        )
        assert [g.status for g in grants] == [
            ReadStatus.BUSY_WRITING,
            ReadStatus.MISS,
            ReadStatus.READ_GRANTED,
        ]
        assert grants[0].handle is None and grants[0].leases == ()
        assert grants[1].handle is None and grants[1].leases == ()
        assert grants[2].handle == grant.handle
        assert grants[2].layout == LAYOUT
        assert grants[2].payload_bytes == 100
        assert len(grants[2].leases) == 1

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
        assert_refused(
            "INVALID_ARGUMENT",
            state.reserve_read,
            env,
            [read("k"), read("k", reader_count)],
        )
        assert state.usage().read_leases == 0

    def test_lease_of_another_client_is_stale(self, state, env):
        commit(state, env, "k")
        lease = state.reserve_read(env, [read("k")])[0].leases[0]
        other = register(state, "b")
        assert state.finish_read(other, [lease]) == [TokenStatus.STALE_TOKEN]
        assert state.finish_read(env, [lease]) == [TokenStatus.OK]


class TestTokens:
    def test_a_write_token_completes_once_and_stays_ok_for_that_call(self, state, env):
        token = state.reserve_write(env, [write("k")])[0].token
        # A write token is not a lease.
        assert state.finish_read(env, [token]) == [TokenStatus.STALE_TOKEN]
        assert state.finish_write(env, [token, token]) == [TokenStatus.OK] * 2
        assert state.finish_write(env, [token]) == [TokenStatus.OK]
        usage = state.usage()
        assert (usage.valid, usage.writing, usage.valid_bytes) == (1, 0, ALIGN)
        # Finished: aborting it is stale, releasing it as a lease still is.
        assert state.abort_write(env, [token]) == [TokenStatus.STALE_TOKEN]
        assert state.finish_read(env, [token]) == [TokenStatus.STALE_TOKEN]
        assert state.reserve_read(env, [read("k")])[0].status is ReadStatus.READ_GRANTED

    def test_abort_twice_is_ok_and_finish_after_abort_is_stale(self, state, env):
        token = state.reserve_write(env, [write("k")])[0].token
        assert state.abort_write(env, [token]) == [TokenStatus.OK]
        assert state.abort_write(env, [token]) == [TokenStatus.OK]
        assert state.finish_write(env, [token]) == [TokenStatus.STALE_TOKEN]
        assert state.usage().consumed == 1

    def test_other_clients_and_unknown_tokens_are_stale(self, state, env):
        token = state.reserve_write(env, [write("k")])[0].token
        other = register(state, "b")
        assert (
            state.finish_write(other, [token, b"unknown"])
            == [TokenStatus.STALE_TOKEN] * 2
        )
        assert state.abort_write(other, [token]) == [TokenStatus.STALE_TOKEN]
        assert state.reserve_read(env, [read("k")])[0].status is ReadStatus.BUSY_WRITING
        assert state.finish_write(env, [token]) == [TokenStatus.OK]


class TestEnvelopes:
    def test_refusals_change_no_state(self, state, env):
        commit(state, env, "valid")
        state.reserve_write(env, [write("busy")])
        before = state.usage()
        clients = state.registered_clients()

        wrong_region = replace(env, region_id="pool-b")
        stale_epoch = replace(env, expected_region_epoch=EPOCH + 1)
        unregistered = envelope("stranger")
        old_incarnation = replace(env, client_incarnation=2)
        for bad in (wrong_region, stale_epoch, unregistered, old_incarnation):
            assert_refused(
                "FAILED_PRECONDITION", state.reserve_write, bad, [write("x")]
            )
            assert_refused("FAILED_PRECONDITION", state.finish_write, bad, [b"t"])
            assert_refused("FAILED_PRECONDITION", state.abort_write, bad, [b"t"])
            assert_refused("FAILED_PRECONDITION", state.reserve_read, bad, [read("x")])
            assert_refused("FAILED_PRECONDITION", state.finish_read, bad, [b"t"])
            assert_refused("FAILED_PRECONDITION", state.check_client, bad)
            assert_refused("FAILED_PRECONDITION", state.close_client, bad)
        for bad in (wrong_region, stale_epoch):
            assert_refused(
                "FAILED_PRECONDITION",
                state.register_client,
                bad,
                FINGERPRINT,
                CAPACITY,
                MODE,
            )

        assert state.usage() == before
        assert state.registered_clients() == clients

    @pytest.mark.parametrize(
        "fingerprint, mapped_bytes, mode",
        [
            (b"other", CAPACITY, MODE),
            (FINGERPRINT, CAPACITY, "coherent"),
            (FINGERPRINT, CAPACITY - 1, MODE),
        ],
    )
    def test_register_mismatch_is_refused(self, state, fingerprint, mapped_bytes, mode):
        assert_refused(
            "FAILED_PRECONDITION",
            state.register_client,
            envelope(),
            fingerprint,
            mapped_bytes,
            mode,
        )
        assert state.registered_clients() == 0

    def test_mismatched_reregister_keeps_the_old_incarnation(self, state, env):
        token = state.reserve_write(env, [write("k")])[0].token
        assert_refused(
            "FAILED_PRECONDITION",
            state.register_client,
            envelope(incarnation=2),
            b"other",
            CAPACITY,
            MODE,
        )
        assert state.finish_write(env, [token]) == [TokenStatus.OK]


class TestClients:
    def test_register_same_incarnation_is_idempotent(self, state, env):
        token = state.reserve_write(env, [write("k")])[0].token
        result = state.register_client(env, FINGERPRINT, CAPACITY + ALIGN, MODE)
        assert result == RegisterResult(
            region_epoch=EPOCH, retired_writes=0, released_leases=0
        )
        assert state.registered_clients() == 1
        assert state.finish_write(env, [token]) == [TokenStatus.OK]

    def test_new_incarnation_retires_the_old_one(self, state, env):
        commit(state, env, "valid")
        lease = state.reserve_read(env, [read("valid", 2)])[0].leases[0]
        old = state.reserve_write(env, [write("busy")])[0]

        new_env = envelope(incarnation=2)
        result = state.register_client(new_env, FINGERPRINT, CAPACITY, MODE)
        assert result == RegisterResult(
            region_epoch=EPOCH, retired_writes=1, released_leases=2
        )
        # A retry of that registration returns the same result.
        assert state.register_client(new_env, FINGERPRINT, CAPACITY, MODE) == result

        usage = state.usage()
        assert (usage.writing, usage.consumed, usage.read_leases, usage.clients) == (
            0,
            1,
            0,
            1,
        )
        assert_refused("FAILED_PRECONDITION", state.finish_write, env, [old.token])
        assert state.finish_write(new_env, [old.token]) == [TokenStatus.STALE_TOKEN]
        assert state.finish_read(new_env, [lease]) == [TokenStatus.STALE_TOKEN]
        again = state.reserve_write(new_env, [write("busy")])[0]
        assert again.status is WriteStatus.WRITE_GRANTED
        assert again.handle.offset > old.handle.offset

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
        assert (usage.writing, usage.valid, usage.consumed, usage.read_leases) == (
            0,
            1,
            2,
            0,
        )
        error = assert_refused("FAILED_PRECONDITION", state.close_client, env)
        assert error.message.startswith("client not registered")
        grants = state.reserve_write(other, [write("w1"), write("valid")])
        assert [g.status for g in grants] == [
            WriteStatus.WRITE_GRANTED,
            WriteStatus.EXISTS_VALID,
        ]
        # The closed client may register again with a new incarnation.
        assert state.register_client(
            envelope(incarnation=3), FINGERPRINT, CAPACITY, MODE
        ) == RegisterResult(region_epoch=EPOCH, retired_writes=0, released_leases=0)


class TestResetRequired:
    def test_reset_required_refuses_every_client_call(self):
        state = make_state(reset_required=True)
        env = envelope()
        error = assert_refused(
            "FAILED_PRECONDITION",
            state.register_client,
            env,
            FINGERPRINT,
            CAPACITY,
            MODE,
        )
        assert error.message.startswith("RESET_REQUIRED")
        for call, arg in (
            (state.reserve_write, [write("k")]),
            (state.finish_write, [b"t"]),
            (state.abort_write, [b"t"]),
            (state.reserve_read, [read("k")]),
            (state.finish_read, [b"t"]),
        ):
            error = assert_refused("FAILED_PRECONDITION", call, env, arg)
            assert error.message.startswith("RESET_REQUIRED")
        for call in (state.close_client, state.check_client):
            error = assert_refused("FAILED_PRECONDITION", call, env)
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
        assert usage.capacity_bytes == CAPACITY
        assert usage.allocated_bytes == 7 * ALIGN
        assert usage.valid_bytes == 2 * ALIGN
        assert usage.writing == 1
        assert usage.valid == 2
        assert usage.consumed == 1
        assert usage.read_leases == 3
        assert usage.clients == 2
        state.check_client(env)
