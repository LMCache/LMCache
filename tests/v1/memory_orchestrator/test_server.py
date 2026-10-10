# SPDX-License-Identifier: Apache-2.0
"""Tests for real memory orchestrator processes and their client."""

# Standard
from collections.abc import Callable, Iterator
from pathlib import Path

# Third Party
import grpc
import pytest

# First Party
from lmcache.v1.memory_orchestrator._proto_gen import memory_orchestrator_pb2 as pb2
from lmcache.v1.memory_orchestrator._proto_gen.memory_orchestrator_pb2_grpc import (
    MemoryOrchestratorStub,
)
from lmcache.v1.memory_orchestrator.api import (
    DEFAULT_LAYOUT_FINGERPRINT,
    CloseResult,
    OrchestratorUnavailableError,
    ReadRequest,
    ReadStatus,
    RegionContract,
    RegionFencedError,
    RegisterResult,
    RequestRejectedError,
    TokenStatus,
    WireLayout,
    WireObjectKey,
    WriteRequest,
    WriteStatus,
)
from lmcache.v1.memory_orchestrator.client import MemoryOrchestratorClient
from lmcache.v1.memory_orchestrator.server import OrchestratorConfig, parse_args
from tests.v1.memory_orchestrator.harness import (
    REGION_ID,
    STARTUP_TIMEOUT_S,
    Orchestrator,
    free_port,
    spawn_orchestrator,
)

ALIGN = 4096
CAPACITY = 1024 * ALIGN
MAX_BATCH = 4
MODE = "software_fenced"
LAYOUT = WireLayout(shapes=((2, 8),), dtypes=("bfloat16",))


def _spawn(state_dir: Path, log_path: Path) -> Orchestrator:
    """Start an orchestrator with this module's settings, without waiting."""
    return spawn_orchestrator(
        state_dir,
        capacity_bytes=CAPACITY,
        alignment_bytes=ALIGN,
        visibility_mode=MODE,
        max_batch_entries=MAX_BATCH,
        log_path=log_path,
    )


def register(client: MemoryOrchestratorClient) -> RegisterResult:
    """Register ``client`` with this module's layout, capacity and mode."""
    return client.register(
        layout_fingerprint=DEFAULT_LAYOUT_FINGERPRINT,
        mapped_bytes=CAPACITY,
        visibility_mode=MODE,
    )


def connect(orchestrator: Orchestrator, client_id: str) -> MemoryOrchestratorClient:
    """Return a client registered with ``orchestrator``."""
    client = MemoryOrchestratorClient(orchestrator.endpoint, REGION_ID, client_id)
    register(client)
    return client


def key(name: str) -> WireObjectKey:
    """Return the key of the object named ``name``."""
    return WireObjectKey(
        name.encode(), "model", kv_rank=0, object_group_id=0, cache_salt=""
    )


def write(name: str, payload_bytes: int = ALIGN) -> WriteRequest:
    """Return a write request for the object named ``name``."""
    return WriteRequest(key=key(name), payload_bytes=payload_bytes, layout=LAYOUT)


def raw_reserve_write(
    client: MemoryOrchestratorClient, names: list[str]
) -> pb2.ReserveWriteRequest:
    """Return a ReserveWrite request in ``client``'s name with a fixed request id."""
    return pb2.ReserveWriteRequest(
        env=pb2.Envelope(
            region_id=REGION_ID,
            expected_region_epoch=client.region_epoch,
            client_id=client.client_id,
            client_incarnation=client.client_incarnation,
            request_id=1 << 40,  # far above any id the client itself allocates
        ),
        entries=[
            pb2.WriteEntry(
                key=pb2.ObjectKey(chunk_hash=name.encode(), model_name="model"),
                payload_bytes=ALIGN,
                layout=pb2.ObjectLayout(
                    shapes=[pb2.TensorShape(dims=[2, 8])], dtypes=["bfloat16"]
                ),
            )
            for name in names
        ],
    )


@pytest.fixture(scope="module")
def shared(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Orchestrator]:
    """One orchestrator for the tests that only need a running region."""
    root = tmp_path_factory.mktemp("shared-orchestrator")
    orchestrator = _spawn(root / "state", root / "orchestrator.log")
    orchestrator.wait_for_marker()
    yield orchestrator
    orchestrator.kill()


@pytest.fixture
def clients(
    shared: Orchestrator,
) -> Iterator[Callable[[str], MemoryOrchestratorClient]]:
    """Registers clients with the shared orchestrator; closes them after."""
    opened: list[MemoryOrchestratorClient] = []

    def open_client(client_id: str) -> MemoryOrchestratorClient:
        client = connect(shared, client_id)
        opened.append(client)
        return client

    yield open_client
    for client in opened:
        client.close()


@pytest.fixture
def spawn(tmp_path: Path) -> Iterator[Callable[[Path], Orchestrator]]:
    """Starts orchestrators on a given state dir; kills them after the test."""
    spawned: list[Orchestrator] = []

    def start(state_dir: Path) -> Orchestrator:
        orchestrator = _spawn(state_dir, tmp_path / f"orchestrator-{len(spawned)}.log")
        spawned.append(orchestrator)
        return orchestrator

    yield start
    for orchestrator in spawned:
        orchestrator.kill()


def test_write_finish_read_flow_in_split_batches(clients):
    writer = clients("mp-writer")
    reader = clients("mp-reader")
    assert writer.describe_region() == RegionContract(
        REGION_ID,
        writer.region_epoch,
        CAPACITY,
        ALIGN,
        DEFAULT_LAYOUT_FINGERPRINT,
        MODE,
        MAX_BATCH,
        False,
    )
    # More entries than one RPC may carry, so the client splits every call below.
    names = [f"flow-{i}" for i in range(2 * MAX_BATCH + 2)]
    count = len(names)

    grants = writer.reserve_write([write(n, 100 + i) for i, n in enumerate(names)])
    assert [g.status for g in grants] == [WriteStatus.WRITE_GRANTED] * count
    offsets = [g.handle.offset for g in grants]
    assert offsets == sorted(set(offsets))
    assert all(offset % ALIGN == 0 for offset in offsets)
    assert {g.handle.length for g in grants} == {ALIGN}
    busy = reader.reserve_read([ReadRequest(key(name), 1) for name in names])
    assert [r.status for r in busy] == [ReadStatus.BUSY_WRITING] * count

    assert writer.finish_write([g.token for g in grants]) == [TokenStatus.OK] * count
    reads = reader.reserve_read([ReadRequest(key(n), 2) for n in [*names, "absent"]])
    assert [r.status for r in reads[:count]] == [ReadStatus.READ_GRANTED] * count
    assert reads[count].status is ReadStatus.MISS
    for i, (grant, granted) in enumerate(zip(grants, reads[:count], strict=True)):
        assert granted.handle == grant.handle
        assert granted.layout == LAYOUT
        assert granted.payload_bytes == 100 + i
        assert len(granted.leases) == 2

    leases = [lease for granted in reads for lease in granted.leases]
    leased = reader.usage().read_leases
    assert reader.finish_read(leases) == [TokenStatus.OK] * 2 * count
    assert reader.finish_read(leases[:1]) == [TokenStatus.OK]
    assert reader.usage().read_leases == leased - 2 * count
    again = reader.reserve_write([write(name) for name in names])
    assert [g.status for g in again] == [WriteStatus.EXISTS_VALID] * count


def test_repeated_request_id_replays_or_rejects(shared, clients):
    client = clients("mp-replay")
    request = raw_reserve_write(client, ["replay-0", "replay-1"])
    with grpc.insecure_channel(shared.endpoint) as channel:
        stub = MemoryOrchestratorStub(channel)
        first = stub.ReserveWrite(request, timeout=5)
        allocated = client.usage().allocated_bytes
        # The same request again gets the recorded reply and allocates nothing.
        second = stub.ReserveWrite(request, timeout=5)
        assert second == first
        assert [g.status for g in first.grants] == [pb2.WriteGrant.WRITE_GRANTED] * 2
        assert client.usage().allocated_bytes == allocated
        # The same request id with another payload is refused.
        with pytest.raises(grpc.RpcError) as excinfo:
            stub.ReserveWrite(raw_reserve_write(client, ["replay-2"]), timeout=5)
        assert excinfo.value.code() is grpc.StatusCode.ALREADY_EXISTS
    assert client.usage().allocated_bytes == allocated
    # The replayed tokens are the live ones.
    assert client.finish_write([g.token for g in second.grants]) == [TokenStatus.OK] * 2


def test_failed_split_batch_rolls_back_the_earlier_rpcs(clients):
    client = clients("mp-partial")
    names = [f"partial-{i}" for i in range(MAX_BATCH)]
    before = client.usage()
    with pytest.raises(RequestRejectedError) as excinfo:
        client.reserve_write([write(name) for name in names] + [write("partial-x", 0)])
    assert excinfo.value.code == "INVALID_ARGUMENT"
    assert client.usage().consumed == before.consumed + MAX_BATCH
    grants = client.reserve_write([write(name) for name in names])
    assert [g.status for g in grants] == [WriteStatus.WRITE_GRANTED] * MAX_BATCH

    client.finish_write([g.token for g in grants])
    with pytest.raises(RequestRejectedError) as excinfo:
        client.reserve_read(
            [ReadRequest(key(name), 1) for name in names] + [ReadRequest(key("x"), 0)]
        )
    assert excinfo.value.code == "INVALID_ARGUMENT"
    assert client.usage().read_leases == before.read_leases


def test_close_client_retires_writes(clients):
    closer = clients("mp-closer")
    observer = clients("mp-observer")
    committed = closer.reserve_write([write("close-valid")])[0]
    closer.finish_write([committed.token])
    closer.reserve_read([ReadRequest(key("close-valid"), 1)])
    busy = closer.reserve_write([write("close-busy")])[0]
    blocked = observer.reserve_write([write("close-busy")])[0]
    assert blocked.status is WriteStatus.BUSY_WRITING
    before = observer.usage()

    assert closer.close() == CloseResult(aborted_writes=1, released_leases=1)
    assert closer.close() is None
    after = observer.usage()
    assert after.consumed == before.consumed + 1
    assert after.read_leases == before.read_leases - 1
    assert after.clients == before.clients - 1
    regrant = observer.reserve_write([write("close-busy")])[0]
    assert regrant.status is WriteStatus.WRITE_GRANTED
    assert regrant.handle.offset > busy.handle.offset


def test_new_incarnation_fences_the_previous_client(shared):
    old = connect(shared, "mp-restarted")
    grant = old.reserve_write([write("restart-busy")])[0]

    new = MemoryOrchestratorClient(shared.endpoint, REGION_ID, "mp-restarted")
    assert register(new) == RegisterResult(
        region_epoch=old.region_epoch, retired_writes=1, released_leases=0
    )
    with pytest.raises(RegionFencedError) as excinfo:
        old.finish_write([grant.token])
    assert excinfo.value.reset_required is False
    assert "client not registered" in str(excinfo.value)
    assert new.finish_write([grant.token]) == [TokenStatus.STALE_TOKEN]
    assert new.close() == CloseResult(aborted_writes=0, released_leases=0)
    assert old.close() is None  # the retired client's refusal is logged, not raised


def test_wrong_region_is_fenced(shared):
    client = MemoryOrchestratorClient(shared.endpoint, "pool-other", "mp-lost")
    with pytest.raises(RegionFencedError) as excinfo:
        client.describe_region()
    assert excinfo.value.reset_required is False
    assert client.close() is None


def test_unreachable_orchestrator_raises_unavailable():
    endpoint = f"127.0.0.1:{free_port()}"
    client = MemoryOrchestratorClient(
        endpoint, REGION_ID, "mp-alone", rpc_timeout_s=1.0, max_attempts=2
    )
    # Empty batches send nothing, so they succeed even without a server.
    assert client.reserve_write([]) == []
    assert client.finish_read([]) == []
    with pytest.raises(OrchestratorUnavailableError):
        client.describe_region()
    assert client.close() is None


def test_second_orchestrator_on_the_same_state_dir_exits_2(shared, spawn):
    second = spawn(shared.state_dir)
    assert second.process.wait(timeout=STARTUP_TIMEOUT_S) == 2
    assert shared.marker()["pid"] == shared.process.pid
    assert str(shared.process.pid) in second.log()


@pytest.mark.parametrize(
    "marker_text",
    [
        '{"region_id": "pool-test", "hostname": "elsewhere", "pid": 1, '
        '"region_epoch": 1, "endpoint": "10.0.0.9:7700", "started_at": "", '
        '"reset_required": false}',
        '{"pid": 1, "hostname": "elsewhere", "endpoint": "10.0.0.9:7700"}',
        "not json",
    ],
    ids=["another-host", "incomplete", "unreadable"],
)
def test_marker_of_another_host_or_unreadable_marker_is_refused(
    spawn, tmp_path, marker_text
):
    (tmp_path / f"{REGION_ID}.marker").write_text(marker_text)
    refused = spawn(tmp_path)
    assert refused.process.wait(timeout=STARTUP_TIMEOUT_S) == 2
    assert refused.marker_path.read_text() == marker_text


def test_restart_after_kill_requires_reset(spawn, tmp_path):
    first = spawn(tmp_path / "state")
    first.wait_for_marker()
    grant = connect(first, "mp-crash").reserve_write([write("crash-busy")])[0]
    assert grant.status is WriteStatus.WRITE_GRANTED
    # Reap it too: a zombie still counts as a live pid.
    first.kill()

    second = spawn(first.state_dir)
    second.wait_for_marker()
    assert second.marker()["reset_required"] is True
    fresh = MemoryOrchestratorClient(second.endpoint, REGION_ID, "mp-crash")
    assert fresh.describe_region().reset_required is True
    with pytest.raises(RegionFencedError) as excinfo:
        register(fresh)
    assert excinfo.value.reset_required is True
    with pytest.raises(RegionFencedError) as excinfo:
        fresh.reserve_write([write("crash-busy")])
    assert excinfo.value.reset_required is True
    assert fresh.close() is None

    # A clean stop keeps the marker: the region still needs an offline reset.
    assert second.stop() == 0
    assert second.marker_path.exists()


def test_graceful_stop_removes_the_marker_unless_clients_remain(spawn, tmp_path):
    idle = spawn(tmp_path / "idle")
    lingering = spawn(tmp_path / "lingering")
    idle.wait_for_marker()
    lingering.wait_for_marker()
    client = connect(idle, "mp-graceful")
    assert client.close() == CloseResult(aborted_writes=0, released_leases=0)
    connect(lingering, "mp-lingering")

    assert idle.stop() == 0
    assert not idle.marker_path.exists()
    assert lingering.stop() == 0
    assert lingering.marker_path.exists()
    assert "clients are still registered" in lingering.log()


def test_parse_args_defaults_and_capacity_rounding(tmp_path):
    required = ["--region-id", "r", "--state-dir", str(tmp_path)]
    assert parse_args([*required, "--capacity-gb", "1.001"]) == OrchestratorConfig(
        region_id="r",
        capacity_bytes=1 << 30,
        alignment_bytes=2 << 20,
        layout_fingerprint=DEFAULT_LAYOUT_FINGERPRINT,
        visibility_mode="software_fenced",
        max_batch_entries=4096,
        listen="0.0.0.0:7700",
        state_dir=tmp_path,
        max_workers=16,
    )
    config = parse_args(
        [*required, "--capacity-bytes", str(10 * ALIGN + 5), "--alignment", "4k"]
        + ["--layout-fingerprint", "custom", "--visibility-mode", "coherent"]
        + ["--listen", "[::]:0"]
    )
    assert config.capacity_bytes == 10 * ALIGN
    assert config.alignment_bytes == ALIGN
    assert config.layout_fingerprint == b"custom"
    assert config.visibility_mode == "coherent"
    assert config.listen == "[::]:0"


@pytest.mark.parametrize(
    "extra",
    [
        ["--capacity-gb", "1"],  # both capacity flags
        ["--alignment", "3000"],  # not a power of two
        ["--alignment", "2X"],  # bad size suffix
        ["--capacity-bytes", "4095", "--alignment", "4K"],  # below the alignment
        ["--listen", "7700"],  # no host
        ["--region-id", "../escape"],  # would leave the state dir
        ["--max-batch-entries", "0"],  # not positive
    ],
)
def test_parse_args_rejects_invalid_flags(tmp_path, extra):
    argv = ["--region-id", "r", "--capacity-bytes", str(CAPACITY)]
    with pytest.raises(SystemExit) as excinfo:
        parse_args([*argv, "--state-dir", str(tmp_path), *extra])
    assert excinfo.value.code == 2
