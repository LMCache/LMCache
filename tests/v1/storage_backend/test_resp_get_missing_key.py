# SPDX-License-Identifier: Apache-2.0
"""A GET of a missing key must fail fast, not hang (native RESP client).

A minimal in-process RESP server answers SET, EXISTS and GET; a GET of an unknown
key gets the nil reply "$-1\\r\\n". Before the fix the client waited for the
fixed-size bulk header forever and the connection was wedged. Now the per-key
results are [True, False, True], the found key's bytes are correct and the same
connection keeps working. No Redis/Valkey needed.
"""

# Standard
import asyncio
import concurrent.futures
import socketserver
import threading
import time

# Third Party
import pytest

pytest.importorskip("lmcache.lmcache_redis")

# First Party
from lmcache.v1.storage_backend.native_clients.resp_client import (  # noqa: E402
    RESPClient,
)

CHUNK = 1 << 20


class _Handler(socketserver.BaseRequestHandler):
    store: dict = {}

    def _line(self, f):
        line = f.readline()
        if not line:
            raise EOFError
        return line

    def handle(self):
        f = self.request.makefile("rb")
        try:
            while True:
                n = int(self._line(f)[1:].strip())  # "*N\r\n"
                args = []
                for _ in range(n):
                    ln = int(self._line(f)[1:].strip())  # "$len\r\n"
                    args.append(f.read(ln))
                    f.read(2)
                cmd = args[0].upper()
                if cmd == b"SET":
                    self.store[args[1]] = args[2]
                    self.request.sendall(b"+OK\r\n")
                elif cmd == b"EXISTS":
                    self.request.sendall(b":%d\r\n" % int(args[1] in self.store))
                elif cmd == b"GET":
                    v = self.store.get(args[1])
                    if v is None:
                        self.request.sendall(b"$-1\r\n")
                    else:
                        self.request.sendall(b"$%d\r\n" % len(v) + v + b"\r\n")
                else:
                    self.request.sendall(b"-ERR unknown\r\n")
        except (EOFError, ConnectionError, ValueError):
            pass


def _server():
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _client(port, workers=1):
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    box = []
    loop.call_soon_threadsafe(
        lambda: box.append(RESPClient("127.0.0.1", port, workers, loop))
    )
    while not box:
        time.sleep(0.01)
    return box[0]


def _in_thread(fn, *a, timeout=5):
    """Run fn on a daemon thread, so a call that hangs fails the test."""
    fut: concurrent.futures.Future = concurrent.futures.Future()

    def run():
        try:
            fut.set_result(fn(*a))
        except BaseException as e:  # noqa: BLE001
            fut.set_exception(e)

    threading.Thread(target=run, daemon=True).start()
    return fut.result(timeout=timeout)  # TimeoutError = the GET hung


def test_get_missing_key_fails_fast_and_connection_survives():
    srv = _server()
    c = _client(
        srv.server_address[1], workers=1
    )  # one connection: a wedge would block the rest
    payload = bytes(range(256)) * (CHUNK // 256)
    _in_thread(c.batch_set_sync, ["k1"], [memoryview(bytearray(payload))])
    bufs = [memoryview(bytearray(CHUNK)) for _ in range(3)]
    res = _in_thread(c.batch_get_sync, ["k1", "missing", "k1"], bufs)
    assert list(res) == [True, False, True]
    assert bytes(bufs[0]) == payload and bytes(bufs[2]) == payload
    buf = memoryview(bytearray(CHUNK))
    assert list(_in_thread(c.batch_get_sync, ["k1"], [buf])) == [True]
    assert bytes(buf) == payload
    srv.shutdown()
