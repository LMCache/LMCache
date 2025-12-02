# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import json
import threading
import time

# Third Party
from fastapi.testclient import TestClient
import msgspec
import pytest
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.controller_manager import LMCacheControllerManager
from lmcache.v1.cache_controller.message import HeartbeatMsg, RegisterMsg
from lmcache.v1.internal_api_server.api_server import app
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket

# Test utilities
from tests.v1.utils import get_available_port

logger = init_logger(__name__)


class TestWorkerInfoAPI:
    """Test suite for the /controller/workers API endpoint."""

    @pytest.fixture
    def zmq_context(self):
        return get_zmq_context()

    @pytest.fixture
    def controller_urls(self):
        pull_port = get_available_port()
        reply_port = get_available_port()
        return {
            "pull": f"127.0.0.1:{pull_port}",
            "reply": f"127.0.0.1:{reply_port}",
        }

    @pytest.fixture
    def real_controller_manager(self, zmq_context, controller_urls):
        controller_manager = LMCacheControllerManager(
            controller_urls=controller_urls,
            health_check_interval=10,
            lmcache_worker_timeout=30,
        )

        self.controller_thread = threading.Thread(
            target=lambda: asyncio.run(controller_manager.start_all()), daemon=True
        )
        self.controller_thread.start()
        time.sleep(0.5)

        yield controller_manager

        if hasattr(controller_manager, "controller_pull_socket"):
            controller_manager.controller_pull_socket.close()
        if hasattr(controller_manager, "controller_reply_socket"):
            controller_manager.controller_reply_socket.close()
        if hasattr(controller_manager, "zmq_context"):
            controller_manager.zmq_context.destroy()

    @pytest.fixture
    def worker_socket(self, zmq_context, controller_urls):
        socket = get_zmq_socket(
            zmq_context,
            controller_urls["pull"],
            protocol="tcp",
            role=zmq.PUSH,
            bind_or_connect="connect",
        )
        yield socket
        socket.close()

    @pytest.fixture
    def client_with_real_controller(self, real_controller_manager):
        app.state.lmcache_controller_manager = real_controller_manager
        return TestClient(app)

    @pytest.fixture
    def client_without_controller(self):
        app.state.lmcache_controller_manager = None
        return TestClient(app)

    def _send_message(self, worker_socket, msg_type, instance_id, worker_id, ip, port):
        """Send message to controller via ZMQ"""
        if msg_type == "register":
            msg = RegisterMsg(
                instance_id=instance_id,
                worker_id=worker_id,
                ip=ip,
                port=port,
                peer_init_url=None,
            )
        else:
            msg = HeartbeatMsg(
                instance_id=instance_id,
                worker_id=worker_id,
                ip=ip,
                port=port,
                peer_init_url=None,
            )

        message_data = msgspec.msgpack.encode(msg)
        worker_socket.send(message_data)
        time.sleep(0.1)

    def _assert_worker_fields(self, worker_data):
        required_fields = {
            "instance_id",
            "worker_id",
            "ip",
            "port",
            "peer_init_url",
            "registration_time",
            "last_heartbeat_time",
        }
        for field in required_fields:
            assert field in worker_data, f"Missing field: {field}"

    def _assert_worker_list_response(self, response, expected_count):
        assert response.status_code == 200
        data = json.loads(response.text)
        assert "workers" in data
        assert "total_count" in data
        assert data["total_count"] == expected_count
        return data

    def _assert_single_worker_response(self, response, instance_id, worker_id):
        assert response.status_code == 200
        data = json.loads(response.text)
        self._assert_worker_fields(data)
        assert data["instance_id"] == instance_id
        assert data["worker_id"] == worker_id
        return data

    def test_real_worker_registration_and_query(
        self, client_with_real_controller, worker_socket
    ):
        # Register workers via real ZMQ communication
        self._send_message(worker_socket, "register", "instance1", 0, "127.0.0.1", 8000)
        self._send_message(worker_socket, "register", "instance1", 1, "127.0.0.1", 8001)
        self._send_message(worker_socket, "register", "instance2", 0, "127.0.0.2", 8002)

        # Send heartbeats to update last_heartbeat_time
        self._send_message(
            worker_socket, "heartbeat", "instance1", 0, "127.0.0.1", 8000
        )
        self._send_message(
            worker_socket, "heartbeat", "instance1", 1, "127.0.0.1", 8001
        )
        self._send_message(
            worker_socket, "heartbeat", "instance2", 0, "127.0.0.2", 8002
        )

        # Test getting all workers
        response = client_with_real_controller.get("/controller/workers")
        data = self._assert_worker_list_response(response, 3)

        workers = data["workers"]
        assert len(workers) == 3

        worker_keys = {(w["instance_id"], w["worker_id"]) for w in workers}
        assert worker_keys == {("instance1", 0), ("instance1", 1), ("instance2", 0)}

        for worker in workers:
            self._assert_worker_fields(worker)
            assert time.time() - worker["registration_time"] < 10
            assert time.time() - worker["last_heartbeat_time"] < 10

    def test_real_workers_by_instance(self, client_with_real_controller, worker_socket):
        self._send_message(worker_socket, "register", "instance1", 0, "127.0.0.1", 8000)
        self._send_message(worker_socket, "register", "instance2", 0, "127.0.0.2", 8002)

        response = client_with_real_controller.get(
            "/controller/workers?instance_id=instance1"
        )
        data = self._assert_worker_list_response(response, 1)

        workers = data["workers"]
        assert len(workers) == 1
        assert workers[0]["instance_id"] == "instance1"
        assert workers[0]["worker_id"] == 0

    def test_real_specific_worker_query(
        self, client_with_real_controller, worker_socket
    ):
        self._send_message(worker_socket, "register", "instance1", 0, "127.0.0.1", 8000)
        self._send_message(
            worker_socket, "heartbeat", "instance1", 0, "127.0.0.1", 8000
        )

        response = client_with_real_controller.get(
            "/controller/workers?instance_id=instance1&worker_id=0"
        )
        data = self._assert_single_worker_response(response, "instance1", 0)

        assert data["ip"] == "127.0.0.1"
        assert data["port"] == 8000
        assert data["peer_init_url"] is None
        assert time.time() - data["last_heartbeat_time"] < 10

    def test_real_nonexistent_worker(self, client_with_real_controller):
        response = client_with_real_controller.get(
            "/controller/workers?instance_id=nonexistent&worker_id=999"
        )
        assert response.status_code == 404
        data = json.loads(response.text)
        assert "detail" in data
        assert "Worker ('nonexistent', 999) not found" in data["detail"]

    def test_real_nonexistent_instance(self, client_with_real_controller):
        response = client_with_real_controller.get(
            "/controller/workers?instance_id=nonexistent"
        )
        assert response.status_code == 404
        data = json.loads(response.text)
        assert "detail" in data
        assert "No workers found for instance nonexistent" in data["detail"]

    def test_real_controller_manager_not_available(self, client_without_controller):
        response = client_without_controller.get("/controller/workers")
        assert response.status_code == 503
        data = json.loads(response.text)
        assert "detail" in data
        assert "Controller manager not available" in data["detail"]
