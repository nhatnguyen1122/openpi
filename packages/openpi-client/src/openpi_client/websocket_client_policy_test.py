import asyncio
import queue
import threading
import time

import pytest
import websockets.asyncio.server

from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy


@pytest.fixture
def slow_policy_server():
    """Model cold starts block the server's event loop while compiling."""
    ready = queue.Queue()

    async def handler(connection):
        await connection.send(msgpack_numpy.packb({}))
        await connection.recv()
        time.sleep(0.3)
        await connection.send(msgpack_numpy.packb({"actions": [1, 2, 3]}))

    async def run():
        stop = asyncio.Event()
        async with websockets.asyncio.server.serve(handler, "127.0.0.1", 0) as server:
            ready.put((server.sockets[0].getsockname()[1], asyncio.get_running_loop(), stop))
            await stop.wait()

    thread = threading.Thread(target=lambda: asyncio.run(run()), daemon=True)
    thread.start()
    port, loop, stop = ready.get(timeout=5)
    try:
        yield port
    finally:
        loop.call_soon_threadsafe(stop.set)
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_inference_survives_server_cold_start(monkeypatch, slow_policy_server):
    # Accelerate the library's default heartbeat so the real cold-start failure
    # can be reproduced in milliseconds instead of waiting forty seconds.
    original_connect = websocket_client_policy.websockets.sync.client.connect

    def connect(*args, **kwargs):
        kwargs.setdefault("ping_interval", 0.01)
        kwargs["ping_timeout"] = 0.02
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", connect)
    client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", slow_policy_server)
    assert client.infer({}) == {"actions": [1, 2, 3]}
