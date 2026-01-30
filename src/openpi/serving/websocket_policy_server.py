import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()

                obs = msgpack_numpy.unpackb(await websocket.recv())

                async for message in websocket:
                    # 第二步：反序列化（msgpack_numpy 会自动还原 numpy 数组，不拆字典）
                    # object_hook=m.decode 强制还原 numpy 数组，而非拆成 bytes 字典
                    obs = msgpack_numpy.unpackb(
                        message,
                        object_hook=msgpack_numpy.decode,  # 核心：还原 numpy 数组
                        raw=True  # 不自动解码 bytes，避免碰二进制数据
                    )

                    # 第三步：仅清理「文本类 bytes 键」（如 b'observation.state'），不碰数组
                def clean_text_bytes(obj):
                    if isinstance(obj, dict):
                        cleaned = {}
                        for k, v in obj.items():
                            # 只解码「可打印的文本 bytes 键」（数组内部不会有这类键）
                            if isinstance(k, bytes):
                                try:
                                    # 尝试解码为 utf-8 文本
                                    decoded_k = k.decode('utf-8')
                                    # 仅保留可打印的文本键（排除二进制数据）
                                    if decoded_k.isprintable():
                                        k = decoded_k
                                except UnicodeDecodeError:
                                    # 二进制键（如数组内部）不处理
                                    pass
                            # 递归清理嵌套字典
                            cleaned[k] = clean_text_bytes(v)
                        return cleaned
                    return obj
                    
                obs = clean_text_bytes(obs)

                # for k, v in obs.items():
                #     print(f"key: {k!r} (type={type(k).__name__}), value type: {type(v).__name__}, shape: {getattr(v, 'shape', 'N/A')}")
                
                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
