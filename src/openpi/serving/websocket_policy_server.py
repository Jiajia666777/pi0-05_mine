import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

import pathlib
import numpy as np
from typing import Dict, Any
# 导入开源库中的归一化工具
import openpi.shared.normalize as normalize

# 图像归一化默认配置（可通过 JSON 覆盖）
DEFAULT_IMAGE_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGE_STD = [0.229, 0.224, 0.225]

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
        norm_stats_path: str | pathlib.Path | None = None,  # 本地归一化统计文件路径
        use_quantile_norm: bool = False,  # 是否使用分位数归一化
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

        # 加载本地归一化统计数据
        self.norm_stats = None
        self.image_norm_mean = DEFAULT_IMAGE_MEAN
        self.image_norm_std = DEFAULT_IMAGE_STD
        self._use_quantile_norm = use_quantile_norm
        if norm_stats_path:
            self._load_norm_stats(norm_stats_path)

    def _load_norm_stats(self, norm_stats_path: str | pathlib.Path) -> None:
        """从本地 JSON 文件加载归一化统计数据"""
        norm_stats_path = pathlib.Path(norm_stats_path)
        if not norm_stats_path.exists():
            raise FileNotFoundError(f"归一化统计文件不存在: {norm_stats_path}")

        try:
            # 使用开源库的加载函数读取 JSON
            self.norm_stats = normalize.load(norm_stats_path.parent)
            logger.info(f"成功加载归一化统计数据: {norm_stats_path}")
            
            # 如果 JSON 中包含图像归一化参数，覆盖默认值
            if "image" in self.norm_stats:
                self.image_norm_mean = self.norm_stats["image"].mean.tolist()
                self.image_norm_std = self.norm_stats["image"].std.tolist()
                logger.info(f"加载图像归一化参数 - mean: {self.image_norm_mean}, std: {self.image_norm_std}")
                
        except Exception as e:
            raise RuntimeError(f"加载归一化统计数据失败: {e}")
        
    def _normalize_state(self, state: np.ndarray, stats: normalize.NormStats) -> np.ndarray:
        """对 proprioceptive state 进行归一化（支持 z-score / 分位数）"""
        if self._use_quantile_norm:
            # 分位数归一化: (x - q01) / (q99 - q01) * 2 - 1
            assert stats.q01 is not None and stats.q99 is not None, "分位数统计数据缺失"
            q01 = stats.q01[..., :state.shape[-1]]
            q99 = stats.q99[..., :state.shape[-1]]
            return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        else:
            # 标准 z-score 归一化: (x - mean) / std
            mean = stats.mean[..., :state.shape[-1]]
            std = stats.std[..., :state.shape[-1]]
            return (state - mean) / (std + 1e-6)
        
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """对单张图像进行归一化"""
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0  # [0,255] -> [0,1]
        
        # 调整维度为 (C, H, W)（适配模型输入）
        if img.ndim == 3 and img.shape[-1] == 3:  # (H, W, C) -> (C, H, W)
            img = np.transpose(img, (2, 0, 1))
        
        # 应用均值/标准差归一化
        mean = np.array(self.image_norm_mean, dtype=np.float32)[:, None, None]
        std = np.array(self.image_norm_std, dtype=np.float32)[:, None, None]
        img = (img - mean) / (std + 1e-6)
        
        return img
    
    def normalize_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        对观测数据进行全量归一化：
        1. state: 使用本地 JSON 中的统计数据
        2. 图像: 使用默认/JSON 中的图像归一化参数
        3. 递归处理嵌套字典
        """
        obs_processed = obs.copy()
        
        def _recursive_normalize(data: Any) -> Any:
            if isinstance(data, dict):
                return {k: _recursive_normalize(v) for k, v in data.items()}
            
            # 处理 State 归一化
            elif isinstance(data, np.ndarray) and "state" in obs_processed and self.norm_stats:
                if "state" in self.norm_stats and data.shape[-1] == self.norm_stats["state"].mean.shape[-1]:
                    return self._normalize_state(data, self.norm_stats["state"])
            
            # 处理图像归一化（匹配包含图像关键词的字段）
            elif isinstance(data, np.ndarray) and data.ndim == 3:
                if any(key in str(obs_processed.keys()).lower() for key in ["image", "cam", "rgb", "wrist"]):
                    return self._normalize_image(data)
            
            return data

        # 遍历所有字段进行归一化
        for key, value in obs_processed.items():
            obs_processed[key] = _recursive_normalize(value)
        
        return obs_processed

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """对模型输出的actions做反归一化，还原为原始业务尺度（默认分位数，兼容Z-score）"""
        # 无归一化统计数据/无actions统计项，直接返回原动作
        if self.norm_stats is None or "actions" not in self.norm_stats:
            return actions
        stats = self.norm_stats["actions"]
        action_dim = actions.shape[-1]

        if self._use_quantile_norm:
            # 分位数反归一化（与归一化严格互逆）: x = (x_norm +1)/2 * (q99-q01) + q01
            if stats.q01 is None or stats.q99 is None:
                logger.error("actions分位数统计数据(q01/q99)缺失，跳过反归一化")
                return actions
            # 维度对齐，取前action_dim维统计值
            q01 = stats.q01[..., :action_dim]
            q99 = stats.q99[..., :action_dim]
            scale = q99 - q01
            scale[scale == 0] = 1e-6  # 除零保护，适配后26维全0的情况
            return (actions + 1.0) / 2.0 * scale + q01
        else:
            # Z-score反归一化（与归一化严格互逆）: x = x_norm * std + mean
            mean = stats.mean[..., :action_dim]
            std = stats.std[..., :action_dim]
            std[std == 0] = 1e-6  # 除零保护，适配后26维全0的情况
            return actions * std + mean
        
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

                # 归一化观测数据
                obs = self.normalize_obs(obs)

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                # action["actions"] = self._unnormalize_actions(action["actions"])

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                print(f"Action: {action}")
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