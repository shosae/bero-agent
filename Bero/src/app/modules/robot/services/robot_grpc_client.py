from __future__ import annotations

from app.shared.path_utils import grpc_dir
import sys
from typing import Tuple

import grpc

_PB2_DIR = grpc_dir()
if str(_PB2_DIR) not in sys.path:
    sys.path.append(str(_PB2_DIR))

import bero_pb2
import bero_pb2_grpc


class RobotGrpcClient:
    """Lazy gRPC stub wrapper for the remote robot controller."""

    def __init__(self, target: str) -> None:
        self._target = target
        self._channel: grpc.Channel | None = None
        self._stub: bero_pb2_grpc.RobotControllerStub | None = None

    def _get_stub(self) -> bero_pb2_grpc.RobotControllerStub:
        if self._stub is None:
            self._channel = grpc.insecure_channel(self._target)
            self._stub = bero_pb2_grpc.RobotControllerStub(self._channel)
        return self._stub

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def navigate(self, waypoint: str) -> Tuple[bool, str]:
        """Invoke Navigate RPC."""
        request = bero_pb2.NavigateRequest(waypoint=waypoint)
        stub = self._get_stub()
        response = stub.Navigate(request)
        return response.success, response.message

    def describe_scene(self, prompt: str) -> Tuple[bool, str]:
        """Invoke DescribeScene RPC."""
        request = bero_pb2.SceneRequest(prompt=prompt)
        stub = self._get_stub()
        response = stub.DescribeScene(request)
        return response.success, response.description

    def deliver(self) -> Tuple[bool, str]:
        """
        Invoke Deliver RPC with a FIXED message.
        Blocks until user presses 'Confirm' on robot UI.
        """
        # 메시지 고정 (User 요청 사항)
        fixed_msg = "도착했습니다.\n수령 확인 버튼을 눌러주세요."
        
        request = bero_pb2.DeliverRequest(message=fixed_msg)
        stub = self._get_stub()
        
        # 로봇 UI에서 버튼 눌릴 때까지 대기 (Blocking)
        response = stub.Deliver(request)
        
        return response.success, "Delivery Confirmed"
