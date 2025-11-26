"""Plan 단계별 tool 실행 서비스."""

from __future__ import annotations

from typing import Dict, Any, Optional

from app.services.navigation_service import NavigationBackend
from app.services.robot_grpc_client import RobotGrpcClient


class ExecutorService:
    """LangGraph plan executor가 호출할 action 핸들러."""

    def __init__(
        self,
        navigation_backend: Optional[NavigationBackend] = None,
        robot_client: RobotGrpcClient | None = None,
    ) -> None:
        self.navigation_backend = navigation_backend
        self.robot_client = robot_client

    def navigate(self, target: str) -> Dict[str, Any]:
        if self.robot_client:
            try:
                success, message = self.robot_client.navigate(target)
                return {
                    "status": "success" if success else "error",
                    "message": message or "",
                    "target": target,
                }
            except Exception as exc:  # pragma: no cover - network failure path
                return {
                    "status": "error",
                    "message": f"gRPC navigate 실패: {exc}",
                    "target": target,
                }
        if self.navigation_backend:
            return self.navigation_backend.navigate(target)
        return {
            "status": "success",
            "message": f"Navigated to {target}",
            "target": target,
        }

    def observe_scene(self, target: str, query: str | None = None) -> Dict[str, Any]:
        prompt = query or f"{target}의 상황을 설명해줘."
        if self.robot_client:
            try:
                success, description = self.robot_client.describe_scene(prompt)
                base_message = description or "장면 설명이 비었습니다."
                return {
                    "status": "success" if success else "error",
                    "message": base_message,
                    "target": target,
                }
            except Exception as exc:  # pragma: no cover - network failure path
                return {
                    "status": "error",
                    "message": f"gRPC observe 실패: {exc}",
                    "target": target,
                }
        return {
            "status": "success",
            "message": f"Observed {target}: {query or '환경 상태를 관찰'}",
            "target": target,
        }

    def deliver_object(self, receiver: str | None = None, item: str | None = None) -> Dict[str, Any]:
        """
        이동은 이미 Navigate로 수행했으므로, 
        여기서는 로봇 화면에 확인창만 띄우고 기다립니다.
        """
        if self.robot_client:
            try:
                success, message = self.robot_client.deliver()
                
                return {
                    "status": "success" if success else "error",
                    "message": message,  # "Delivery Confirmed"
                    "target": receiver,
                }
            except Exception as exc:
                return {
                    "status": "error",
                    "message": f"gRPC deliver 실패: {exc}",
                    "target": receiver,
                }
        
        return {
            "status": "success",
            "message": f"[Simulated] Delivered item to {receiver}",
        }

    def summarize_mission(self, summary: str | None = None) -> Dict[str, Any]:
        return {
            "status": "reported",
            "message": summary or "임무 요약이 준비되었습니다.",
        }

    def wait(self) -> Dict[str, Any]:
        """사용자 지시에 따라 단순 대기."""
        return {
            "status": "success",
            "message": "지시에 따라 대기합니다.",
        }
