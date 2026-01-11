#!/usr/bin/env python3
"""Robot Brain gRPC 서버.

로봇에서 올라온 자연어 명령을 LangGraph 오케스트레이터에 전달하고,
실행 결과 요약을 다시 로봇에게 돌려준다.
"""

from __future__ import annotations

from concurrent import futures
import logging
from pathlib import Path
import sys
from typing import Any, Mapping, MutableMapping, Sequence

from dotenv import load_dotenv

import grpc

import bero_pb2
import bero_pb2_grpc


# .env를 먼저 불러와 LangGraph dev와 동일한 환경 변수를 적용
_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env", override=False)

_SRC_DIR = _ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.append(str(_SRC_DIR))

from agent.master_graph_builder import server_graph as master_graph

LOGGER = logging.getLogger("brain_server")


def _build_reply(state: Mapping[str, Any] | None) -> str:
    """그래프 최종 state에서 로봇이 말할 답변을 추출."""
    default_reply = "요청하신 작업을 처리했습니다."
    if not isinstance(state, Mapping):
        return default_reply

    # Master Graph는 answer 필드를 사용
    answer = str(state.get("answer") or "").strip()
    if answer:
        return answer

    # execution_logs가 있으면 로깅 (디버깅용)
    execution_logs = state.get("execution_logs")
    if execution_logs:
        LOGGER.info("Execution logs: %s", execution_logs)

    return default_reply


class RobotBrainService(bero_pb2_grpc.RobotBrainServicer):
    """RobotBrain RPC 구현체."""

    def __init__(self) -> None:
        self._graph = master_graph

    def ProcessCommand(self, request, context):
        text = (request.text or "").strip()
        # 클라이언트가 보낸 session_id가 있다면 사용, 없으면 디폴트값
        # 실제로는 request.session_id 같은 필드가 proto에 있어야 함
        user_id = getattr(request, "user_id", "default_user") 
        
        LOGGER.info("[Brain] ProcessCommand 호출: %s (User: %s)", text, user_id)
        
        if not text:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("text 필드가 비었습니다.")
            return bero_pb2.ChatResponse(reply="무엇을 도와드릴까요?")

        try:
            # thread_id 설정 필수!
            config = {"configurable": {"thread_id": user_id}}
            
            # invoke에 config 전달
            state = self._graph.invoke(
                {"question": text}, 
                config=config
            )
            reply = _build_reply(state)
        
        except Exception as exc:  # pragma: no cover - runtime failure path
            LOGGER.exception("LangGraph 실행 실패")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"LangGraph 실행에 실패했습니다: {exc}")
            reply = "처리 중 문제가 발생했어요. 잠시 후 다시 시도해 주세요."

        return bero_pb2.ChatResponse(reply=reply)


def serve() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    bero_pb2_grpc.add_RobotBrainServicer_to_server(RobotBrainService(), server)

    listen_addr = "[::]:50052"  # Brain 서비스 포트
    server.add_insecure_port(listen_addr)
    LOGGER.info("[Brain] RobotBrain gRPC 서버 시작: %s", listen_addr)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
