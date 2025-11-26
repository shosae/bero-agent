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

import grpc

import bero_pb2
import bero_pb2_grpc


_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.append(str(_SRC_DIR))

from agent.graph import graph as orchestrator_graph  # type: ignore  # noqa: E402


LOGGER = logging.getLogger("brain_server")


def _extract_summary_from_logs(logs: Sequence[Mapping[str, Any]] | None) -> str:
    """최종 summarize_mission 로그를 찾아 사용자 응답으로 사용한다."""
    if not logs:
        return ""

    def _pick_message(entry: Mapping[str, Any] | MutableMapping[str, Any]) -> str:
        result = entry.get("result") if isinstance(entry, Mapping) else None
        if isinstance(result, Mapping):
            return str(result.get("message") or "").strip()
        return ""

    for entry in reversed(logs):
        step = entry.get("step") if isinstance(entry, Mapping) else None
        if isinstance(step, Mapping) and step.get("action") == "summarize_mission":
            message = _pick_message(entry)
            if message:
                return message

    for entry in reversed(logs):
        message = _pick_message(entry)
        if message:
            return message
    return ""


def _build_reply(state: Mapping[str, Any] | None) -> str:
    """그래프 최종 state에서 로봇이 말할 답변을 추출한다."""
    default_reply = "요청하신 작업을 처리했습니다."
    if not isinstance(state, Mapping):
        return default_reply

    answer = str(state.get("answer") or "").strip()
    if answer:
        return answer

    summary = _extract_summary_from_logs(state.get("execution_logs"))
    if summary:
        return summary

    failure_context = str(state.get("failure_context") or "").strip()
    if failure_context:
        LOGGER.warning("Failure context returned to robot: %s", failure_context)

    return default_reply


class RobotBrainService(bero_pb2_grpc.RobotBrainServicer):
    """RobotBrain RPC 구현체."""

    def __init__(self) -> None:
        self._graph = orchestrator_graph

    def ProcessCommand(self, request, context):
        text = (request.text or "").strip()
        LOGGER.info("[Brain] ProcessCommand 호출: %s", text)

        if not text:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("text 필드가 비었습니다.")
            return bero_pb2.ChatResponse(reply="무엇을 도와드릴까요?")

        try:
            state = self._graph.invoke({"question": text})
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
