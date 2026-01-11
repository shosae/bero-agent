from __future__ import annotations

from langchain_core.tools import tool

from app.modules.robot.graph.robot_graph import build_robot_body_graph

_ROBOT_GRAPH = build_robot_body_graph()


@tool("robot_mission_tool")
def robot_mission_tool(instruction: str) -> str:
    """
    물리적인 로봇 이동/관찰/배달이 필요할 때 호출.
    LangGraph 기반 로봇 서브그래프를 실행.
    """
    try:
        result = _ROBOT_GRAPH.invoke({"question": instruction})
        return result.get("answer") or "응답이 없습니다."
    except Exception as exc:  # pragma: no cover - defensive
        return f"로봇 미션 실행 중 오류: {exc}"
