"""LangGraph wrapper for the Master Agent with tool-calling nodes."""

from __future__ import annotations

import json
from typing import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from app.master.agent import SYSTEM_PROMPT, _build_llm, _parse_json
from app.tools import calendar_tool, gmail_tool, rag_tool, robot_mission_tool


class MasterState(TypedDict, total=False):
    question: str
    tool: str | None
    tool_input: dict | None
    tool_output: str
    answer: str


def _decide(state: MasterState) -> MasterState:
    llm = _build_llm()
    question = state.get("question", "")
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    resp = llm.invoke(messages)
    payload = _parse_json(getattr(resp, "content", ""))
    tool = payload.get("tool")
    tool_input = payload.get("tool_input") or {}
    final_answer = payload.get("final_answer")
    base = {
        "tool": None,
        "tool_input": {},
        "tool_output": "",
        "answer": "",
    }
    if tool:
        base.update({"tool": tool, "tool_input": tool_input})
        return base
    base["answer"] = final_answer or json.dumps(payload, ensure_ascii=False)
    return base


def _run_specific_tool(tool_callable):
    def _runner(state: MasterState) -> MasterState:
        tool_input = state.get("tool_input") or {}
        try:
            output = tool_callable.invoke(tool_input)
        except Exception as exc:  # pragma: no cover
            output = f"도구 실행 오류: {exc}"
        return {"tool_output": output}

    return _runner


def _summarize(state: MasterState) -> MasterState:
    llm = _build_llm()
    question = state.get("question", "")
    tool_output = state.get("tool_output", "")
    prompt = f"""다음은 사용자 요청과 도구 실행 결과입니다.
사용자 요청: {question}
도구 결과: {tool_output}

[지침]
- 도구 결과에 없는 정보는 만들지 말고, 그대로 요약하거나 그대로 전달합니다.
- 도구 결과가 비어 있으면 "도구에서 제공된 정보가 없습니다." 정도로 간단히 알려줍니다.
- 한두 문장 한국어로만 답변합니다. 목록/JSON/코드블록은 사용하지 않습니다.
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    content = getattr(resp, "content", resp)
    try:
        payload = json.loads(content)
        return {"answer": payload.get("final_answer") or content}
    except Exception:
        return {"answer": content}


def build_master_graph():
    graph = StateGraph(MasterState)
    graph.add_node("decide", _decide)
    graph.add_node("run_calendar", _run_specific_tool(calendar_tool))
    graph.add_node("run_gmail", _run_specific_tool(gmail_tool))
    graph.add_node("run_rag", _run_specific_tool(rag_tool))
    graph.add_node("run_robot", _run_specific_tool(robot_mission_tool))
    graph.add_node("summarize", _summarize)
    graph.set_entry_point("decide")
    graph.add_conditional_edges(
        "decide",
        lambda s: s.get("tool") or "done",
        {
            "calendar_tool": "run_calendar",
            "gmail_tool": "run_gmail",
            "rag_tool": "run_rag",
            "robot_mission_tool": "run_robot",
            "done": END,
        },
    )
    graph.add_edge("run_calendar", "summarize")
    graph.add_edge("run_gmail", "summarize")
    graph.add_edge("run_rag", "summarize")
    graph.add_edge("run_robot", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()
