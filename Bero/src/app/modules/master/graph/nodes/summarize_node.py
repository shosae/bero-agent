from __future__ import annotations

from app.modules.master.services.summarize_service import summarize_result


def run_summarize(state, *, llm):
    """툴 실행 출력 요약."""

    question = state.get("question", "")
    tool_output = state.get("tool_output", "")
    tool_name = state.get("tool") or ""
    return summarize_result(
        question=question,
        tool_name=tool_name,
        tool_output=tool_output,
        llm=llm,
    )
