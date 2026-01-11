from __future__ import annotations

from langchain_core.messages import AIMessage
from app.modules.master.services.executor_service import execute_plan


def run_execute_plan(state, *, llm):
    """계획된 스텝들을 순서대로 실행."""

    question = state.get("question", "")
    steps = state.get("steps") or []
    
    result = execute_plan(question, steps, llm=llm)
    final_answer = result.get("answer") or ""

    return {
        "execution_logs": result.get("execution_logs") or [],
        "answer": final_answer,
        "messages": [AIMessage(content=final_answer)]
    }