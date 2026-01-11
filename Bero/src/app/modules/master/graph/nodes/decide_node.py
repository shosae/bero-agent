from __future__ import annotations

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.modules.master.services.decision_service import MasterAgentToolService


def run_decide(state, *, llm):
    """LLM 기반 도구 선택."""

    question = state.get("question", "")
    messages = state.get("messages", [])
    
    execution_logs = state.get("execution_logs", [])

    conversation_messages = list(messages)
    
    if execution_logs:
        logs_text = "\n".join(
            f"[이전 도구 실행 결과]\nTool: {log.get('tool')}\nOutput: {log.get('output')}" 
            for log in execution_logs
        )
        conversation_messages.append(SystemMessage(content=logs_text))

    if question:
        if not messages or messages[-1].content != question:
            conversation_messages.append(HumanMessage(content=question))

    service = MasterAgentToolService()
    decision = service.decide(question, llm=llm, messages=conversation_messages)
    
    output = {
        "steps": decision.steps, 
        "answer": decision.answer,
        "messages": []
    }

    if question and (not messages or messages[-1].content != question):
        output["messages"].append(HumanMessage(content=question))

    if decision.answer:
        output["messages"].append(AIMessage(content=decision.answer))
    
    return output