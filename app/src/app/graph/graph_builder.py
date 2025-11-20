"""LangGraph graph builder."""

from __future__ import annotations

from typing import Dict, Any, List, Literal, TypedDict

from langgraph.graph import StateGraph, END

from app.graph.nodes.plan_node import run_plan_node
from app.graph.nodes.execute_node import build_action_tools
from app.graph.nodes.conversation_node import run_conversation_node
from app.graph.nodes.intent_node import classify_intent as classify
from app.graph.nodes.validator_node import run_validator
from app.services.executor_service import ExecutorService


class OrchestratorState(TypedDict, total=False):
    question: str
    mode: Literal["conversation", "plan"]
    answer: str
    plan: Dict[str, Any]
    validation: Any
    execution_logs: List[Dict[str, Any]]
    plan_queue: List[Dict[str, Any]]
    current_step: Dict[str, Any]
    agent_status: Literal["continue", "done"]
    plan_attempts: int
    plan_status: Literal["pending", "ok", "retry", "failed"]


def build_orchestrator_graph(llm, retriever, executor: ExecutorService):
    """간단한 계획/대화 그래프를 구성한다."""

    action_tools = build_action_tools(executor)

    def classify_intent(state: OrchestratorState) -> OrchestratorState:
        question = state.get("question", "")
        mode = classify(question)
        return {"mode": mode}

    def conversation(state: OrchestratorState) -> OrchestratorState:
        answer = run_conversation_node(state.get("question", ""), llm, retriever)
        return {"answer": answer}

    def plan_and_execute(state: OrchestratorState) -> OrchestratorState:
        result = run_plan_node(state.get("question", ""), llm, [])
        plan_queue = list(result.plan.get("plan", []))
        attempts = int(state.get("plan_attempts") or 0) + 1
        return {
            "plan": result.plan,
            "plan_queue": plan_queue,
            "plan_attempts": attempts,
            "plan_status": "pending",
        }

    MAX_PLAN_RETRIES = 3

    def validate_plan_node(state: OrchestratorState) -> OrchestratorState:
        plan_obj = state.get("plan")
        if not plan_obj:
            return {
                "plan_status": "failed",
                "validation": None,
                "answer": "PLAN 생성 결과가 비어 있어 실행을 종료합니다.",
            }
        validation = run_validator(
            plan_obj,
            question=state.get("question", ""),
            llm=llm,
        )
        if validation.is_valid:
            return {"validation": validation, "plan_status": "ok"}

        attempts = int(state.get("plan_attempts") or 0)
        if attempts < MAX_PLAN_RETRIES:
            return {
                "validation": validation,
                "plan_status": "retry",
                "plan_queue": [],
            }

        error_text = "; ".join(validation.errors)
        return {
            "validation": validation,
            "plan_status": "failed",
            "answer": f"PLAN 검증 실패로 중단합니다: {error_text}",
        }

    def executor(state: OrchestratorState) -> OrchestratorState:
        queue = state.get("plan_queue") or []
        if not queue:
            return {"agent_status": "done", "plan_queue": []}
        next_step = queue[0]
        remaining = queue[1:]
        return {
            "current_step": next_step,
            "plan_queue": remaining,
            "agent_status": "continue",
        }

    def execute_current_step(state: OrchestratorState) -> OrchestratorState:
        step = state.get("current_step") or {}
        action = step.get("action")
        params = step.get("params") or {}
        tool = action_tools.get(action)
        if tool is None:
            result = {"status": "error", "message": f"Unsupported action {action}"}
        else:
            result = tool.invoke(params)
        logs = (state.get("execution_logs") or []) + [{"step": step, "result": result}]
        return {"execution_logs": logs, "current_step": {}}

    graph = StateGraph(OrchestratorState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("conversation", conversation)
    graph.add_node("plan", plan_and_execute)
    graph.add_node("validate_plan", validate_plan_node)
    graph.add_node("executor", executor)
    graph.add_node("execute_step", execute_current_step)
    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        lambda s: s.get("mode", "conversation"),
        {"conversation": "conversation", "plan": "plan"},
    )
    graph.add_edge("conversation", END)
    graph.add_edge("plan", "validate_plan")
    graph.add_conditional_edges(
        "validate_plan",
        lambda s: s.get("plan_status", "ok"),
        {"ok": "executor", "retry": "plan", "failed": END},
    )
    graph.add_conditional_edges(
        "executor",
        lambda s: s.get("agent_status", "done"),
        {"continue": "execute_step", "done": END},
    )
    graph.add_edge("execute_step", "executor")
    return graph.compile()
