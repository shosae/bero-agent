"""LangGraph graph builder."""

from __future__ import annotations

from typing import Dict, Any, List, Literal, TypedDict

from langgraph.graph import StateGraph, END

from app.graph.nodes.plan_node import run_plan_node
from app.graph.nodes.execute_node import build_action_tools
from app.graph.nodes.conversation_node import run_conversation_node
from app.graph.nodes.intent_node import classify_intent
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


def build_orchestrator_graph(llm, retriever, executor: ExecutorService):
    """간단한 계획/대화 그래프를 구성한다."""

    action_tools = build_action_tools(executor)

    def entry(state: OrchestratorState) -> OrchestratorState:
        question = state.get("question", "")
        mode = classify_intent(question)
        return {"mode": mode}

    def conversation(state: OrchestratorState) -> OrchestratorState:
        answer = run_conversation_node(state.get("question", ""), llm, retriever)
        return {"answer": answer}

    def plan_and_execute(state: OrchestratorState) -> OrchestratorState:
        result = run_plan_node(state.get("question", ""), llm, [])
        plan_queue = list(result.plan.get("plan", []))
        return {
            "plan": result.plan,
            "validation": result.validation,
            "plan_queue": plan_queue,
            "execution_logs": [],
        }

    def plan_agent(state: OrchestratorState) -> OrchestratorState:
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
    graph.add_node("entry", entry)
    graph.add_node("conversation", conversation)
    graph.add_node("plan", plan_and_execute)
    graph.add_node("plan_agent", plan_agent)
    graph.add_node("execute_step", execute_current_step)
    graph.set_entry_point("entry")
    graph.add_conditional_edges(
        "entry",
        lambda s: s.get("mode", "conversation"),
        {"conversation": "conversation", "plan": "plan"},
    )
    graph.add_edge("conversation", END)
    graph.add_edge("plan", "plan_agent")
    graph.add_conditional_edges(
        "plan_agent",
        lambda s: s.get("agent_status", "done"),
        {"continue": "execute_step", "done": END},
    )
    graph.add_edge("execute_step", "plan_agent")
    return graph.compile()
