"""LangGraph subgraph for robot missions (Gatekeeper -> Plan -> Validate -> Execute)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, TypedDict

from langgraph.graph import StateGraph, END

from app.graph.nodes.intent_node import classify_intent as classify
from app.graph.nodes.plan_node import run_plan_node
from app.graph.nodes.validator_node import run_validator
from app.graph.nodes.execute_node import build_action_tools
from app.services.executor_service import ExecutorService
from app.services.mission_summary_service import MissionSummaryService
from app.services.planner_service import PlannerUnsupportedError
from app.services.llm_service import build_llm, LLMConfig
from app.services.robot_grpc_client import RobotGrpcClient
from app.config.settings import load_settings


class RobotState(TypedDict, total=False):
    question: str
    plan: Dict[str, Any]
    plan_queue: List[Dict[str, Any]]
    current_step: Dict[str, Any]
    execution_logs: List[Dict[str, Any]]
    agent_status: Literal["continue", "done"]
    plan_status: Literal["pending", "ok", "retry", "failed", "unsupported"]
    plan_attempts: int
    plan_feedback: str
    answer: str


def build_robot_body_graph(llm=None, executor: ExecutorService | None = None):
    settings = load_settings()
    llm = llm or build_llm(
        LLMConfig(
            provider=settings.llm_provider,
            model=settings.llm_model,
            temperature=settings.temperature,
            langgraph_api_key=os.getenv("LANGGRAPH_API_KEY"),
            langgraph_base_url=os.getenv("LANGGRAPH_BASE_URL"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL"),
        )
    )
    executor = executor or ExecutorService(
        robot_client=RobotGrpcClient(settings.robot_grpc_target)
    )
    action_tools = build_action_tools(executor)
    summary_service = MissionSummaryService()

    def gatekeeper(state: RobotState) -> RobotState:
        question = state.get("question", "")
        prediction = classify(question, llm=llm)
        if prediction.conversation_mode == "unsupported" or prediction.intent != "plan":
            reason = prediction.reason or "이 요청은 수행할 수 없습니다."
            return {"answer": f"거부됨: {reason}", "agent_status": "done"}
        return {
            "plan_attempts": 0,
            "plan_feedback": "",
            "execution_logs": [],
            "agent_status": "continue",
        }

    def plan(state: RobotState) -> RobotState:
        attempts = int(state.get("plan_attempts") or 0) + 1
        try:
            result = run_plan_node(
                state.get("question", ""),
                llm,
                [],
                retry_feedback=None,
            )
        except PlannerUnsupportedError as exc:
            return {
                "answer": str(exc).strip() or "지원하지 않는 요청입니다.",
                "agent_status": "done",
                "plan_status": "unsupported",
            }
        plan_queue = list(result.plan.get("plan", []))
        return {
            "plan": result.plan,
            "plan_queue": plan_queue,
            "plan_attempts": attempts,
            "plan_status": "pending",
        }

    def validate(state: RobotState) -> RobotState:
        plan_obj = state.get("plan")
        if not plan_obj:
            return {
                "plan_status": "failed",
                "answer": "PLAN 생성 결과가 비어 있습니다.",
                "agent_status": "done",
            }
        validation = run_validator(
            plan_obj,
            question=state.get("question", ""),
            llm=llm,
            extra_context=None,
        )
        if getattr(validation, "llm_verdict", None) == "UNSUPPORTED":
            reason = getattr(validation, "llm_reason", "") or "지원하지 않는 요청입니다."
            return {
                "plan_status": "unsupported",
                "answer": f"거부됨: {reason}",
                "agent_status": "done",
            }
        if not validation.is_valid:
            errors = "; ".join(validation.errors)
            return {
                "plan_status": "failed",
                "answer": f"PLAN 검증 실패: {errors}",
                "agent_status": "done",
            }
        return {"plan_status": "ok"}

    def executor_node(state: RobotState) -> RobotState:
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

    def execute_step(state: RobotState) -> RobotState:
        step = state.get("current_step") or {}
        action = step.get("action")
        params = step.get("params") or {}
        tool = action_tools.get(action)
        if action == "summarize_mission":
            params = dict(params)
            params["summary"] = summary_service.build_summary(
                question=state.get("question", ""),
                plan=state.get("plan"),
                execution_logs=state.get("execution_logs") or [],
                llm=llm,
            )
            tool = action_tools.get(action)
        if tool is None:
            return {
                "answer": f"Unsupported action {action}",
                "agent_status": "done",
                "plan_queue": [],
            }
        result = tool.invoke(params)
        logs = (state.get("execution_logs") or []) + [{"step": step, "result": result}]
        status = (result.get("status") or "").lower()
        if status and status not in {"success", "reported"}:
            message = result.get("message") or "실패했습니다."
            return {
                "execution_logs": logs,
                "answer": f"{action} 단계 실패: {message}",
                "plan_queue": [],
                "agent_status": "done",
            }
        if not state.get("plan_queue"):
            final_msg = result.get("message") or "임무를 완료했습니다."
            return {
                "execution_logs": logs,
                "answer": final_msg,
                "agent_status": "done",
                "plan_queue": [],
            }
        return {"execution_logs": logs}

    graph = StateGraph(RobotState)
    graph.add_node("gatekeeper", gatekeeper)
    graph.add_node("plan", plan)
    graph.add_node("validate", validate)
    graph.add_node("executor", executor_node)
    graph.add_node("execute_step", execute_step)
    graph.set_entry_point("gatekeeper")
    graph.add_conditional_edges(
        "gatekeeper",
        lambda s: "done" if s.get("agent_status") == "done" else "plan",
        {"done": END, "plan": "plan"},
    )
    graph.add_edge("plan", "validate")
    graph.add_conditional_edges(
        "validate",
        lambda s: s.get("plan_status", "failed"),
        {"ok": "executor", "failed": END, "unsupported": END},
    )
    graph.add_conditional_edges(
        "executor",
        lambda s: s.get("agent_status", "done"),
        {"continue": "execute_step", "done": END},
    )
    graph.add_edge("execute_step", "executor")
    return graph.compile()
