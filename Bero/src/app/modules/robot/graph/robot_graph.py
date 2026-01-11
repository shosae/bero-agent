from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.config.settings import load_settings
from app.modules.robot.graph.nodes.execute_node import build_action_tools
from app.modules.robot.graph.nodes.plan_node import run_plan_node
from app.modules.robot.graph.nodes.validator_node import run_validator
from app.modules.robot.services.executor_service import ExecutorService
from app.modules.robot.services.mission_summary_service import MissionSummaryService
from app.modules.robot.services.planner_service import PlannerUnsupportedError
from app.modules.robot.services.robot_grpc_client import RobotGrpcClient
from app.shared.llm_factory import LLMConfig, build_llm

from app.shared.path_utils import artifacts_dir

_ARTIFACT_DIR = artifacts_dir()
_VALIDATION_LOG = _ARTIFACT_DIR / "plan_validation.log"
_TRACE_LOG = _ARTIFACT_DIR / "orchestrator_trace.log"
_EXECUTION_LOG = _ARTIFACT_DIR / "execution_logs.log"


def _append_validation_log(
    *,
    question: str,
    plan: Dict[str, Any],
    validation: Any,
    status: str,
) -> None:
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "question": question,
        "plan": plan,
        "errors": list(getattr(validation, "errors", []) or []),
        "warnings": list(getattr(validation, "warnings", []) or []),
        "llm_verdict": getattr(validation, "llm_verdict", None),
        "llm_reason": getattr(validation, "llm_reason", None),
    }
    _VALIDATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _VALIDATION_LOG.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def _append_trace_entry(
    *,
    question: str,
    phase: str,
    payload: Dict[str, Any],
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "phase": phase,
        "payload": payload,
    }
    _TRACE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _TRACE_LOG.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _append_execution_log(
    *,
    question: str,
    logs: List[Dict[str, Any]],
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "logs": logs,
    }
    _EXECUTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _EXECUTION_LOG.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _format_retry_feedback(validation: Any) -> str:
    errors = list(getattr(validation, "errors", []) or [])
    warnings = list(getattr(validation, "warnings", []) or [])
    if not errors and not warnings:
        return ""
    lines = ["이전 PLAN 검증에서 발견된 문제:"]
    if errors:
        lines.append("오류:")
        lines.extend(f"- {err}" for err in errors)
    if warnings:
        lines.append("경고:")
        lines.extend(f"- {warn}" for warn in warnings)
    return "\n".join(lines)


def _format_failure_context(
    *,
    question: str,
    logs: List[Dict[str, Any]],
    error_message: str | None,
    failed_step: Dict[str, Any] | None,
) -> str:
    data = {
        "user_request": question,
        "failed_step": {},
        "error_message": (error_message or "").strip(),
        "log_summary": [],
    }

    if failed_step:
        action = failed_step.get("action", "unknown")
        params = failed_step.get("params", {}) or {}
        target = params.get("target") or params.get("receiver")
        data["failed_step"] = {
            "action": action,
            "target": target,
        }

    if logs:
        for entry in logs[:5]:
            if not isinstance(entry, dict):
                continue
            step = entry.get("step") or {}
            result = entry.get("result") or {}
            action = step.get("action", "unknown")
            params = step.get("params", {}) or {}
            target = params.get("target") or params.get("receiver")
            status = result.get("status", "")
            data["log_summary"].append(
                {
                    "action": action,
                    "target": target,
                    "status": status,
                }
            )

    return json.dumps(data, ensure_ascii=False)

_UNSUPPORTED_ERROR_MARKERS = (
    "허용되지 않은 action",
    "잘못된 target 장소",
)

_UNSUPPORTED_PLAN_MESSAGE = "해당 요청은 아직 지원하지 않고 있어요. 다른 요청을 도와드릴까요?"


def _has_unsupported_plan_error(errors: List[str]) -> bool:
    for err in errors:
        for marker in _UNSUPPORTED_ERROR_MARKERS:
            if marker in err:
                return True
    return False


class RobotState(TypedDict, total=False):
    question: str
    answer: str
    plan: Dict[str, Any]
    validation: Any
    execution_logs: List[Dict[str, Any]]
    plan_queue: List[Dict[str, Any]]
    current_step: Dict[str, Any]
    agent_status: Literal["continue", "done"]
    plan_attempts: int
    plan_status: Literal[
        "pending",
        "ok",
        "retry",
        "failed",
        "unsupported",
    ]
    plan_feedback: str
    mission_status: Literal["pending", "failed", "completed", "unsupported"]
    failure_context: str
    messages: Annotated[list, add_messages]
    plan_history: List[Dict[str, Any]]
    summary: str
    final_status: Literal["completed", "failed", "unsupported"]


def build_robot_body_graph(llm=None, executor: ExecutorService | None = None):
    """브레인에서 직접 호출할 로봇 오케스트레이터 그래프."""
    settings = load_settings()
    llm = llm or build_llm(
        LLMConfig(
            provider=settings.robot_llm_provider,
            model=settings.robot_llm_model,
            temperature=settings.robot_llm_temperature,
            ollama_base_url=settings.ollama_base_url,
            openai_api_key=settings.openai_api_key,
            google_api_key=settings.google_api_key,
            groq_api_key=settings.groq_api_key,
            langgraph_api_key=settings.langgraph_api_key,
            langgraph_base_url=settings.langgraph_base_url,
            
        )
    )

    executor = executor or ExecutorService(
        robot_client=RobotGrpcClient(settings.robot_grpc_target)
    )
    action_tools = build_action_tools(executor)
    summary_service = MissionSummaryService()

    def plan_and_execute(state: RobotState) -> RobotState:
        feedback = (state.get("plan_feedback") or "").strip()
        attempts = int(state.get("plan_attempts") or 0) + 1
        # 각 요청마다 초기화
        base_state = {
            "plan": {},
            "plan_queue": [],
            "execution_logs": [],
            "plan_feedback": "",
            "mission_status": "pending",
            "plan_status": "pending",
            "plan_history": list(state.get("plan_history") or []),
        }
        try:
            result = run_plan_node(
                state.get("question", ""),
                llm,
                retry_feedback=feedback if feedback else None,
            )
        except PlannerUnsupportedError as exc:
            _append_trace_entry(
                question=state.get("question", ""),
                phase="plan",
                payload={
                    "attempt": attempts,
                    "error": "unsupported_plan",
                    "reason": str(exc),
                },
            )
            error_msg = str(exc).strip() or _UNSUPPORTED_PLAN_MESSAGE
            return {
                **base_state,
                "plan_attempts": attempts,
                "plan_status": "unsupported",
                "mission_status": "unsupported",
                "failure_context": _format_failure_context(
                    question=state.get("question", ""),
                    logs=[],
                    error_message=error_msg,
                    failed_step=None,
                ),
            }

        plan_queue = list(result.plan.get("plan", []))
        plan_history = list(state.get("plan_history") or [])
        plan_history.append(
            {
                "question": state.get("question", ""),
                "plan": result.plan,
            }
        )
        _append_trace_entry(
            question=state.get("question", ""),
            phase="plan",
            payload={
                "attempt": attempts,
                "plan": result.plan,
            },
        )
        return {
            **base_state,
            "plan": result.plan,
            "plan_queue": plan_queue,
            "plan_attempts": attempts,
            "plan_status": "pending",
            "plan_history": plan_history,
        }

    MAX_PLAN_RETRIES = 3

    def validate_plan_node(state: RobotState) -> RobotState:
        plan_obj = state.get("plan")
        if state.get("plan_status") == "unsupported":
            # failure_context는 이미 plan_and_execute에서 설정됨
            return {
                "validation": None,
                "plan_status": "unsupported",
                "plan_queue": [],
                "plan_feedback": "",
                "mission_status": "unsupported",
            }
        if not plan_obj:
            return {
                "plan_status": "failed",
                "validation": None,
                "answer": "PLAN 생성 결과가 비어 있어 실행을 종료합니다.",
                "plan_feedback": "",
                "mission_status": "failed",
            }
        validation = run_validator(
            plan_obj,
            question=state.get("question", ""),
            llm=llm,
            extra_context=state.get("extra_context", None),
        )
        question = state.get("question", "")
        llm_verdict = getattr(validation, "llm_verdict", None)
        llm_reason = getattr(validation, "llm_reason", "")

        if llm_verdict == "UNSUPPORTED":
            status = "unsupported"
        else:
            status = "ok" if validation.is_valid else None
            attempts = int(state.get("plan_attempts") or 0)
            if not validation.is_valid:
                status = "retry" if attempts < MAX_PLAN_RETRIES else "failed"
            if not validation.is_valid and _has_unsupported_plan_error(validation.errors):
                status = "unsupported"

        _append_validation_log(
            question=question,
            plan=plan_obj,
            validation=validation,
            status=status or "unknown",
        )
        _append_trace_entry(
            question=question,
            phase="validation",
            payload={
                "status": status,
                "errors": list(validation.errors),
                "warnings": list(validation.warnings),
                "llm_verdict": getattr(validation, "llm_verdict", None),
                "llm_reason": getattr(validation, "llm_reason", None),
            },
        )

        if llm_verdict == "UNSUPPORTED":
            reason_text = (llm_reason or "").strip()
            base = "죄송하지만 이 요청을 수행할 수 없습니다. 다른 요청을 해주시겠어요?"
            answer = f"{base}\n\n상세: {reason_text}" if reason_text else base
            return {
                "validation": validation,
                "plan_status": "unsupported",
                "plan_queue": [],
                "plan_feedback": "",
                "mission_status": "unsupported",
                "failure_context": _format_failure_context(
                    question=question,
                    logs=state.get("execution_logs") or [],
                    error_message=answer,
                    failed_step=None,
                ),
            }

        if status == "unsupported":
            return {
                "validation": validation,
                "plan_status": "unsupported",
                "plan_queue": [],
                "plan_feedback": "",
                "mission_status": "unsupported",
                "failure_context": _format_failure_context(
                    question=question,
                    logs=state.get("execution_logs") or [],
                    error_message=_UNSUPPORTED_PLAN_MESSAGE,
                    failed_step=None,
                ),
            }

        if status == "ok":
            return {"validation": validation, "plan_status": "ok", "plan_feedback": ""}

        if status == "retry":
            feedback = _format_retry_feedback(validation)
            return {
                "validation": validation,
                "plan_status": "retry",
                "plan_queue": [],
                "plan_feedback": feedback,
                "mission_status": "pending",
            }

        error_text = "; ".join(validation.errors)
        return {
            "validation": validation,
            "plan_status": "failed",
            "plan_feedback": "",
            "mission_status": "failed",
            "failure_context": _format_failure_context(
                question=question,
                logs=state.get("execution_logs") or [],
                error_message=error_text,
                failed_step=None,
            ),
        }

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
        if tool is None:
            return {
                "execution_logs": state.get("execution_logs") or [],
                "answer": f"Unsupported action {action}",
                "agent_status": "done",
                "plan_queue": [],
                "mission_status": "failed",
            }
        result = tool.invoke(params)
        logs = (state.get("execution_logs") or []) + [{"step": step, "result": result}]
        _append_trace_entry(
            question=state.get("question", ""),
            phase="execution_step",
            payload={
                "step": step,
                "result": result,
                "remaining_queue": state.get("plan_queue") or [],
            },
        )
        status = (result.get("status") or "").lower()
        if status and status not in {"success", "reported"}:
            failure_text = _format_failure_context(
                question=state.get("question", ""),
                logs=logs,
                error_message=result.get("message"),
                failed_step=step,
            )
            _append_execution_log(
                question=state.get("question", ""),
                logs=logs,
            )
            return {
                "execution_logs": logs,
                "current_step": {},
                "plan_queue": [],
                "agent_status": "done",
                "mission_status": "failed",
                "failure_context": failure_text,
            }
        if not state.get("plan_queue"):
            _append_execution_log(
                question=state.get("question", ""),
                logs=logs,
            )
        return {"execution_logs": logs, "current_step": {}, "mission_status": "completed"}

    def summarize(state: RobotState) -> RobotState:
        logs = state.get("execution_logs") or []
        status = state.get("mission_status") or "completed"
        plan_obj = state.get("plan") or {}
        failure_ctx = state.get("failure_context", "")
        
        # failure_context가 있으면 failure_reason으로 전달
        failure_reason = failure_ctx if failure_ctx else None
        
        summary_text = summary_service.build_summary(
            question=state.get("question", ""),
            plan=plan_obj,
            execution_logs=logs,
            llm=llm,
            failure_reason=failure_reason,
        )
        
        return {
            "summary": summary_text,
            "answer": summary_text,
            "final_status": "unsupported" if status == "unsupported" else ("failed" if status == "failed" else "completed"),
            "plan_queue": [],
            "current_step": {},
        }

    graph = StateGraph(RobotState)
    graph.add_node("plan", plan_and_execute)
    graph.add_node("validate_plan", validate_plan_node)
    graph.add_node("executor", executor_node)
    graph.add_node("execute_step", execute_step)
    graph.add_node("summarize", summarize)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "validate_plan")
    graph.add_conditional_edges(
        "validate_plan",
        lambda s: s.get("plan_status", "ok"),
        {
            "ok": "executor",
            "retry": "plan",
            "failed": "summarize",
            "unsupported": "summarize",
        },
    )
    graph.add_conditional_edges(
        "executor",
        lambda s: (
            "failure"
            if s.get("mission_status") == "failed"
            else s.get("agent_status", "done")
        ),
        {"continue": "execute_step", "done": "summarize", "failure": "summarize"},
    )
    graph.add_edge("execute_step", "executor")
    graph.add_edge("summarize", END)
    return graph.compile()
