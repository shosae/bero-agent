from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

from app.modules.robot.services.planner_service import generate_plan, PlannerUnsupportedError
from app.modules.robot.services.validator_service import CompositeValidatorService
from app.modules.robot.services.executor_service import ExecutorService
from app.modules.robot.services.robot_grpc_client import RobotGrpcClient
from app.modules.robot.services.mission_summary_service import MissionSummaryService
from app.config.settings import load_settings
from app.shared.llm_factory import LLMConfig, build_llm
from app.shared.path_utils import project_root


def _execute_plan(
    plan: Dict[str, Any],
    executor: ExecutorService,
) -> Dict[str, Any]:
    """Execute plan steps sequentially."""
    steps = plan.get("plan", [])
    logs = []

    for idx, step in enumerate(steps):
        action = step.get("action", "")
        params = step.get("params", {}) or {}

        try:
            if action == "navigate":
                result = executor.navigate(params.get("target", ""))
            elif action == "observe_scene":
                result = executor.observe_scene(
                    params.get("target", ""),
                    params.get("question"),
                )
            elif action == "deliver_object":
                result = executor.deliver_object(params.get("receiver"))
            elif action == "wait":
                result = executor.wait()
            else:
                result = {"status": "error", "message": f"Unknown action: {action}"}

            logs.append({
                "step": step,
                "result": result,
            })

            if result.get("status") == "error":
                return {
                    "status": "failed",
                    "logs": logs,
                    "error": result.get("message", ""),
                }

        except Exception as exc:
            logs.append({
                "step": step,
                "result": {"status": "error", "message": str(exc)},
            })
            return {
                "status": "failed",
                "logs": logs,
                "error": str(exc),
            }

    return {
        "status": "completed",
        "logs": logs,
    }


def run_robot(
    question: str,
    *,
    llm=None,
    executor: ExecutorService | None = None,
    max_retries: int = 2,
    verbose: bool = True,
) -> str:
    """Main robot control loop: plan → validate → execute → summarize."""

    settings = load_settings()
    if llm is None:
        llm = build_llm(
            LLMConfig(
                provider=settings.robot_llm_provider,
                model=settings.robot_llm_model,
                temperature=settings.robot_llm_temperature,
                openai_api_key=settings.openai_api_key,
                google_api_key=settings.google_api_key,
                langgraph_api_key=settings.langgraph_api_key,
                langgraph_base_url=settings.langgraph_base_url,
                groq_api_key=settings.groq_api_key,
                ollama_base_url=settings.ollama_base_url,
            )
        )
    executor = executor or ExecutorService(
        robot_client=RobotGrpcClient(settings.robot_grpc_target)
    )
    validator = CompositeValidatorService()
    summary_service = MissionSummaryService()

    if verbose:
        print(f"\n{'='*60}")
        print(f"🤖 Robot Agent 시작")
        print(f"{'='*60}")
        print(f"📝 사용자 요청: {question}")
        print(f"{'='*60}\n")

    # 1) Plan generation
    if verbose:
        print("📋 STEP 1: Plan Generation")
        print("-" * 60)
    
    for attempt in range(max_retries):
        try:
            plan, _ = generate_plan(question, llm=llm)
            if verbose:
                print(f"✅ Plan 생성 성공 (attempt {attempt + 1}/{max_retries})")
                print(f"📄 Plan 내용:")
                print(json.dumps(plan, indent=2, ensure_ascii=False))
                print()
            break
        except PlannerUnsupportedError as exc:
            if verbose:
                print(f"❌ 지원하지 않는 요청: {exc}\n")
            return json.dumps(
                {"status": "unsupported", "message": str(exc)},
                ensure_ascii=False,
            )
        except Exception as exc:
            if verbose:
                print(f"⚠️  Plan 생성 실패 (attempt {attempt + 1}/{max_retries}): {exc}")
            if attempt == max_retries - 1:
                if verbose:
                    print(f"❌ 최대 재시도 횟수 초과\n")
                return json.dumps(
                    {"status": "failed", "error": f"Plan generation failed: {exc}"},
                    ensure_ascii=False,
                )

    # 2) Validation
    if verbose:
        print(f"\n{'='*60}")
        print("✔️  STEP 2: Plan Validation")
        print("-" * 60)
    
    validation = validator.validate(plan, question=question, llm=llm)
    
    if verbose:
        if validation.is_valid:
            print("✅ Validation 통과")
        else:
            print("❌ Validation 실패")
        
        if validation.errors:
            print(f"🚨 Errors: {validation.errors}")
        if validation.warnings:
            print(f"⚠️  Warnings: {validation.warnings}")
        if validation.llm_verdict:
            print(f"🤖 LLM Verdict: {validation.llm_verdict}")
        if validation.llm_reason:
            print(f"💭 LLM Reason: {validation.llm_reason}")
        print()
    
    if not validation.is_valid:
        return json.dumps(
            {
                "status": "failed",
                "error": "Plan validation failed",
                "errors": validation.errors or [],
                "warnings": validation.warnings or [],
            },
            ensure_ascii=False,
        )

    # 3) Execution
    if verbose:
        print(f"\n{'='*60}")
        print("⚙️  STEP 3: Plan Execution")
        print("-" * 60)
    
    execution_result = _execute_plan(plan, executor)
    
    if verbose:
        print(f"📊 실행 결과: {execution_result['status']}")
        if execution_result.get("logs"):
            print(f"\n🔍 실행 로그:")
            for i, log in enumerate(execution_result["logs"], 1):
                step = log.get("step", {})
                result = log.get("result", {})
                action = step.get("action", "unknown")
                params = step.get("params", {})
                status = result.get("status", "unknown")
                
                print(f"\n  [{i}] {action}")
                print(f"      Params: {params}")
                print(f"      Status: {status}")
                if result.get("message"):
                    print(f"      Message: {result['message']}")
        print()

    # 실행 실패 여부 확인
    execution_failed = execution_result["status"] == "failed"
    failure_reason = execution_result.get("error", "") if execution_failed else None
    
    if execution_failed and verbose:
        print(f"❌ 실행 실패: {failure_reason}\n")

    # 4) Summarization (성공/실패 관계없이 항상 실행)
    if verbose:
        print(f"\n{'='*60}")
        print("📝 STEP 4: Summarization")
        print("-" * 60)
    
    logs = execution_result.get("logs", [])
    summary = summary_service.build_summary(
        question=question,
        plan=plan,
        execution_logs=logs,
        llm=llm,
        failure_reason=failure_reason,  # 실패 시 이유 전달
    )
    
    if verbose:
        print(f"✅ Summary 생성 완료")
        print(f"📄 Summary:")
        print(f"{summary}")
        print(f"\n{'='*60}")
        if execution_failed:
            print("⚠️  Robot Agent 완료 (실패 후 요약됨)")
        else:
            print("🎉 Robot Agent 완료")
        print(f"{'='*60}\n")

    # 실패 시에도 summary 포함하여 반환
    return json.dumps(
        {
            "status": "failed" if execution_failed else "completed",
            "summary": summary,
            "logs": logs,
            "error": failure_reason if execution_failed else None,
        },
        ensure_ascii=False,
    )


def main(argv: List[str]) -> int:
    root = project_root()
    for env_path in (root / ".env",):
        if env_path.exists():
            load_dotenv(env_path, override=False)

    src_dir = root / "app" / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    if len(argv) < 2:
        print("Usage: python -m app.modules.robot.robot_cli \"<question>\"")
        return 1
    question = argv[1]
    result = run_robot(question)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
