from __future__ import annotations

import json
import os
import sys
from typing import Dict, Any, List
import logging

from pathlib import Path
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[4]  # RLN/Bero
load_dotenv(_ROOT / ".env", override=False)

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.modules.master.tools import get_master_tools
from app.modules.master.services.decision_service import (
    MasterAgentToolService,
)
from app.modules.master.services.summarize_service import summarize_result
from app.modules.master.services.executor_service import execute_plan
from app.modules.master.services import history_service
from app.config.settings import load_settings
from app.shared.llm_factory import LLMConfig, build_llm
from app.shared.path_utils import project_root


logger = logging.getLogger(__name__)


def _run_tool(tool_name: str, tool_input: Dict[str, Any]) -> str:
    tools = {t.name: t for t in get_master_tools()}
    tool = tools.get(tool_name)
    if tool is None:
        return f"지원하지 않는 도구: {tool_name}"
    try:
        return tool.invoke(tool_input or {})
    except Exception as exc:  # pragma: no cover
        return f"도구 실행 오류: {exc}"


def run_master(
    question: str,
    history: List[BaseMessage] | None = None,
    *,
    llm=None,
) -> str:
    """Main control loop: ask LLM (decide), maybe call a tool, then summarize."""

    if llm is None:
        settings = load_settings()
        llm = build_llm(
            LLMConfig(
                provider=settings.master_llm_provider,
                model=settings.master_llm_model,
                temperature=settings.master_llm_temperature,
                openai_api_key=settings.openai_api_key,
                google_api_key=settings.google_api_key,
                langgraph_api_key=settings.langgraph_api_key,
                langgraph_base_url=settings.langgraph_base_url,
                groq_api_key=settings.groq_api_key,
                ollama_base_url=settings.ollama_base_url,
            )
        )

    recent_entries = history_service.get_recent(limit=5)
    history_text = history_service.format_history(recent_entries)

    logger.info("Question: %s", question)
    if history_text:
        logger.info("Recent history (truncated): %s", history_text[:700])

    messages: List[BaseMessage] = []
    if history_text:
        messages.append(SystemMessage(content=history_text))
    messages.append(HumanMessage(content=question))

    # 1) Decide (LLM tool selection)
    decision_service = MasterAgentToolService()
    decision_obj = decision_service.decide(
        question,
        llm=llm,
        messages=messages,
    )
    steps = decision_obj.steps or []
    final_answer = decision_obj.answer

    logger.info("Decision steps: %s", steps)
    logger.info("Decision final_answer: %s", final_answer)

    if steps:
        result = execute_plan(question, steps, llm=llm)
        logger.info("Execution logs: %s", result.get("execution_logs"))
        logger.info("Summarized answer: %s", result.get("answer"))
        answer = result.get("answer") or json.dumps(result, ensure_ascii=False)
        try:
            history_service.add_history(question, answer, steps)
        except Exception:
            pass
        return answer

    # No steps: return final_answer directly.
    answer = final_answer or "응답을 생성하지 못했습니다."
    try:
        history_service.add_history(question, answer, [])
    except Exception:
        pass
    return answer


def main(argv: List[str]) -> int:
    # Load env files similar to run_graph_cli
    root = project_root()
    for env_path in (root / ".env",):
        if env_path.exists():
            load_dotenv(env_path, override=True)

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    # Ensure src on sys.path when running as script
    src_dir = root / "app" / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    if len(argv) < 2:
        print("Usage: python -m app.modules.master.master_cli \"<question>\"")
        return 1
    question = argv[1]
    answer = run_master(question)
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
