"""Master Agent: tool-calling loop using ChatOllama with JSON-only outputs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_ollama import ChatOllama

from app.tools import get_master_tools


SYSTEM_PROMPT = r"""
당신은 스마트 오피스 로봇의 두뇌인 'Master Agent'입니다.
사용자의 요청을 분석하여 **대화**를 할지 **도구(Tool)**를 사용할지 결정하십시오.

[사용 가능한 도구 목록]
1. calendar_tool: 일정 확인이 필요할 때 사용 (인자: time_range)
2. gmail_tool: 이메일 확인이 필요할 때 사용 (인자: query)
3. rag_tool: 사내 규정, 매뉴얼, 박물관 정보 등 지식 검색이 필요할 때 사용 (인자: query)
4. robot_mission_tool: 물리적 이동, 배달, 확인, 안내 등 로봇 행동이 필요할 때 사용 (인자: instruction)

[답변 규칙]
반드시 아래 JSON 형식으로만 응답해야 합니다. 사족이나 마크다운을 포함하지 마십시오.

{
  "thought": "사용자의 의도와 도구 선택 이유",
  "tool": "도구이름" 또는 null,
  "tool_input": { "arg_name": "value" } 또는 null,
  "final_answer": "사용자에게 할 말" 또는 null
}

[시나리오별 작성법]
- 도구 사용 시: "tool"에 이름 명시, "final_answer"는 null.
- 단순 대화/결과 보고 시: "tool"은 null, "final_answer"에 답변 작성.
"""


def _build_llm() -> ChatOllama:
    return ChatOllama(
        model=os.getenv("LLM_MODEL", "llama-3-korean-bllossom"),
        base_url=os.getenv("OLLAMA_BASE_URL"),
        temperature=0,
        format="json",
    )


def _parse_json(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except Exception:
        return {}


def _run_tool(tool_name: str, tool_input: Dict[str, Any]) -> str:
    tools = {t.name: t for t in get_master_tools()}
    tool = tools.get(tool_name)
    if tool is None:
        return f"지원하지 않는 도구: {tool_name}"
    try:
        return tool.invoke(tool_input or {})
    except Exception as exc:  # pragma: no cover
        return f"도구 실행 오류: {exc}"


def run_master(question: str, history: List[BaseMessage] | None = None) -> str:
    """Main control loop: ask LLM, maybe call a tool, then summarize."""

    llm = _build_llm()
    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    if history:
        messages.extend(history)
    messages.append(HumanMessage(content=question))

    first = llm.invoke(messages)
    payload = _parse_json(getattr(first, "content", ""))
    tool_name = payload.get("tool")
    tool_input = payload.get("tool_input") or {}
    final_answer = payload.get("final_answer")

    if tool_name:
        # Run tool and re-prompt for final answer.
        tool_output = _run_tool(tool_name, tool_input)
        messages.append(SystemMessage(content=f"[도구 실행 결과]\n{tool_output}"))
        second = llm.invoke(
            messages
            + [
                HumanMessage(
                    content="도구 결과를 반영해 final_answer만 포함된 JSON을 다시 작성하세요. tool은 null로 두세요."
                )
            ]
        )
        payload = _parse_json(getattr(second, "content", ""))
        return payload.get("final_answer") or tool_output

    # No tool: return final_answer directly.
    return final_answer or json.dumps(payload, ensure_ascii=False)


def main(argv: List[str]) -> int:
    # Load env files similar to run_graph_cli
    root = Path(__file__).resolve().parents[3]
    for env_path in (root / ".env", root / "app" / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)

    # Ensure src on sys.path when running as script
    src_dir = root / "app" / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    if len(argv) < 2:
        print("Usage: python -m app.master.agent \"<question>\"")
        return 1
    question = argv[1]
    answer = run_master(question)
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
