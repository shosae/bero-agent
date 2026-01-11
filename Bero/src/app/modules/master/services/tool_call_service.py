from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool


def _extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
    """Best-effort extraction of tool call name/args from an LLM response."""

    calls = []
    raw_calls = getattr(response, "tool_calls", None) or []
    for call in raw_calls:
        name = getattr(call, "name", None) or getattr(call, "id", None)
        args = getattr(call, "args", None) or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"raw": args}
        calls.append({"tool": name, "args": args})

    # Fallback: some providers (e.g., Ollama without native tool calling)
    # may return a JSON blob in `content` instead of structured tool_calls.
    if not calls:
        content = getattr(response, "content", None)
        if isinstance(content, str):
            try:
                data = json.loads(content)
            except Exception:
                data = None
        elif isinstance(content, dict):
            data = content
        else:
            data = None

        if isinstance(data, dict):
            name = data.get("name") or data.get("tool")
            args = data.get("arguments") or data.get("args") or {}
            if name:
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                calls.append({"tool": name, "args": args})

    return calls


# tool_call_service.py 내부 run_tool_calling 함수

def run_tool_calling(question: str, *, tools: List[BaseTool], llm) -> Dict[str, Any]:
    
    # [핵심] "상황에 따른 판단"을 지시하는 시스템 프롬프트
    system_prompt = (
        "You are a smart assistant. "
        "You have access to the following tools. "
        "1. If the user asks to perform a specific action (e.g., search, email, calculation), "
        "you MUST call the appropriate tool with valid JSON arguments. "
        "2. If the user greets you or asks a general question that doesn't require a tool, "
        "just reply naturally in Korean. "
        "3. Do NOT output JSON if you are just chatting."
    )

    llm_with_tools = llm.bind_tools(tools)
    
    response = llm_with_tools.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    return {
        "tool_calls": _extract_tool_calls(response),
        "raw_response": response,
    }