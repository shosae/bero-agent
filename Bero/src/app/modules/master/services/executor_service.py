from __future__ import annotations

from typing import Any, Dict, List

from app.modules.master.tools import get_master_tools
from app.modules.master.services.summarize_service import summarize_result


def execute_plan(
    question: str,
    steps: List[Dict[str, Any]],
    *,
    llm,
) -> Dict[str, Any]:
    """
    Run tools sequentially based on a pre-generated plan (no LLM tool-calling).

    - steps: [{"tool": str, "tool_input": dict}, ...]
    - unknown tools are skipped with a log entry.
    - If a preceding step yields "no result" for downstream use, remaining steps may be skipped.
    """

    logs: List[Dict[str, Any]] = []
    tools = {t.name: t for t in get_master_tools()}

    def _should_skip_rest(tool_name: str, output: str) -> bool:
        """Heuristic: if preceding tool says there's nothing to do, skip remaining steps."""
        text = (output or "").lower()
        if tool_name == "gmail_tool":
            return ("없" in text) or ("no email" in text) or ("empty" in text)
        return False

    for step in steps:
        tool_name = step.get("tool")
        tool_input = step.get("tool_input") or {}
        tool = tools.get(tool_name)
        if tool is None:
            logs.append(
                {
                    "tool": tool_name or "unknown_tool",
                    "input": tool_input,
                    "output": f"지원하지 않는 도구: {tool_name}",
                }
            )
            continue

        try:
            output = tool.invoke(tool_input)
        except Exception as exc:  # pragma: no cover
            output = f"도구 실행 오류: {exc}"

        logs.append({"tool": tool_name, "input": tool_input, "output": output})

        if _should_skip_rest(tool_name or "", str(output)):
            break

    combined_output = "\n".join(
        f"[{entry.get('tool')}] {entry.get('output')}" for entry in logs
    ) or "도구 실행 결과가 없습니다."

    summary = summarize_result(
        question=question,
        tool_name="multi_tool",
        tool_output=combined_output,
        llm=llm,
    )

    return {
        "execution_logs": logs,
        "answer": summary.get("answer") or combined_output,
    }
