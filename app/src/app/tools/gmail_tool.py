"""Gmail tool placeholder for Master Agent."""

from __future__ import annotations

from langchain_core.tools import tool


@tool("gmail_tool")
def gmail_tool(query: str | None = None) -> str:
    """
    이메일 확인이 필요할 때 사용합니다.
    현재는 더미로 query를 에코합니다.
    """
    if not query:
        return "이메일 조회: query가 필요합니다."
    return f"[Gmail] '{query}'에 대한 이메일을 조회했습니다. (더미 응답)"
