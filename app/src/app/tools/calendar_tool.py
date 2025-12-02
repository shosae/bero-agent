"""Calendar tool placeholder for Master Agent."""

from __future__ import annotations

from langchain_core.tools import tool


@tool("calendar_tool")
def calendar_tool(time_range: str | None = None) -> str:
    """
    일정 확인이 필요할 때 사용합니다.
    현재는 더미로 time_range를 에코합니다.
    """
    if not time_range:
        return "일정 조회: time_range가 필요합니다."
    return f"[Calendar] {time_range} 기간의 일정을 조회했습니다. (더미 응답)"
