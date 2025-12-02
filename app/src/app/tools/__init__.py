"""Tool registry for the Master Agent."""

from __future__ import annotations

from typing import List

from langchain_core.tools import BaseTool

from .calendar_tool import calendar_tool
from .gmail_tool import gmail_tool
from .rag_tool import rag_tool
from .robot_tool import robot_mission_tool


def get_master_tools() -> List[BaseTool]:
    """Return the list of tools available to the Master Agent."""
    return [calendar_tool, gmail_tool, rag_tool, robot_mission_tool]
