"""Plan execution helper tools."""

from __future__ import annotations

from typing import Dict, Optional

from langchain_core.tools import BaseTool, tool

from app.services.executor_service import ExecutorService


def build_action_tools(executor: ExecutorService) -> Dict[str, BaseTool]:
    """Return LangChain tool objects that wrap executor actions."""

    @tool("navigate")
    def navigate_tool(target: str = "unknown") -> Dict[str, str]:
        """Move the robot to the given target location."""
        return executor.navigate(target)

    @tool("observe_scene")
    def observe_scene_tool(
        target: str = "unknown",
        query: Optional[str] = None,
        question: Optional[str] = None,
    ) -> Dict[str, str]:
        """Observe the scene at the target location to answer the query."""
        detail = query or question
        return executor.observe_scene(target, detail)

    @tool("deliver_object")
    def deliver_object_tool(
        receiver: Optional[str] = None,
        item: Optional[str] = None,
    ) -> Dict[str, str]:
        """Deliver an item to the receiver."""
        return executor.deliver_object(receiver, item)

    @tool("summarize_mission")
    def summarize_mission_tool(summary: Optional[str] = None) -> Dict[str, str]:
        """Summarize the completed mission."""
        return executor.summarize_mission(summary)


    return {
        "navigate": navigate_tool,
        "observe_scene": observe_scene_tool,
        "deliver_object": deliver_object_tool,
        "summarize_mission": summarize_mission_tool,
    }
