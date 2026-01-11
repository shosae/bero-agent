from __future__ import annotations

from typing import Dict, Optional

from langchain_core.tools import BaseTool, tool

from app.modules.robot.services.executor_service import ExecutorService


def build_action_tools(executor: ExecutorService) -> Dict[str, BaseTool]:
    """Return LangChain tool objects that wrap executor actions."""

    @tool("navigate")
    def navigate_tool(target: str = "unknown") -> Dict[str, str]:
        """Move the robot to the given target location."""
        return executor.navigate(target)

    @tool("observe_scene")
    def observe_scene_tool(
        target: str = "unknown",
        question: Optional[str] = None,
    ) -> Dict[str, str]:
        """Observe the scene at the target location to answer the query."""
        return executor.observe_scene(target, question)

    @tool("deliver_object")
    def deliver_object_tool(
        receiver: Optional[str] = None,
    ) -> Dict[str, str]:
        """Deliver an item to the receiver."""
        return executor.deliver_object(receiver)

    @tool("wait")
    def wait_tool() -> Dict[str, str]:
        """Stay put until further notice."""
        return executor.wait()

    return {
        "navigate": navigate_tool,
        "observe_scene": observe_scene_tool,
        "deliver_object": deliver_object_tool,
        "wait": wait_tool,
    }
