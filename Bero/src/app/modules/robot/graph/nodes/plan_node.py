from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from langchain_core.documents import Document

from app.modules.robot.services.planner_service import generate_plan


@dataclass
class PlanNodeResult:
    plan: Dict[str, Any]
    context_docs: List[Document]


def run_plan_node(
    question: str,
    llm,
    retry_feedback: str | None = None,
) -> PlanNodeResult:
    """Planner node가 호출할 진입점."""

    if retry_feedback:
        augmented_question = f"{question}\n\n[검증 피드백]\n{retry_feedback}"
    else:
        augmented_question = question

    plan_dict= generate_plan(augmented_question, llm)
    return PlanNodeResult(plan=plan_dict)
