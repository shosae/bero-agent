"""Validator node helpers."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from app.services.validator_service import validate_plan, PlanValidationResult


def run_validator(
    plan_dict: Any,
    *,
    question: str,
    llm: BaseChatModel,
    extra_context: str | None = None,
) -> PlanValidationResult:
    """PLAN JSON 구조 + LLM 검증."""

    return validate_plan(
        plan_dict,
        question=question,
        llm=llm,
        extra_context=extra_context,
    )
