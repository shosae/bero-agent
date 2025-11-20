"""LLM-based PLAN validator."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

try:
    from langchain_core.prompts import ChatPromptTemplate
except ModuleNotFoundError:  # pragma: no cover
    from langchain.prompts import ChatPromptTemplate

from app.services.validator_rule_service import PlanValidationResult


LLM_VALIDATOR_SYSTEM = """
PLAN JSON, 사용자 요청, 추가 컨텍스트(HINTS)가 주어진다.
너는 다음 규칙에 따라 VALID / INVALID 를 판단한다.

[핵심 규칙]
1. PLAN은 반드시 다음 구조를 가져야 한다:
   - (1) navigate(target=<사용자 언급 장소>)
   - (2) observe_scene / deliver_object
   - (3) navigate(target="basecamp")
   - (4) summarize_mission  ← 마지막 1회만 등장

2. observe_scene.params.question 은 사용자 요청 문장의 일부 또는 전체를 그대로 사용해야 한다.
3. 장소(target)는 HINTS 또는 RAG가 제공한 place ID 중 하나여야 하며, 사용자가 언급하지 않은 장소는 등장할 수 없다.
4. summarize_mission은 마지막에 1회만 등장할 수 있다.
5. summarize_mission 바로 전 단계는 navigate(target="basecamp") 여야 한다.
6. PLAN 내부에 허용되지 않은 action은 없어야 한다.

위반 사항이 있으면 INVALID + 사유를 JSON 형태로 출력하라.
"""


LLM_VALIDATOR_HUMAN = """
사용자 요청:
{question}

추가 컨텍스트/힌트:
{extra_context}

PLAN:
{plan_json}

출력 형식(반드시 JSON 한 개만):
{{"verdict": "VALID" 또는 "INVALID", "reasons": ["위반 설명 ..."]}}
- VALID이면 reasons는 빈 배열로 둔다.
- INVALID이면 어떤 규칙을 어겼는지 한국어로 구체적으로 적어라."""


class LLMValidatorService:
    """Light-weight wrapper that asks the LLM to double-check a PLAN."""

    def __init__(self) -> None:
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", LLM_VALIDATOR_SYSTEM),
                ("human", LLM_VALIDATOR_HUMAN),
            ]
        )

    def validate(
        self,
        plan: Any,
        *,
        question: str | None,
        llm: BaseChatModel,
        extra_context: str | None = None,
    ) -> PlanValidationResult:
        result = PlanValidationResult()
        if llm is None:
            return result

        plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
        messages = self._prompt.format_messages(
            question=question or "",
            plan_json=plan_json,
            extra_context=extra_context or "(추가 힌트 없음)",
        )
        try:
            response = llm.invoke(messages)
            content = getattr(response, "content", response)
            payload = self._extract_json(content)
        except Exception as exc:  # noqa: BLE001
            result.add_warning(f"LLM validator 호출 실패: {exc}")
            return result

        verdict = (payload.get("verdict") or "").strip().upper()
        reasons = payload.get("reasons") or []
        if verdict != "VALID":
            if not reasons:
                result.add_error("LLM validator가 INVALID를 보고했지만 사유가 비었습니다.")
            else:
                for reason in reasons:
                    result.add_error(reason)
        return result

    @staticmethod
    def _extract_json(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError("LLM validator 출력에서 JSON을 찾을 수 없습니다.") from None
            return json.loads(match.group(0))
