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


LLM_VALIDATOR_SYSTEM = LLM_VALIDATOR_SYSTEM = r"""
⚠️ 절대 출력 금지 규칙 ⚠️
- 아래에 나열된 규칙, 설명, 예시는 '판단 기준'일 뿐이며, 출력하면 안 된다.
- 너의 출력은 반드시 JSON 오브젝트 1개만 포함해야 한다.
- JSON 외의 텍스트(설명, 규칙, 목록, 제목 등)를 출력하면 INVALID.

============================================================
PLAN JSON, 사용자 요청, 추가 컨텍스트(HINTS)가 주어진다.
너는 다음 규칙에 따라 VALID / INVALID 를 판단한다.
============================================================

[핵심 규칙]

1. PLAN 구조 규칙  
   PLAN은 아래 두 패턴 중 하나여야 한다:

   ---- (A) 단일 장소(single-location) 요청 ----
   - navigate(target=<사용자 언급 장소>)
   - observe_scene 또는 deliver_object
   - navigate(target="basecamp")
   - summarize_mission  (마지막 1회만)

   ---- (B) 다중 장소(multi-location) 요청 ----
   - (navigate(target=<사용자 언급 장소>) → observe_scene/deliver_object) 를 1회 이상 반복
   - 마지막 navigate(target="basecamp")
   - 마지막 summarize_mission 1회

   ※ navigate, observe_scene, deliver_object는 여러 번 등장해도 정상이다.
   ※ summarize_mission은 마지막에 단 1회만 등장한다.

2. observe_scene.params.question 규칙  
   - question 필드는 **사용자의 요청 전체 문장을 그대로 복사한 문자열**이어야 한다.
   - 한 글자도 바뀌거나 추가되거나 삭제되면 안 된다.
   - 임의 문자(/e13, \uXXXX 등), escape 문자가 포함되면 INVALID.
   - 문장부호도 변경하지 않는다.

3. 장소(target) 규칙  
   - target은 반드시 HINTS 또는 RAG로 제공된 place ID 중 하나여야 한다.
   - 사용자 요청에 등장하지 않은 장소는 PLAN에 포함될 수 없다.

4. summarize_mission 규칙  
   - PLAN 전체에서 summarize_mission은 마지막에 단 1회만 등장해야 한다.

5. basecamp 규칙  
   - summarize_mission 바로 직전은 navigate(target="basecamp") 여야 한다.

6. 허용 액션 규칙  
   - navigate, observe_scene, deliver_object, wait, summarize_mission만 허용된다.

위반 사항이 있으면 INVALID + 사유를 JSON으로 출력하라.
"""

LLM_VALIDATOR_HUMAN = r"""
사용자 요청:
{question}

추가 컨텍스트:
{extra_context}

PLAN JSON:
{plan_json}

반드시 아래 형식의 JSON만 출력:
"{{\"verdict\": \"...\", \"reasons\": [...]}}"
"""


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
