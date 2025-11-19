"""Planner LLM 호출과 PLAN 생성/검증 로직."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from app.services.validator_service import validate_plan, PlanValidationResult

try:
    from langchain_core.prompts import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )
except ModuleNotFoundError:  # pragma: no cover
    from langchain.prompts import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )


# ---------- PLAN 모델 ----------


class Step(BaseModel):
    action: str
    params: dict = Field(default_factory=dict)


class Plan(BaseModel):
    plan: List[Step] = Field(default_factory=list)


DEFAULT_HINTS = "\n".join(
    [
        "- 아래 항목은 '자연어 장소명 → target 문자열' 매핑 규칙이다. 나열된 모든 장소를 방문하라는 뜻이 아니다.",
        '- 교수님 관련 표현(예: "교수님 방", "교수님 연구실", "연구실" 등)이 있을 때만 navigate / observe_scene에서 target "professor_office"를 사용할 것.',
        '- 복도 관련 표현(예: "복도", "복도 중앙" 등)이 있을 때만 navigate / observe_scene에서 target "corridor_center"를 사용할 것.',
        '- 화장실 앞 관련 표현(예: "화장실 앞" 등)이 있을 때만 navigate / observe_scene에서 target "restroom_front"를 사용할 것.',
        "- 사용자의 요청 문장에 등장하지 않은 장소(target)는 PLAN에 절대 포함하지 말 것.",
        '- 예를 들어, 요청이 "교수님 방을 순찰하고 와"라면 target "professor_office"만 사용하고, "corridor_center"나 "restroom_front"는 절대 사용하지 말 것.',
        "- 로봇이 실제로 복도나 다른 경로를 지나가더라도, 사용자가 요청하지 않은 장소는 PLAN에 action으로 포함하지 말 것.",
        "- '순찰', '보고 와', '확인하고 와' 등 상황 판단 요청일 경우, 사용자의 요청에 등장한 장소만 차례대로 방문해 각각 관찰할 것.",
        '- PLAN의 기본 순서는 [navigate → action → (필요 시 추가 action들) → navigate target "basecamp" → summarize_mission]을 따를 것.',
        '- summarize_mission은 PLAN 전체에서 마지막에 한 번만 사용하며, 중간에 사용하지 말 것.',
        '- summarize_mission 바로 앞에는 항상 한 번만 navigate target "basecamp"를 둘 것.',
    ]
)


# ---------- 프롬프트 ----------

PLAN_SYSTEM_PROMPT = """
너는 이동 로봇을 위한 Task Planner이다.
입력으로 주어진 한국어 요청과 HINTS를 기반으로,
반드시 5가지 액션만을 사용하여 하나의 PLAN JSON을 생성해야 한다.

[1] 절대 규칙
- 출력은 JSON 객체 하나만 포함해야 한다.
- JSON 바깥에는 어떤 문장, 설명, 코드 블록도 포함하면 안 된다.
- JSON 내부에도 주석을 넣으면 안 된다.
- JSON 최상위에는 "plan" 키만 존재해야 한다.
- HINTS에 적힌 지시는 최우선적으로 따라야 한다.
- 사용자가 요청하지 않은 장소·행동은 절대 포함하지 않는다.
- 사용자의 요청 문장에 등장하지 않은 장소(target)는 PLAN에 포함하지 않는다.
- 로봇이 실제로 지나가야 하는 경로(예: 복도)라 하더라도, 사용자가 그 장소를 순찰·방문하라고 명시하지 않았다면 PLAN에 action으로 넣지 않는다.

[2] 사용 가능한 action 목록
- navigate
- deliver_object
- observe_scene
- wait
- summarize_mission
이 5개 외의 action은 절대 생성하지 않는다.

[3] JSON 스키마 (고정)
{
  "plan": [
    {
      "action": "<string>",
      "params": { ... }
    },
    ...
  ]
}

[4] PLAN 구성 규칙
- "plan" 배열은 비어 있으면 안 된다.
- "summarize_mission" 단독 사용은 금지한다.
- PLAN 전체에서 "summarize_mission"은 맨 마지막에 딱 한 번만 사용해야 한다.
- "summarize_mission" 바로 앞에는 반드시 navigate target "basecamp"가 한 번 등장해야 한다.
- 최소 1개 이상의 action(navigate / deliver_object / observe_scene / wait)을 수행해야 한다.

[5] 태스크 기본 흐름
현장 작업 요청이 있을 경우 기본 구조는 다음과 같다.

1) 목적지로 이동 (navigate)
2) 요청된 작업 수행 (observe_scene / deliver_object / wait 중 하나)
3) basecamp로 이동 (navigate, target "basecamp")
4) summarize_mission 실행

- 단일 장소만 언급된 순찰 요청(예: "교수님 방을 순찰하고 와")의 경우,
  해당 장소에 대해서만 [navigate → observe_scene]을 수행한 뒤,
  navigate target "basecamp" → summarize_mission 순서로 마무리한다.
- 여러 장소가 언급된 경우,
  각 장소에 대해 [navigate → action]을 순차적으로 수행한 뒤,
  마지막에 한 번만 navigate target "basecamp" → summarize_mission 을 실행한다.


[5-1] 관찰/질문 요청 처리 규칙
- 사용자가 특정 사실을 궁금해하는 경우(예: "복도에 콜라가 있는지", "복도에 사람이 몇 명인지")에는
  해당 내용을 observe_scene의 params에 question 필드로 넣어라.
  예시:
  {
    "action": "observe_scene",
    "params": {
      "target": "corridor_center",
      "question": "복도에 콜라가 있는지 확인"
    }
  }
- 사람 수처럼 개수를 세는 요청일 경우, question에 그 사실을 그대로 한국어로 적어도 된다.
- observe_scene은 "무엇을 보고 판단해야 하는지"를 question으로 명시하는 용도로 사용한다.

HINTS가 이 기본 흐름과 다르게 지시하면 HINTS를 우선한다.

[5-2] question 필드 작성 규칙
- observe_scene의 params.question 필드는 사용자의 원래 요청 문장에서 발췌한 한국어 구문만 사용할 것.
- 새로운 문장을 창작하지 말고, 사용자의 요청을 그대로 가져오거나, 그 일부만 잘라서 사용할 것.
- 의미 없는 음절 조합, 오타처럼 보이는 표현, 모델이 임의로 만든 단어는 사용하지 말 것.
예시:
요청: "복도에 콜라 있는지 보고 와"
→ question: "복도에 콜라 있는지 보고 와"


[6] 출력 시 유의사항
- 반드시 JSON 문법을 만족해야 한다.
- key의 순서나 들여쓰기는 상관없다.
"""

PLAN_USER_PROMPT = """사용자의 요청은 다음과 같다:
"{question}"

추가 지시(HINTS):
[HINTS_START]
{hints}
[HINTS_END]

위 요청과 HINTS만을 바탕으로 JSON PLAN을 출력하라.
JSON 이외의 어떤 텍스트도 출력하지 마라."""


@dataclass(slots=True)
class PlannerDependencies:
    llm: BaseChatModel


def _build_plan_chain(llm: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                PLAN_SYSTEM_PROMPT,
                template_format="jinja2",
            ),
            HumanMessagePromptTemplate.from_template(
                PLAN_USER_PROMPT,
                template_format="jinja2",
            ),
        ]
    )
    return prompt | llm.with_structured_output(Plan)


def generate_plan(
    question: str,
    llm: BaseChatModel,
    waypoint_docs: List[Document] | None = None,
) -> Tuple[dict, List[Document], PlanValidationResult]:
    """PLANNER LLM을 호출해 PLAN JSON과 검증 결과를 반환한다."""

    docs = waypoint_docs or []
    chain = _build_plan_chain(llm)
    plan_obj = chain.invoke({"question": question, "hints": DEFAULT_HINTS})
    plan_dict = plan_obj.model_dump() if isinstance(plan_obj, Plan) else plan_obj
    validation = validate_plan(plan_dict)
    return plan_dict, docs, validation
