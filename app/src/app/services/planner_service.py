"""Planner LLM 호출과 PLAN 생성/검증 로직."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

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


DEFAULT_HINTS = r"""
[장소 매핑 규칙]
- 교수님 관련 표현(예: "교수님 방", "교수님 연구실", "연구실" 등)이 포함될 때만 target "professor_office" 사용.
- 복도 관련 표현(예: "복도", "복도 중앙" 등)이 포함될 때만 target "corridor_center" 사용.
- 화장실 앞 관련 표현(예: "화장실 앞")이 포함될 때만 target "restroom_front" 사용.
- 사용자가 언급하지 않은 장소(target)는 PLAN에 절대 포함하지 않는다.
- 여러 장소가 등장하면 사용자가 말한 **순서대로** 방문해야 한다.

[행동(action) 결정 규칙]
- 사용자가 각 장소에서 무엇을 하라고 했는지는 다음과 같은 패턴으로 결정한다:
  * "보고 와", "~확인해", "~봐줘", "~인지 확인" → observe_scene
  * "전달해줘", "갖다줘", "드리고 와", "~건네" → deliver_object
  * "기다려", "~나오면", "~될 때까지 기다려" → wait
- 명령이 없으면 임의로 추가하거나 추론하지 않는다.

[question/params 규칙 (모든 action 공통 적용)]
- 각 action의 params.question(또는 equivalent 필드)은
  **해당 장소와 직접 연결된 '사용자 원문 구절'을 그대로 복사**해야 한다.
- 원문에 없는 단어를 추가/삭제/변경하면 INVALID.
- escape 문자(\uXXXX, /e13 등) 출력 금지.
- 여러 장소가 있으면, 각 장소에 대해 원문에서 그 장소와 연결된 구절을 각각 발췌한다.

[PLAN 구조 규칙]
- 각 장소마다:
    navigate(target=<장소>)
    action(해당 장소의 행동)
- 모든 장소 처리 후:
    navigate(target="basecamp")
    summarize_mission
"""


# ---------- 프롬프트 ----------

PLAN_SYSTEM_PROMPT = r"""
너는 이동 로봇을 위한 고정식 Task Planner이다.
너의 출력은 반드시 JSON 하나만 포함해야 한다. JSON 밖에 다른 텍스트는 절대 금지한다.

============================================================
[1] 절대 규칙
============================================================
- JSON 외 텍스트 출력 금지.
- JSON 최상위 키는 "plan" 하나만 존재.
- action은 다음 4개만 사용: navigate, observe_scene, deliver_object, summarize_mission.
- target은 반드시 HINTS 또는 RAG로 주어진 place ID 중 하나여야 하며,
  사용자 요청에 등장한 장소만 PLAN에 포함해야 한다.
- summarize_mission은 PLAN 마지막에 1회만 등장.
- summarize_mission 바로 앞에는 navigate(target="basecamp")가 반드시 존재해야 한다.

============================================================
[2] PLAN 기본 구조
============================================================

(단일 장소 요청일 때)
1) navigate(target=<장소>)
2) action(observe_scene 또는 deliver_object)
3) navigate(target="basecamp")
4) summarize_mission

(다중 장소 요청일 때)
각 장소마다:
    - navigate(target=<장소>)
    - action(observe_scene 또는 deliver_object)
모든 장소 처리 후:
    - navigate(target="basecamp")
    - summarize_mission

============================================================
[3] observe_scene.question 규칙 (중요)
============================================================
- question 값은 "사용자 원문에서 해당 장소와 관련된 구절을 그대로 발췌" 해야 한다.
- 새로운 문장/단어를 생성하면 안 된다.
- 단어 추가/삭제/순서 변경 금지.
- "\uXXXX", "/e13" 같은 escape 이상문자 출력 금지.
- 장소마다 서로 다른 질문을 써도 된다. (각 장소 관련 substring을 사용)

예)
사용자 요청: "교수님 방과 복도 좀 보고 와줘"
→ professor_office에서는 "교수님 방" 또는 "교수님 방 보고 와줘" 등 원문 substring
→ corridor_center에서는 "복도" 또는 "복도 보고 와줘" 등 원문 substring

============================================================
[4] 출력 형식
============================================================
{
  "plan": [
    {
      "action": "...",
      "params": { ... }
    }
  ]
}
"""

PLAN_USER_PROMPT = """
사용자의 요청:
{{question}}

추가 지시(HINTS):
{{hints}}

위 정보를 기반으로 JSON PLAN만 출력하라.
JSON 외의 텍스트는 절대 출력하지 않는다.
"""


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
) -> Tuple[dict, List[Document]]:
    """PLANNER LLM을 호출해 PLAN JSON을 생성한다."""

    docs = waypoint_docs or []
    chain = _build_plan_chain(llm)
    plan_obj = chain.invoke({"question": question, "hints": DEFAULT_HINTS})
    plan_dict = plan_obj.model_dump() if isinstance(plan_obj, Plan) else plan_obj
    return plan_dict, docs
