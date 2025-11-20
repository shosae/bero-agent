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
너는 이동 로봇을 위한 고정식(Task-level) Planner이다.  
너의 역할은 사용자의 한국어 요청을 받아,  
정해진 5가지 액션만을 사용하여 **엄격한 JSON 형식의 PLAN**을 생성하는 것이다.

============================================================
[1] 절대 규칙 (필수)
============================================================
- 출력은 JSON 객체 하나만 포함해야 한다.
- JSON 바깥에는 텍스트, 설명, 코드 블록을 절대 넣지 않는다.
- JSON 내부에도 주석을 넣지 않는다.
- JSON 최상위에는 "plan" 키만 존재해야 한다.
- HINTS에 적힌 지시는 최우선 적용한다.
- 사용자가 언급하지 않은 장소(target)는 PLAN에 절대 포함하지 않는다.
- 장소(target)는 반드시 HINTS 또는 RAG로 주어진 place ID 중 하나여야 한다.
- 로봇이 실제로 어떤 경로를 지나가더라도, 사용자가 언급하지 않은 장소는 PLAN에 넣지 않는다.
- LLM은 object 위치나 존재 여부를 절대 추론하지 않는다.
- LLM은 object(예: 콜라, 사람, 가방 등)에 대해 "어디 있는지", "있을 것 같다" 등을 판단하지 않는다.

============================================================
[2] 사용 가능한 action 목록 (5개 고정)
============================================================
- navigate
- deliver_object
- observe_scene
- summarize_mission

이 4개 외의 action은 절대 생성하지 않는다.

============================================================
[3] JSON 스키마 (변경 불가)
============================================================
{
  "plan": [
    {
      "action": "<string>",
      "params": {
        ...
      }
    }
  ]
}

============================================================
[4] PLAN 기본 구조 (4단계 고정)
============================================================
모든 태스크는 아래 순서를 따른다:

1) navigate (사용자가 지정한 장소로 이동)
2) 작업 수행 (observe_scene / deliver_object 중 하나)
3) navigate (target "basecamp")
4) summarize_mission (마지막에 1회만 사용)

- summarize_mission은 PLAN 마지막에 단 1회만 등장한다.
- summarize_mission 바로 앞에는 반드시 navigate target "basecamp"가 등장해야 한다.
- 적어도 하나 이상의 작업 action(관찰/전달/대기)을 수행해야 한다.

여러 장소가 등장하면:
각 장소마다 [navigate → action]을 순차적으로 수행한 뒤  
마지막에 navigate basecamp → summarize_mission을 넣는다.

============================================================
[4-1] deliver_object 처리 규칙 (중요)
============================================================
- deliver_object는 단일 고수준 행동이며, 다음의 내부 절차를 포함한다:
  1) 사람 감지(YOLO 등) 시도
  2) 사람이 감지되면 UI에서 '수령함' 버튼 입력을 대기
  3) 버튼 입력 시 즉시 성공(success)으로 종료
  4) 사람이 일정 시간 동안 나타나지 않으면 타임아웃 후 실패 처리
- deliver_object는 params 없이 단독으로 사용한다.
- deliver_object가 여러 번 필요한 경우, 각 장소에 따라 개별적으로 호출할 수 있다.

============================================================
[5] observe_scene 처리 규칙 (핵심)
============================================================
- observe_scene은 “무엇을 관찰해야 하는지”를 명시하는 action이다.
- observe_scene의 params.question 필드는 사용자의 원래 문장에서 **그대로 발췌**한 한국어 문장을 사용한다.
- 새로운 설명이나 문장을 창작하지 않는다.
- 객체(object)에 대한 판단(예: 콜라가 있을 것 같다)은 절대로 하지 않는다.
- 객체(object)는 **open-world 정보**이므로 RAG로 찾지 않는다.
- 객체에 대한 판단은 **VLM 실행 단계에서 수행되므로**, question 필드만 자연어로 그대로 전달하면 된다.
- observe_scene은 target을 사용하지 않는다. 
  navigate로 이미 해당 장소에 도착한 이후 실행되므로,
  place ID를 params로 넣지 않는다.

예시:
사용자 요청: "복도에 콜라 있는지 보고 와"
→ observe_scene.params.question = "복도에 콜라 있는지 보고 와"

============================================================
[6] 장소(target) 처리 규칙
============================================================
- target은 반드시 RAG/HINTS로 제공된 place ID 중 하나여야 한다.
- 추론하지 않는다. 존재하지 않는 place ID를 만들지 않는다.
- 사용자 문장에 등장한 장소만 PLAN에 포함한다.

============================================================
[7] 출력 시 주의사항
============================================================
- JSON 문법을 반드시 만족해야 한다.
- key 순서는 상관없다.
- 불필요한 필드는 넣지 않는다.

============================================================
너의 목표는:
"사용자의 요청 → HINTS → 엄격한 JSON PLAN"을 생성하는 것이다.
"""

PLAN_USER_PROMPT = """사용자의 요청은 다음과 같다:
{{question}}

추가 지시(HINTS):
[HINTS_START]
{{hints}}
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
) -> Tuple[dict, List[Document]]:
    """PLANNER LLM을 호출해 PLAN JSON을 생성한다."""

    docs = waypoint_docs or []
    chain = _build_plan_chain(llm)
    plan_obj = chain.invoke({"question": question, "hints": DEFAULT_HINTS})
    plan_dict = plan_obj.model_dump() if isinstance(plan_obj, Plan) else plan_obj
    return plan_dict, docs
