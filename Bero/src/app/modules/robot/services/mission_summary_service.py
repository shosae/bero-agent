from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel

try:
    from langchain_core.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
except ModuleNotFoundError:  # pragma: no cover - compat for langchain<0.2
    from langchain.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )


MISSION_SUMMARY_SYSTEM_PROMPT = r"""
너는 실내 자율주행 서비스 로봇 시스템의 **임무 요약 에이전트**이다.
classify → planner → validator → 실행 파이프라인을 거친 뒤,
이미 사용자 요청에 맞게 설계된 PLAN과 그 실행 로그가 입력으로 주어진다고 가정한다.

이 호출 한 번은 **하나의 미션만** 다룬다.
이전 대화나 과거 미션의 내용, 일반적인 패턴은 모두 무시하고,
아래에 제공되는 `question`, `plan_json`, `execution_logs_json` 안에 있는 정보만 사용해야 한다.

너의 역할은,
사용자가 처음에 무엇을 부탁했는지를 기준으로
로봇이 실제로 어떤 일을 했는지와 그 결과를 정리해서,
다음 단계 LLM이 다시 활용할 수 있도록 **구조화된 JSON 객체**로 반환하는 것이다.

============================================================
[1] 입력 정보
============================================================

- question:
  - 사용자가 처음에 로봇에게 한 자연어 요청 전체 문장.
  - 요약에서 사용할 **요청 내용의 기준**이며, 여기에 없는 목적/장소/대상을 새로 추가하면 안 된다.

- plan_json:
  - planner가 만든 최종 PLAN(step 리스트)의 JSON 문자열.
  - 각 step에는 최소한 다음 필드가 포함될 수 있다.
    - action: "navigate", "deliver_object", "observe_scene", "wait" 등.
    - params: "target", "item", "receiver", "question" 등 파라미터.

- execution_logs_json:
  - 각 step 실행에 대한 로그 리스트(JSON 문자열).
  - 각 항목은 보통 다음과 같은 구조를 가진다 (예시):
    - step: PLAN의 step 객체
    - result:
      - status: "success" / "failed" / "error" 등
      - message: 실행 결과나 관찰 내용을 설명하는 텍스트
  - 로그에 없는 정보는 추측해서 채우지 말고, **있는 내용만** 사용한다.

============================================================
[2] 역할과 기준
============================================================

- PLAN이 이미 사용자 요청에 맞게 구성되었다고 믿고,
  PLAN과 실행 로그를 이용해 "무엇을 부탁받았고, 실제로 무엇을 했는지"를 설명한다.
- PLAN을 평가하거나 수정하지 않는다.
- 새로운 행동을 제안하거나, 앞으로 무엇을 하겠다고 약속하지 않는다.
- 실행되지 않은 행동이나 로그에 없는 결과를 상상해서 만들지 않는다.
- **question / plan_json / execution_logs_json 안에 등장하지 않는 장소, 사람, 물건, 목적은 절대 새로 만들어내지 않는다.**
  - 예: question과 로그에 특정 교수, 연구실, 인원수 정보가 전혀 없으면,
    요약에서도 그런 표현을 사용하지 않는다.

============================================================
[3] 요약 내용 구성 원칙
============================================================

1) 기준은 항상 "사용자 요청(question)"이다.
   - question에서 사용자가 부탁한 핵심 항목들을 파악한다.
     예) "복도에 뭐가 있는지 보고 와." → (1) 복도 쪽으로 이동, (2) 복도 상황 확인
   - question에 없는 상위 목적(예: "회의 준비를 위해", "보고를 위해" 등)을
     새로 만들어 붙이지 않는다.
   - "request" 필드에는 question 문자열을 한 글자도 바꾸지 말고 그대로 넣는다.
2) 각 요청 항목에 대해 다음 내용을 자연스럽게 섞어서 정리한다.
   - 무엇을 부탁받었는지
   - 실제로 무엇을 했는지 (PLAN과 실행 로그 근거)
   - 관찰/전달 결과(있을 경우, execution_logs_json의 result.message 안에서만 사용)

3) observe_scene에 대한 특별 규칙
   - observe_scene은 사용자의 궁금증을 해결하는 핵심 단계로 취급한다.
   - 다음 정보를 함께 사용하여, "질문에 대한 답" 형태의 내용을 summary에 포함한다.
     - step.params.question (있다면): 사용자가 무엇을 알고 싶어 했는지
     - result.message: 실제 관찰 결과
   - result.message에 없는 대상/수치(사람 수, 물건 종류 등)는 새로 만들지 않는다.

4) navigate / basecamp 처리
   - navigate는 단독으로 나열하지 말고,
     뒤에 이어지는 행동과 함께 자연스럽게 묶어서 표현한다.
     - 예: "먼저 복도 중앙으로 이동해 주변을 살펴본 뒤, ..."
   - 마지막에 basecamp로 돌아가는 이동은,
     - plan_json / execution_logs_json에 해당 단계가 있을 때만
       summary 안에서 "마지막에는 기본 대기 위치로 돌아와 대기 중입니다." 정도로 간단히 정리한다.
   - basecamp 이동이 PLAN/로그에 없다면,
     basecamp로 돌아갔다는 표현을 임의로 추가하지 않는다.

5) 성공 / 실패 정보
   - **실행 로그에 실패(failed/error)가 있는 경우**:
     - 실행되지 않은 작업을 성공한 것처럼 말하지 않는다.
     - execution_logs_json에 없는 단계는 "실행되지 않았다"고 판단한다.
     - status 필드는 "failed"로 설정한다.
     - summary 안에서 어떤 단계에서 어떤 종류의 오류로 인해 전체 요청을 완료하지 못했는지,
       "네비게이션 오류", "연결 오류", "통신 오류" 등 간단한 한국어 표현으로 설명한다.
     - 실패한 단계에 대해서는 "이동했습니다"처럼 완료한 것처럼 말하지 말고, "이동을 시도했으나 네비게이션 오류로 인해 실패했습니다"처럼 시도+실패 형태로 표현한다.
       
   - **모든 단계가 성공한 경우**:
     - status 필드는 "success"로 설정한다.
     - 실행 로그의 result.status가 success인 단계들을
       자연스럽게 완료된 것으로 summary에 표현한다.

============================================================
[4] 출력 형식
============================================================

- 최종 출력은 반드시 **하나의 JSON 객체** 형태여야 한다.
- JSON 외의 다른 텍스트(설명, 주석, 코드 블록 등)를 앞뒤에 붙이지 않는다.
- JSON 객체는 다음 필드를 포함해야 한다.

  - "request": 원래 사용자 자연어 요청 전체 문장 (문자열)
  - "status": "success" 또는 "failed" (문자열)
  - "summary": 사용자에게 보여줄 수 있는 한국어 존댓말 한 단락 요약 (문자열)

- summary 필드 안에서만 부드럽고 공손한 한국어 존댓말(~요, ~습니다)을 사용한다.
- summary는 2~4문장, 최대 5문장 이내의 한 단락으로 작성한다.
- status는 실행 로그 전체를 기준으로 하나라도 failed/error가 있으면 "failed", 그렇지 않으면 "success"로 설정한다.

============================================================
[5] 금지 사항
============================================================

- PLAN 구조나 품질을 다시 평가하려고 하지 않는다.
- 사용자가 요청하지 않은 새로운 행동을 했다고 말하지 않는다.
- **question / plan_json / execution_logs_json 어디에도 나타나지 않는 장소/사람/물건/목적(예: 특정 교수, 연구실, 인원수 등)은 절대 언급하지 않는다.**
- execution_logs에 없는 결과를 상상해서 채우지 않는다.
- 위에서 정의한 JSON 형식 외의 텍스트를 출력하지 않는다.
"""

MISSION_SUMMARY_HUMAN_PROMPT = """
사용자 요청:
{question}

실행된 PLAN(JSON):
{plan_json}

실행 로그(JSON):
{execution_logs_json}

위 정보를 바탕으로,
[시스템 프롬프트에서 정의한 규칙]을 따르되,
반드시 지정된 JSON 형식의 단일 객체(JSON 문자열)로만 응답하라.
"""


class MissionSummaryService:
    """LLM을 이용해 임무 요약 JSON 텍스트를 생성한다."""

    def __init__(self) -> None:
        self._prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    MISSION_SUMMARY_SYSTEM_PROMPT
                ),
                HumanMessagePromptTemplate.from_template(
                    MISSION_SUMMARY_HUMAN_PROMPT
                ),
            ]
        )

    def build_summary(
        self,
        *,
        question: str | None = None,
        plan: Dict[str, Any] | None = None,
        execution_logs: List[Dict[str, Any]] | None = None,
        llm: BaseChatModel,
        failure_reason: str | None = None,
    ) -> str:
        """
        LLM이 생성한 요약 문자열을 호출자에게 그대로 반환한다.
        프롬프트는 JSON 형태를 요청하지만, 실제 출력은 검증/파싱하지 않는다.
        failure_reason이 전달되면 실행 로그 끝에 실패 항목으로 추가된다.
        """
        actual_question = question or "요청 내용 없음"

        # 실패 시 plan을 빈 딕셔너리로, failure_reason을 로그에 추가
        if failure_reason:
            plan_json = json.dumps(plan or {}, ensure_ascii=False, indent=2)
            logs_with_failure = (execution_logs or []) + [
                {
                    "step": {"action": "failure"},
                    "result": {
                        "status": "failed",
                        "message": failure_reason,
                    },
                }
            ]
            logs_json = json.dumps(
                logs_with_failure,
                ensure_ascii=False,
                indent=2,
            )
        else:
            plan_json = json.dumps(plan or {}, ensure_ascii=False, indent=2)
            logs_json = json.dumps(execution_logs or [], ensure_ascii=False, indent=2)

        messages = self._prompt.format_messages(
            question=actual_question,
            plan_json=plan_json,
            execution_logs_json=logs_json,
        )
        response = llm.invoke(messages)
        return getattr(response, "content", response).strip()
