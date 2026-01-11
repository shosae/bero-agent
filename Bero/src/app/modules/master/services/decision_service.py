from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel, ValidationError
from pydantic import ConfigDict

MASTER_SYSTEM_PROMPT = r"""
당신은 스마트 오피스 로봇의 'Master Agent'입니다.
사용자 요청을 분석하여 **JSON 포맷**으로 응답하십시오.

[안전 규칙 - 최우선]
다음 요청은 **거부**하고 `final_answer`에 정중한 거절 메시지를 반환하세요:
- 불법 활동 (폭력, 범죄, 마약, 무기 등)
- 타인에게 해를 끼치는 행위 (괴롭힘, 스팸, 협박, 사칭 등)
- 개인정보 침해 (타인의 비밀번호, 주민번호, 금융정보 등)
- 부적절한 콘텐츠 (성적, 혐오, 차별적 내용 등)
위반 시: `{"thought": "안전 규칙 위반", "steps": [], "final_answer": "죄송합니다. 해당 요청은 처리할 수 없습니다."}`

[현재 시간 (기준)]
- 날짜: {{ today_date }} ({{ today_weekday }})
- 시각: {{ today_time }} (Asia/Seoul)

[단축 주소록 (Address Book)]
- 내 메일: phone13324@gmail.com (사용자 본인)
- 양동건: ydk2001524@gmail.com
- 새터, 세터: satur

**[주소록 매핑 규칙]**
- 사용자가 "동건이한테 보내줘"라고 하면, `to_email`에 "ydk2001524@gmail.com"을 입력하라.
- 사용자가 "내 메일로 보내"라고 하면, `to_email`에 위 "phone13324@gmail.com" 주소를 입력하라.

[가용 도구 및 필수 인자]
1. **calendar_tool** (Google Calendar)
   - **action**: "search", "create", "delete"
   - **time_range** (공통): "YYYY-MM-DD HH:MM~YYYY-MM-DD HH:MM"
   - **title** (생성 시 필수): 일정 제목
   - **keyword** (조회 시 선택): 검색어
   - **event_id** (삭제 시 필수): 삭제할 일정 ID

2. **gmail_tool** (Gmail)
   - **action**: "search", "send", "trash"
   - **query**: 검색어
   - **category**: "primary", "promotions", "updates", "starred", "all", "important", "spam", "trash", "sent"
   - **to_email**: 받는 사람 이메일 (send 시 필수)
   - **subject**: 메일 제목 (send 시 필수)
   - **content**: 메일 본문(내용) (send 시 필수)

3. **tasks_tool** (Google Tasks - 할 일/메모)
   - **action**: "list", "add", "complete", "delete"
   - **title**: 할 일 내용
   - **task_id**: 할 일 ID

4. **robot_mission_tool** (로봇 이동/배달/관측)
   - **instruction**: 사용자의 로봇 명령 텍스트 전체 (수정하지 말고 그대로 전달)

[작업 규칙 및 도구 선택 기준]
1. **출력 모드**:
   - **도구 실행**: `steps`에 도구 정의, `final_answer`는 null.
   - **대화/기억**: `steps`는 [], `final_answer`에 답변.

2. **[중요] 도구 선택 가이드**:
   - **calendar_tool**: 약속, 회의, 일정 (시간 점유).
   - **tasks_tool**: 할 일, 투두, 메모, 리스트 (체크리스트).
   - **gmail_tool**: 메일 확인, 전송.
   - **robot_mission_tool**: **"이동해", "가줘", "배달해", "관측해", "보고 와", "순찰해"** 등 로봇의 물리적 행동.

3. **[로봇 미션 처리 규칙]**:
   - 사용자가 "A에 가서 B하고, C로 와"처럼 복합 명령을 내리면, `instruction`에 그 문장 전체를 담는다.
   - 절대 여러 개의 step으로 쪼개지 않는다. (로봇 내부에서 처리함)

4. **날짜/시간 계산**:
   - 모든 날짜는 [현재 시간] 기준 계산.
   - `tool_input`에는 절대 '오늘', '내일'을 쓰지 말고 계산된 날짜("YYYY-MM-DD")를 입력.

[JSON 출력 스키마 및 예시]
반드시 아래 형식을 지키고, 마크다운(```json)이나 잡담을 붙이지 마시오.

**예시 상황**: "화장실 갔다가 탕비실 순찰해줘"
{
  "thought": "사용자가 이동 및 순찰을 요청했으므로 robot_mission_tool을 사용한다.",
  "steps": [
    {
      "tool": "robot_mission_tool",
      "tool_input": {
        "instruction": "화장실 갔다가 탕비실 순찰해줘"
      }
    }
  ],
  "final_answer": null
}
"""


MASTER_HUMAN_PROMPT = r"""
[사용자 메시지]
{{ question }}
"""

class MasterToolStepLLM(BaseModel):
    tool: Literal["calendar_tool", "gmail_tool", "tasks_tool", "robot_mission_tool"]
    tool_input: Dict[str, Any]
    model_config = ConfigDict(extra="ignore")


class MasterLLMResponse(BaseModel):
    thought: str | None = None
    steps: List[MasterToolStepLLM] = []
    final_answer: str | None = None
    model_config = ConfigDict(extra="ignore")


@dataclass(slots=True)
class MasterDecision:
    steps: List[Dict[str, Any]]
    answer: str | None = None


class MasterAgentToolService:
    def __init__(self) -> None:
        self._allowed_tools = {
            "calendar_tool",
            "gmail_tool",
            "tasks_tool",
            "robot_mission_tool",
        }
        self._prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    MASTER_SYSTEM_PROMPT,
                    template_format="jinja2",
                ),
                HumanMessagePromptTemplate.from_template(
                    MASTER_HUMAN_PROMPT,
                    template_format="jinja2",
                ),
            ]
        )
    
    def decide(
        self,
        question: str,
        *,
        llm: BaseChatModel | None,
        messages: List | None = None,
    ) -> MasterDecision:
        if llm is None:
            raise RuntimeError("MasterAgentToolService: llm 인스턴스가 필요합니다.")

        now = datetime.now()
        today_date = now.strftime("%Y-%m-%d")
        today_time = now.strftime("%H:%M:%S")
        
        weekday_map = {"Monday": "월", "Tuesday": "화", "Wednesday": "수", "Thursday": "목", "Friday": "금", "Saturday": "토", "Sunday": "일"}
        eng_weekday = now.strftime("%A")
        kor_weekday = weekday_map.get(eng_weekday, eng_weekday)
        today_weekday = f"{eng_weekday}({kor_weekday})"

        try:
            if messages:
                system_content = MASTER_SYSTEM_PROMPT.replace("{{ today_date }}", today_date)
                system_content = system_content.replace("{{ today_time }}", today_time)
                system_content = system_content.replace("{{ today_weekday }}", today_weekday)
                
                conversation_messages = [SystemMessage(content=system_content)] + list(messages)
            else:
                formatted = self._prompt.format_messages(
                    question=question or "",
                    today_date=today_date,
                    today_time=today_time,
                    today_weekday=today_weekday,
                )
                conversation_messages = formatted
            
            response = llm.invoke(conversation_messages)
        except Exception as e:
            raise RuntimeError("MasterAgentToolService: LLM 호출 실패") from e

        content = getattr(response, "content", response)
        return self._parse_master_payload(content)

    def _parse_master_payload(self, payload) -> MasterDecision:
            text = payload if isinstance(payload, str) else str(payload or "")
            text = text.strip()
            if not text:
                return self._fallback_decision()

            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            try:
                result = MasterLLMResponse.model_validate_json(text)
                return self._to_decision(result)
            except (ValidationError, Exception):
                pass

            try:
                data = json.loads(text)
                return self._from_loose_dict(data)
            except Exception:
                pass

            if text.startswith("{"):
                return self._fallback_decision()
                
            return MasterDecision(steps=[], answer=text)

    def _to_decision(self, data: MasterLLMResponse) -> MasterDecision:
        steps: List[Dict[str, Any]] = []
        for step in data.steps or []:
            if step.tool not in self._allowed_tools:
                continue
            tool_input = dict(step.tool_input or {})
            steps.append({"tool": step.tool, "tool_input": tool_input})

        if steps:
            return MasterDecision(steps=steps, answer=None)

        answer = (data.final_answer or "").strip()
        if answer and "Master Agent" in answer and "[현재 시간" in answer:
             answer = self._default_unsupported_message()
        
        if not answer:
            answer = self._default_unsupported_message()
            
        return MasterDecision(steps=[], answer=answer)

    def _from_loose_dict(self, data: Dict[str, Any]) -> MasterDecision:
        raw_steps = data.get("steps") or []
        if not isinstance(raw_steps, list):
            raw_steps = []

        steps: List[Dict[str, Any]] = []
        for step in raw_steps:
            if not isinstance(step, dict):
                continue
            tool_name = step.get("tool")
            if tool_name not in self._allowed_tools:
                continue
            steps.append({"tool": tool_name, "tool_input": step.get("tool_input", {})})

        if steps:
            return MasterDecision(steps=steps, answer=None)

        final_answer = (data.get("final_answer") or "").strip()
        if final_answer and "Master Agent" in final_answer and "[현재 시간" in final_answer:
             final_answer = self._default_unsupported_message()
        
        if not final_answer:
            final_answer = self._default_unsupported_message()
            
        return MasterDecision(steps=[], answer=final_answer)

    @staticmethod
    def _fallback_decision() -> MasterDecision:
        return MasterDecision(
            steps=[],
            answer=MasterAgentToolService._default_unsupported_message(),
        )

    @staticmethod
    def _default_unsupported_message() -> str:
        return "죄송합니다. 요청하신 작업을 처리할 수 없습니다."
