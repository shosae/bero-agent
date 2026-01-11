from __future__ import annotations

import json
from typing import Dict, Any

from langchain_core.messages import HumanMessage


def summarize_result(
    *,
    question: str,
    tool_name: str | None,
    tool_output: str,
    llm,
) -> Dict[str, Any]:
    """Generate a short user-facing answer based on tool output."""

    tool = tool_name or ""

    # 1. 로봇 동작 (이동, 행동) - 특수 처리가 필요한 경우
    if tool == "robot_mission_tool":
        prompt = f"""다음은 사용자 요청과 로봇 미션 실행 결과입니다.
        사용자 요청: {question}
        실행 결과: {tool_output}

        [지침]
        - 실행 결과에 있는 내용만 사용하여, 무엇을 했고 어떤 상태인지 한국어 한두 문장으로 요약하세요.
        - 실패/중단 정보가 있다면 이유를 한 문장으로 덧붙이세요.
        - 목록/JSON/코드블록 없이 자연스러운 문장으로만 답변합니다.
        """

    # 2. 다중 도구 실행 (Multi-step) - 여러 작업이 섞인 경우
    elif tool == "multi_tool":
        prompt = f"""다음은 사용자의 복합적인 요청을 처리하기 위해 여러 도구를 순차적으로 실행한 결과입니다.
        사용자 요청: {question}
        종합 실행 결과:
        {tool_output}

        [지침]
        - 각 단계의 실행 결과를 자연스럽게 이어서 하나의 답변으로 만드세요.
        - 예: "네, 내일 2시에 일정을 등록하고 팀장님께 메일도 발송했습니다."
        - 중간에 실패한 작업이 있다면 그 부분도 명확히 언급해 주세요.
        - ID, 링크, 기술적인 코드는 절대 말하지 말고 구어체로 요약하세요.
        """

    # 3. 단일 일반 도구 (Calendar, Gmail, Tasks 등)
    else:
        prompt = f"""다음은 사용자 요청과 도구({tool}) 실행 결과입니다.
        사용자 요청: {question}
        도구 결과: {tool_output}

        [상황 판단 규칙]
        1. **삭제(Delete/Trash) 요청이었으나 결과가 '검색(Search/List)'만 수행된 경우:**
           - 절대 "삭제했다"고 거짓말하지 마십시오.
           - "해당 항목(일정/메일/할일)을 찾았습니다. 삭제할까요?"라고 되물어보십시오.
           - 검색된 내용(제목/시간)을 언급하여 확인시켜 주십시오.
        
        2. **그 외의 경우 (생성, 전송, 조회 등):**
           - 결과의 핵심(성공 여부, 내용)만 한국어 구어체로 짧게 요약해 주세요.
           - **ID, 링크, 파일명 같은 기술적인 정보는 절대 말하지 마십시오.** (로봇이 스피커로 말하기 부적절함)
           - 예: "네, 내일 오후 3시에 치과 일정을 등록했습니다."
           - 예: "할 일 목록에 '치약 사기'를 추가했습니다."
        """

    resp = llm.invoke([HumanMessage(content=prompt)])
    content = getattr(resp, "content", resp)
    try:
        # LLM이 가끔 JSON 포맷으로 답할 경우를 대비
        payload = json.loads(content)
        return {"answer": payload.get("final_answer") or content}
    except Exception:
        # 일반 텍스트로 답하면 그대로 반환
        return {"answer": content}