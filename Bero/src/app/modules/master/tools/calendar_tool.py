from __future__ import annotations

import datetime
from typing import List, Tuple, Optional
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_core.tools import tool

from app.shared.path_utils import auth_dir

# -------------------------------------------------------------------------
# 1. 설정 및 인증
# -------------------------------------------------------------------------
SCOPES = ["https://www.googleapis.com/auth/calendar"]
KST = ZoneInfo("Asia/Seoul")

def _load_credentials() -> Credentials:
    """Google OAuth 인증 토큰 로드 (없으면 브라우저 띄움)."""
    cred_path = auth_dir() / "credentials.json"
    token_path = auth_dir() / "token.json"
    
    creds: Credentials | None = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(cred_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json(), encoding="utf-8")
        
    return creds

def _get_service():
    """Calendar API 서비스 객체 빌드."""
    creds = _load_credentials()
    return build("calendar", "v3", credentials=creds)


# -------------------------------------------------------------------------
# 2. 날짜/시간 파싱 헬퍼 (AI가 보낸 문자열 -> datetime 변환)
# -------------------------------------------------------------------------
def _parse_datetime(dt_str: str) -> datetime.datetime:
    """문자열을 KST datetime으로 변환."""
    dt_str = dt_str.strip()
    # 포맷 1: "2024-12-05 14:00"
    try:
        dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        return dt.replace(tzinfo=KST)
    except ValueError:
        pass
    
    # 포맷 2: "2024-12-05" (시간 없으면 00:00 처리)
    try:
        dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d")
        return dt.replace(tzinfo=KST)
    except ValueError:
        raise ValueError(f"지원하지 않는 날짜 형식입니다: {dt_str}")

def _parse_time_range(time_range: str) -> Tuple[str, str, bool]:
    """
    'start~end' 문자열을 파싱하여 ISO 포맷 문자열 2개와 종일 여부를 반환.
    Returns: (start_iso, end_iso, is_all_day)
    """
    if not time_range or "~" not in time_range:
        # 범위가 없으면 현재 시간 기준 (Search용 예외 처리)
        now = datetime.datetime.now(tz=KST)
        return now.isoformat(), (now + datetime.timedelta(days=30)).isoformat(), False

    start_str, end_str = time_range.split("~", 1)
    start_dt = _parse_datetime(start_str)
    end_dt = _parse_datetime(end_str)

    # 입력 문자열 길이를 보고 '종일 일정'인지 '시간 일정'인지 추론
    # (예: "2024-12-05" -> 길이 10 -> 종일)
    is_all_day = len(start_str.strip()) <= 10 and len(end_str.strip()) <= 10

    return start_dt.isoformat(), end_dt.isoformat(), is_all_day


# -------------------------------------------------------------------------
# 3. 핵심 기능 (Search, Create, Delete)
# -------------------------------------------------------------------------
def _search_events(service, time_range: str | None, keyword: str | None, limit: int = 5) -> str:
    """일정 검색 (ID 및 종료 시간 포함)."""
    try:
        if time_range and "~" in time_range:
            time_min, time_max, _ = _parse_time_range(time_range)
        else:
            now = datetime.datetime.now(tz=KST)
            time_min = now.isoformat()
            time_max = (now + datetime.timedelta(days=30)).isoformat()

        print(f"[Calendar] 검색: {time_min} ~ {time_max} (키워드: {keyword})")

        events_result = service.events().list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            q=keyword,
            maxResults=limit,
            singleEvents=True,
            orderBy="startTime"
        ).execute()

        events = events_result.get("items", [])
        if not events:
            return "검색된 일정이 없습니다."

        results = []
        for event in events:
            # 1. 시작/종료 시간 원본 가져오기
            start_raw = event["start"].get("dateTime", event["start"].get("date"))
            end_raw = event["end"].get("dateTime", event["end"].get("date"))
            summary = event.get("summary", "(제목 없음)")
            evt_id = event["id"]

            # 2. 보기 좋게 포맷팅 (시간 일정 vs 종일 일정)
            if "T" in start_raw:
                # 시간 일정인 경우 (예: 2025-12-05T14:00:00+09:00)
                # 파이썬 3.7+부터 fromisoformat 사용 가능
                s_dt = datetime.datetime.fromisoformat(start_raw)
                e_dt = datetime.datetime.fromisoformat(end_raw)
                
                # 같은 날이면 종료 시간은 '시:분'만 표시, 다른 날이면 날짜까지 표시
                if s_dt.date() == e_dt.date():
                    time_str = f"[{s_dt.strftime('%Y-%m-%d %H:%M')} ~ {e_dt.strftime('%H:%M')}]"
                else:
                    time_str = f"[{s_dt.strftime('%Y-%m-%d %H:%M')} ~ {e_dt.strftime('%Y-%m-%d %H:%M')}]"
            else:
                # 종일 일정인 경우 (예: 2025-12-05)
                time_str = f"[{start_raw} (종일)]"

            results.append(f"- {time_str} {summary} (ID: {evt_id})")
        
        return "\n".join(results)

    except Exception as e:
        return f"일정 조회 중 오류 발생: {str(e)}"


def _create_event(service, time_range: str, title: str, description: str = "") -> str:
    """일정 생성."""
    try:
        start_iso, end_iso, is_all_day = _parse_time_range(time_range)
        
        body = {
            "summary": title,
            "description": description,
        }
        
        if is_all_day:
            # 종일 일정은 'dateTime' 대신 'date' 사용 (YYYY-MM-DD)
            body["start"] = {"date": start_iso.split("T")[0], "timeZone": "Asia/Seoul"}
            body["end"] = {"date": end_iso.split("T")[0], "timeZone": "Asia/Seoul"}
        else:
            # 시간 일정
            body["start"] = {"dateTime": start_iso, "timeZone": "Asia/Seoul"}
            body["end"] = {"dateTime": end_iso, "timeZone": "Asia/Seoul"}

        event = service.events().insert(calendarId="primary", body=body).execute()
        evt_id = event.get("id", "")
                
        # 날짜 정보를 보기 좋게 변환 (AI가 요약하기 좋게)
        if is_all_day:
            date_info = start_iso.split("T")[0]
        else:
            dt_obj = datetime.datetime.fromisoformat(start_iso)
            date_info = dt_obj.strftime("%m월 %d일 %H시 %M분")

        return f"일정이 정상적으로 등록되었습니다. (제목: {title}, 일시: {date_info}, ID: {evt_id})"

    except Exception as e:
        return f"일정 생성 실패: {str(e)}"


def _delete_event(service, event_id: str) -> str:
    """일정 삭제."""
    try:
        service.events().delete(calendarId="primary", eventId=event_id).execute()
        return "일정이 성공적으로 삭제되었습니다."
    except Exception as e:
        return f"일정 삭제 실패: {str(e)}"


# -------------------------------------------------------------------------
# 4. LangGraph Tool 정의
# -------------------------------------------------------------------------
@tool("calendar_tool")
def calendar_tool(
    action: str,
    time_range: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    keyword: Optional[str] = None,
    event_id: Optional[str] = None,
) -> str:
    """
    Google Calendar 관리 도구.
    
    Args:
        action: "search" (조회), "create" (생성), "delete" (삭제) 중 하나.
        time_range: "YYYY-MM-DD HH:MM~YYYY-MM-DD HH:MM" 형식의 기간 (생성/조회 시 사용).
        title: 일정 제목 (생성 시 필수).
        description: 일정 설명 (생성 시 선택).
        keyword: 검색어 (조회 시 선택).
        event_id: 삭제할 일정의 ID (삭제 시 필수).
    """
    service = _get_service()

    if action == "search":
        # time_range나 keyword 중 하나라도 있으면 검색
        return _search_events(service, time_range, keyword)
    
    elif action == "create":
        if not title or not time_range:
            return "오류: 일정 생성을 위해서는 'title'과 'time_range'가 필수입니다."
        return _create_event(service, time_range, title, description or "")
    
    elif action == "delete":
        if not event_id:
            return "오류: 삭제를 위해서는 'event_id'가 필수입니다. 먼저 조회(search)하여 ID를 확인하세요."
        return _delete_event(service, event_id)
    
    else:
        return f"지원하지 않는 action입니다: {action}"