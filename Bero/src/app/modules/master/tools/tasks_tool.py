from __future__ import annotations

from typing import Optional
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_core.tools import tool

from app.shared.path_utils import auth_dir

# -------------------------------------------------------------------------
# 1. 설정 및 인증
# -------------------------------------------------------------------------
# 주의: 이 스코프를 사용하려면 token.json을 삭제하고 다시 로그인해야 합니다.
SCOPES = ["https://www.googleapis.com/auth/tasks"]

def _load_credentials() -> Credentials:
    """Google OAuth 인증 토큰 로드."""
    cred_path = auth_dir() / "credentials.json"
    token_path = auth_dir() / "token.json"
    
    creds: Credentials | None = None
    
    # 기존 토큰 파일이 있으면 로드 시도
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception:
            # 스코프가 변경되었거나 파일이 손상된 경우 무시하고 재인증 진행
            creds = None

    # 토큰이 없거나 유효하지 않으면 새로고침 또는 재로그인
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None # 리프레시 실패 시 재로그인

        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(str(cred_path), SCOPES)
            creds = flow.run_local_server(port=0)
        
        # 새 토큰 저장
        token_path.write_text(creds.to_json(), encoding="utf-8")
        
    return creds

def _get_service():
    """Tasks API 서비스 객체 빌드."""
    creds = _load_credentials()
    return build("tasks", "v1", credentials=creds)


# -------------------------------------------------------------------------
# 2. 핵심 기능 (List, Add, Complete, Delete)
# -------------------------------------------------------------------------

def _list_tasks(service, show_completed: bool = False, limit: int = 10) -> str:
    """할 일 목록 조회."""
    try:
        results = service.tasks().list(
            tasklist='@default', 
            showCompleted=show_completed,
            maxResults=limit
        ).execute()
        
        items = results.get('items', [])
        if not items:
            return "할 일이 없습니다."

        lines = []
        for task in items:
            # 완료된 작업은 [v], 미완료는 [ ] 표시
            status_mark = "[v]" if task['status'] == 'completed' else "[ ]"
            title = task.get('title', '(제목 없음)')
            task_id = task['id']
            lines.append(f"{status_mark} {title} (ID: {task_id})")
            
        return "[Google Tasks 목록]\n" + "\n".join(lines)

    except Exception as e:
        return f"할 일 목록 조회 실패: {str(e)}"


def _add_task(service, title: str, notes: str = "") -> str:
    """할 일 추가."""
    try:
        body = {
            'title': title,
            'notes': notes
        }
        result = service.tasks().insert(tasklist='@default', body=body).execute()
        
        return f"할 일이 추가되었습니다.\n- 내용: {title}\n- ID: {result['id']}"

    except Exception as e:
        return f"할 일 추가 실패: {str(e)}"


def _complete_task(service, task_id: str) -> str:
    """할 일 완료 처리."""
    try:
        # 1. 기존 태스크 정보 가져오기
        task = service.tasks().get(tasklist='@default', task=task_id).execute()
        
        # 2. 상태를 'completed'로 변경
        task['status'] = 'completed'
        
        # 3. 업데이트 수행
        service.tasks().update(tasklist='@default', task=task_id, body=task).execute()
        
        return f"할 일을 완료 처리했습니다. (ID: {task_id})"

    except Exception as e:
        return f"완료 처리 실패: {str(e)}"


def _delete_task(service, task_id: str) -> str:
    """할 일 삭제."""
    try:
        service.tasks().delete(tasklist='@default', task=task_id).execute()
        return f"할 일을 삭제했습니다. (ID: {task_id})"

    except Exception as e:
        return f"삭제 실패: {str(e)}"


# -------------------------------------------------------------------------
# 3. LangGraph Tool 정의
# -------------------------------------------------------------------------
@tool("tasks_tool")
def tasks_tool(
    action: str,
    title: Optional[str] = None,
    task_id: Optional[str] = None,
    notes: Optional[str] = None
) -> str:
    """
    Google Tasks(할 일) 관리 도구.
    
    Args:
        action: "list" (목록 조회), "add" (추가), "complete" (완료), "delete" (삭제) 중 하나.
        title: 할 일 내용 (action="add"일 때 필수).
        task_id: 할 일 ID (action="complete" 또는 "delete"일 때 필수).
        notes: 할 일 메모 (action="add"일 때 선택).
    """
    try:
        service = _get_service()

        if action == "list":
            return _list_tasks(service)
        
        elif action == "add":
            if not title:
                return "오류: 할 일을 추가하려면 'title'(내용)이 필수입니다."
            return _add_task(service, title, notes or "")
        
        elif action == "complete":
            if not task_id:
                return "오류: 완료 처리하려면 'task_id'가 필요합니다. 먼저 목록(list)을 조회하세요."
            return _complete_task(service, task_id)
        
        elif action == "delete":
            if not task_id:
                return "오류: 삭제하려면 'task_id'가 필요합니다. 먼저 목록(list)을 조회하세요."
            return _delete_task(service, task_id)
        
        else:
            return f"지원하지 않는 action입니다: {action}"

    except Exception as e:
        return f"Tasks 도구 실행 중 오류 발생: {str(e)}"