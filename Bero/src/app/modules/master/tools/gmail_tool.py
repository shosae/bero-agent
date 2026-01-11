from __future__ import annotations

import base64
from typing import List, Optional
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain_core.tools import tool

from app.shared.path_utils import auth_dir

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

def _load_credentials() -> Credentials:
    cred_path = auth_dir() / "credentials.json"
    token_path = auth_dir() / "token.json"
    creds: Credentials | None = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(cred_path), SCOPES
            )
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds

def _get_service():
    creds = _load_credentials()
    return build("gmail", "v1", credentials=creds)


# -------------------------------------------------------------------------
# 카테고리(라벨) 처리
# -------------------------------------------------------------------------
def _get_label_id(category: str) -> str:
    """카테고리명을 Gmail Label ID로 변환"""
    mapping = {
        "primary": "CATEGORY_PERSONAL",   # 기본
        "promotions": "CATEGORY_PROMOTIONS", # 프로모션
        "updates": "CATEGORY_UPDATES",    # 업데이트
        "starred": "STARRED",             # 별표
        "important": "IMPORTANT",         # 중요
        "spam": "SPAM",                   # 스팸
        "trash": "TRASH",                 # 휴지통
        "sent": "SENT",                   # 보낸메일함
        "all": "INBOX"                    # 전체 (기본값)
    }
    return mapping.get(category.lower(), "INBOX")


def _search_messages(service, max_results: int, query: str, category: str = "all") -> str:
    """라벨 필터링을 포함한 메일 검색"""
    label_id = _get_label_id(category)
    
    # query가 없으면 기본적으로 '최근 7일' 조건 추가
    final_query = query if query else "newer_than:7d"
    
    print(f"[Gmail] 검색: 카테고리='{category}'({label_id}), 쿼리='{final_query}'")

    try:
        resp = (
            service.users()
            .messages()
            .list(userId="me", q=final_query, maxResults=max_results, labelIds=[label_id])
            .execute()
        )
        msgs = resp.get("messages", [])
        if not msgs:
            return f"[{category}] 조건에 맞는 메일이 없습니다."

        lines: List[str] = []
        for meta in msgs:
            # format='full'로 가져오면 snippet(미리보기)이 자동 포함됨
            msg = service.users().messages().get(userId="me", id=meta["id"]).execute()
            
            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
            subject = headers.get("Subject", "(제목없음)")
            sender = headers.get("From", "(알 수 없음)")
            date = headers.get("Date", "")
            snippet = msg.get("snippet", "")

            lines.append(
                f"- [ID: {meta['id']}]\n"
                f"  From: {sender}\n"
                f"  Date: {date}\n"
                f"  Subject: {subject}\n"
                f"  Snippet: {snippet}..."
            )
            
        return f"[Gmail 검색 결과 - {category}]\n" + "\n".join(lines)

    except HttpError as e:
        return f"Gmail 검색 오류: {e}"


def _send_email(service, to_email: str, subject: str, content: str) -> str:
    if not to_email or not subject or not content:
        return "오류: 수신자, 제목, 내용이 모두 필요합니다."
    try:
        message = MIMEText(content)
        message["to"] = to_email
        message["from"] = "me"
        message["subject"] = subject
        raw_msg = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        body = {"raw": raw_msg}
        sent = service.users().messages().send(userId="me", body=body).execute()
        return f"[Gmail 전송 완료] ID: {sent['id']}"
    except HttpError as error:
        return f"메일 전송 실패: {error}"


def _trash_email(service, message_id: str) -> str:
    if not message_id:
        return "오류: 삭제할 메일 ID가 없습니다. 먼저 검색하세요."
    try:
        service.users().messages().trash(userId="me", id=message_id).execute()
        return f"[Gmail 삭제 완료] ID: {message_id}"
    except HttpError as error:
        return f"메일 삭제 실패: {error}"


# -------------------------------------------------------------------------
# Tool 정의
# -------------------------------------------------------------------------
@tool("gmail_tool")
def gmail_tool(
    action: str = "search",
    query: Optional[str] = None,
    category: str = "all",
    to_email: Optional[str] = None,
    subject: Optional[str] = None,
    content: Optional[str] = None,
    message_id: Optional[str] = None,
) -> str:
    """
    Gmail 도구.
    - action: search, send, trash
    - category (search용): primary(기본), promotions(프로모션), updates(업데이트), starred(별표), important(중요)
    """
    service = _get_service()

    if action == "send":
        return _send_email(service, to_email, subject, content)
    elif action == "trash":
        return _trash_email(service, message_id)
    else:
        return _search_messages(service, max_results=5, query=query, category=category)