import os.path
import datetime
import base64
import time
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ==========================================
# 1. 권한 설정 (SCOPES)
# ==========================================
SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks"
]

# ==========================================
# 2. 인증 처리 (Auth)
# ==========================================
def get_credentials() -> Credentials:
    creds: Credentials | None = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open("token.json", "w", encoding="utf-8") as token:
            token.write(creds.to_json())
    return creds


# ==========================================
# 3. Gmail 함수
# ==========================================
def gmail_read_recent(service, count=3):
    print(f"\n[Gmail] 최근 메일 {count}개 조회 중...")
    try:
        results = service.users().messages().list(
            userId="me", labelIds=["INBOX"], maxResults=count
        ).execute()
        messages = results.get("messages", [])

        if not messages:
            print("  - 메일함이 비어있습니다.")
        else:
            for msg in messages:
                detail = service.users().messages().get(
                    userId="me", id=msg['id'], format="metadata"
                ).execute()
                headers = {h['name']: h['value'] for h in detail['payload']['headers']}
                subject = headers.get("Subject", "(제목 없음)")
                print(f"  - [{msg['id']}] {subject}")
    except HttpError as error:
        print(f"  ! Gmail 에러: {error}")

def gmail_send_email(service, to_email, subject, content):
    print(f"\n[Gmail] 메일 전송 시도... (받는이: {to_email})")
    try:
        message = MIMEText(content)
        message["to"] = to_email
        message["from"] = "me"
        message["subject"] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        body = {"raw": raw}
        sent = service.users().messages().send(userId="me", body=body).execute()
        print(f"  - 전송 성공! (ID: {sent['id']})")
        return sent['id']
    except HttpError as error:
        print(f"  ! Gmail 전송 실패: {error}")
        return None

def gmail_trash_email(service, msg_id):
    if not msg_id: return
    print(f"\n[Gmail] 메일 삭제(휴지통) 시도... (ID: {msg_id})")
    try:
        service.users().messages().trash(userId="me", id=msg_id).execute()
        print("  - 삭제 성공")
    except HttpError as error:
        print(f"  ! Gmail 삭제 실패: {error}")


# ==========================================
# 4. Calendar 함수
# ==========================================
def calendar_read_upcoming(service, count=3):
    print(f"\n[Calendar] 다가오는 일정 {count}개 조회 중...")
    try:
        now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        events_result = service.events().list(
            calendarId="primary", timeMin=now,
            maxResults=count, singleEvents=True,
            orderBy="startTime"
        ).execute()
        events = events_result.get("items", [])

        if not events:
            print("  - 예정된 일정이 없습니다.")
        else:
            for event in events:
                start = event["start"].get("dateTime", event["start"].get("date"))
                print(f"  - [{start}] {event.get('summary', '(제목 없음)')}")
    except HttpError as error:
        print(f"  ! Calendar 에러: {error}")

def calendar_create_event(service, summary):
    print(f"\n[Calendar] 새 일정 생성 시도... (제목: {summary})")
    try:
        now = datetime.datetime.now()
        start_time = (now + datetime.timedelta(hours=1)).isoformat()
        end_time = (now + datetime.timedelta(hours=2)).isoformat()
        body = {
            "summary": summary,
            "description": "API 테스트용 자동 생성 일정입니다.",
            "start": {"dateTime": start_time, "timeZone": "Asia/Seoul"},
            "end": {"dateTime": end_time, "timeZone": "Asia/Seoul"},
        }
        event = service.events().insert(calendarId="primary", body=body).execute()
        print(f"  - 생성 성공! (ID: {event['id']})")
        return event['id']
    except HttpError as error:
        print(f"  ! Calendar 생성 실패: {error}")
        return None

def calendar_delete_event(service, event_id):
    if not event_id: return
    print(f"\n[Calendar] 일정 삭제 시도... (ID: {event_id})")
    try:
        service.events().delete(calendarId="primary", eventId=event_id).execute()
        print("  - 삭제 성공")
    except HttpError as error:
        print(f"  ! Calendar 삭제 실패: {error}")


# ==========================================
# 5. Google Tasks 함수
# ==========================================
def tasks_read_list(service, count=5):
    """[Tasks] 기본 할 일 목록 조회"""
    print(f"\n[Tasks] 할 일 목록 조회 중 (최대 {count}개)...")
    try:
        results = service.tasks().list(tasklist='@default', maxResults=count).execute()
        items = results.get('items', [])

        if not items:
            print("  - 할 일이 없습니다.")
        else:
            for item in items:
                status = "[v]" if item['status'] == 'completed' else "[ ]"
                print(f"  - {status} {item['title']} (ID: {item['id']})")
    except HttpError as error:
        print(f"  ! Tasks 에러: {error}")

def tasks_create_task(service, title):
    """[Tasks] 할 일 추가"""
    print(f"\n[Tasks] 새 할 일 추가 시도... (내용: {title})")
    try:
        task = {'title': title}
        result = service.tasks().insert(tasklist='@default', body=task).execute()
        print(f"  - 추가 성공! (ID: {result['id']})")
        return result['id']
    except HttpError as error:
        print(f"  ! Tasks 추가 실패: {error}")
        return None

def tasks_delete_task(service, task_id):
    """[Tasks] 할 일 삭제"""
    if not task_id: return
    print(f"\n[Tasks] 할 일 삭제 시도... (ID: {task_id})")
    try:
        service.tasks().delete(tasklist='@default', task=task_id).execute()
        print("  - 삭제 성공")
    except HttpError as error:
        print(f"  ! Tasks 삭제 실패: {error}")


# ==========================================
# 6. 메인 실행
# ==========================================
def main():
    # 1. 인증
    creds = get_credentials()
    
    # 2. 서비스 객체 생성
    try:
        gmail_service = build("gmail", "v1", credentials=creds)
        calendar_service = build("calendar", "v3", credentials=creds)
        tasks_service = build("tasks", "v1", credentials=creds)
    except HttpError as err:
        print(f"서비스 연결 실패: {err}")
        return

    print("="*50)
    print("   Google API Integration Test (Gmail, Calendar, Tasks)")
    print("="*50)

    # ----------------------------------
    # [STEP 1] 읽기 (Read)
    # ----------------------------------
    gmail_read_recent(gmail_service)
    calendar_read_upcoming(calendar_service)
    tasks_read_list(tasks_service)

    # ----------------------------------
    # [STEP 2] 쓰기 (Write)
    # ----------------------------------
    # Gmail
    profile = gmail_service.users().getProfile(userId="me").execute()
    my_email = profile['emailAddress']
    sent_msg_id = gmail_send_email(gmail_service, my_email, "[API Test] 테스트 메일", "자동 삭제될 메일입니다.")

    # Calendar
    new_event_id = calendar_create_event(calendar_service, "[API Test] 임시 일정")

    # Tasks
    new_task_id = tasks_create_task(tasks_service, "[API Test] 임시 할 일")

    # ----------------------------------
    # [STEP 3] 삭제 (Delete) - 정리 단계
    # ----------------------------------
    print("\n>>> 생성된 데이터를 정리(삭제)하기 위해 3초 대기합니다...")
    time.sleep(3)

    if sent_msg_id: gmail_trash_email(gmail_service, sent_msg_id)
    if new_event_id: calendar_delete_event(calendar_service, new_event_id)
    if new_task_id: tasks_delete_task(tasks_service, new_task_id)

    print("\n" + "="*50)
    print("   모든 테스트 완료")
    print("="*50)

if __name__ == "__main__":
    main()