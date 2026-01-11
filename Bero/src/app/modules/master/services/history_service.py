from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any
from app.shared.path_utils import artifacts_dir

_DB_PATH = artifacts_dir() / "master_history.db"


def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            steps_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def add_history(question: str, answer: str, steps: List[Dict[str, Any]] | None = None) -> None:
    """Persist a single Q/A pair with optional steps."""

    steps_json = json.dumps(steps or [], ensure_ascii=False)
    created_at = datetime.now().isoformat(timespec="seconds")
    conn = _get_conn()
    with conn:
        conn.execute(
            "INSERT INTO history (question, answer, steps_json, created_at) VALUES (?, ?, ?, ?)",
            (question, answer, steps_json, created_at),
        )


def get_recent(limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch recent history entries, newest first."""

    conn = _get_conn()
    rows = conn.execute(
        "SELECT question, answer, steps_json, created_at FROM history ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    result: List[Dict[str, Any]] = []
    for row in rows:
        q, a, steps_json, ts = row
        try:
            steps = json.loads(steps_json or "[]")
        except Exception:
            steps = []
        result.append({"question": q, "answer": a, "steps": steps, "created_at": ts})
    return result


def format_history(entries: List[Dict[str, Any]]) -> str:
    """Format history entries into a short text block for prompts."""

    if not entries:
        return ""
    lines = ["최근 대화 기록 (최신 순):"]
    for item in entries:
        ts = item.get("created_at", "")
        q = item.get("question", "")
        a = item.get("answer", "")
        lines.append(f"- [{ts}] 사용자: {q}")
        lines.append(f"           답변: {a}")
    return "\n".join(lines)
