from __future__ import annotations

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.language_models.chat_models import BaseChatModel

from app.modules.master.graph.master_graph import build_master_graph
from app.config.settings import load_settings
from app.shared.llm_factory import LLMConfig, build_llm
from app.shared.path_utils import artifacts_dir


def _build_llm() -> BaseChatModel:
    """Master Agent용 LLM 빌드."""
    settings = load_settings()
    return build_llm(
        LLMConfig(
            provider=settings.master_llm_provider,
            model=settings.master_llm_model,
            temperature=settings.master_llm_temperature,
            openai_api_key=settings.openai_api_key,
            google_api_key=settings.google_api_key,
            langgraph_api_key=settings.langgraph_api_key,
            langgraph_base_url=settings.langgraph_base_url,
            groq_api_key=settings.groq_api_key,
            ollama_base_url=settings.ollama_base_url,
        )
    )

def _build_checkpointer() -> SqliteSaver:
    """SqliteSaver를 사용한 영구 저장 체크포인터 생성."""
    # DB 파일 위치 지정, 연결
    db_path = artifacts_dir() / "master_checkpoints.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)

llm_instance = _build_llm()


# -------------------------------------------------------------------------
# 1. LangGraph Studio용 그래프
# -------------------------------------------------------------------------
# Studio는 자체적으로 DB를 연결하므로 checkpointer=None으로 설정
graph = build_master_graph(llm=llm_instance, checkpointer=None)


# -------------------------------------------------------------------------
# 2. 서버(Brain Server)용 그래프
# -------------------------------------------------------------------------
# 로봇에서 돌릴 때는 checkpointer를 연결
memory = _build_checkpointer()
server_graph = build_master_graph(llm=llm_instance, checkpointer=memory)
