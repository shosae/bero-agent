from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from app.shared.path_utils import project_root


@dataclass(slots=True)
class AppSettings:
    docs_dir: Path
    vectorstore_dir: Path
    embedding_model: str
    # Master Agent LLM settings
    master_llm_provider: str
    master_llm_model: str
    master_llm_temperature: float
    # Robot Agent LLM settings
    robot_llm_provider: str
    robot_llm_model: str
    robot_llm_temperature: float
    # Common settings
    chunk_size: int
    chunk_overlap: int
    robot_grpc_target: str
    normalize_embeddings: bool
    # API Keys
    openai_api_key: str
    google_api_key: str
    groq_api_key: str
    ollama_base_url: str
    langgraph_api_key: str
    langgraph_base_url: str


def load_settings() -> AppSettings:
    root = project_root()
    normalize = os.getenv("NORMALIZE_EMBEDDINGS", "false").strip().lower()
    return AppSettings(
        # Master Agent LLM settings
        master_llm_provider=os.getenv("MASTER_LLM_PROVIDER", ""),
        master_llm_model=os.getenv("MASTER_LLM_MODEL", ""),
        master_llm_temperature=float(os.getenv("MASTER_LLM_TEMPERATURE", "0")),
        
        # Robot Agent LLM settings
        robot_llm_provider=os.getenv("ROBOT_LLM_PROVIDER", ""),
        robot_llm_model=os.getenv("ROBOT_LLM_MODEL", ""),
        robot_llm_temperature=float(os.getenv("ROBOT_LLM_TEMPERATURE", "0")),
        
        # RAG settings
        embedding_model=os.getenv("EMBEDDINGS_MODEL", ""),
        docs_dir=Path(os.getenv("DOCS_DIR", root / "data" / "seed")).resolve(),
        vectorstore_dir=Path(os.getenv("VECTORSTORE_DIR", root / "artifacts" / "vectorstore")).resolve(),
        chunk_size=int(os.getenv("CHUNK_SIZE", "700")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        normalize_embeddings=normalize in {"1", "true", "yes", "on"},
        
        # API Keys & URLs
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        langgraph_api_key=os.getenv("LANGGRAPH_API_KEY", ""),
        langgraph_base_url=os.getenv("LANGGRAPH_BASE_URL", ""),

        # Common settings
        robot_grpc_target=os.getenv("ROBOT_GRPC_TARGET", "100.76.44.116:50051"),
    )
