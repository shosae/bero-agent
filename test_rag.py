#!/usr/bin/env python
"""Simple utility to smoke-test the RLN RAG stack with the local artifacts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.documents import Document


ROOT = Path(__file__).resolve().parent
APP_ROOT = ROOT / "app"
SRC_DIR = APP_ROOT / "src"


def _load_env() -> None:
    """Load the same .env files used by the LangGraph CLI."""
    for env_path in (ROOT / ".env", APP_ROOT / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)


def _setup_path() -> None:
    """Allow `from app...` imports without installing the package."""
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


_load_env()
_setup_path()

from app.config.settings import load_settings  # pylint: disable=wrong-import-position
from app.services.llm_service import LLMConfig, build_llm  # pylint: disable=wrong-import-position
from app.services.rag_service import (  # pylint: disable=wrong-import-position
    RAGService,
    VectorStoreConfig,
    load_vectorstore,
)


def _format_context(docs: Iterable[Document]) -> str:
    """Convert retrieved docs into a printable/contextual string."""
    lines = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        label = (
            meta.get("source")
            or meta.get("doc_id")
            or meta.get("title")
            or f"doc-{idx}"
        )
        snippet = (doc.page_content or "").strip()
        if not snippet:
            continue
        preview = snippet if len(snippet) < 400 else snippet[:400] + "..."
        lines.append(f"[{label}] {preview}")
    return "\n\n".join(lines) if lines else "No context retrieved."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test retrieval + LLM response using RLN/app artifacts."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="복도 상황이 어떤지 알려줘.",
        help="질문/명령 문장. 기본값은 간단한 복도 질의.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=int(os.getenv("TOP_K", "4")),
        help="retriever에서 가져올 문서 수",
    )
    parser.add_argument(
        "--show-scores",
        action="store_true",
        help="FAISS similarity score를 함께 출력합니다.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    settings = load_settings()

    rag_service = RAGService(
        VectorStoreConfig(
            docs_dir=settings.docs_dir,
            vectorstore_dir=settings.vectorstore_dir,
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            normalize_embeddings=settings.normalize_embeddings,
        )
    )
    retriever = rag_service.get_conversation_retriever(top_k=args.top_k)
    docs = retriever.invoke(args.question)
    context_text = _format_context(docs)

    print("=== Retrieval 결과 ===")
    print(f"vectorstore: {settings.vectorstore_dir}")
    print(context_text)
    if args.show_scores:
        store = load_vectorstore(rag_service.conversation_config)
        print("\n=== Distance Scores ===")
        for doc, score in store.similarity_search_with_score(
            args.question, k=args.top_k
        ):
            extra = ""
            if settings.normalize_embeddings:
                cosine = 1 - (score / 2.0)
                cosine = max(-1.0, min(1.0, cosine))
                normalized = (cosine + 1) / 2
                extra = f" | 정규화된 유사도 ≈ {normalized:.3f}"
            print(f"유사도 거리: {score}{extra}")
            print(f"내용: {doc.page_content.strip()}")
            print("---")
    print("\n=== LLM 응답 (llama3-bllossom) ===")

    llm = build_llm(
        LLMConfig(
            provider=settings.llm_provider,
            model=settings.llm_model,
            temperature=settings.temperature,
            langgraph_api_key=os.getenv("LANGGRAPH_API_KEY"),
            langgraph_base_url=os.getenv("LANGGRAPH_BASE_URL"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL"),
        )
    )

    prompt = f"""아래는 로봇 운영 관련 참고 컨텍스트입니다.
컨텍스트:
{context_text}

질문:
{args.question}

컨텍스트에서 답을 찾을 수 있으면 해당 내용만 간단히 설명하고,
없다면 모른다고 정중히 말하세요."""

    response = llm.invoke(prompt)
    print(getattr(response, "content", response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
