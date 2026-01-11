from __future__ import annotations

from typing import Iterable

from langchain_core.documents import Document
from langchain_core.tools import tool

from app.config.settings import load_settings
from app.modules.master.rag.rag_service import RAGService, VectorStoreConfig


def _format_docs(docs: Iterable[Document]) -> str:
    lines = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        label = meta.get("source") or meta.get("doc_id") or meta.get("title") or f"doc-{idx}"
        snippet = (doc.page_content or "").strip()
        if not snippet:
            continue
        preview = snippet if len(snippet) < 400 else snippet[:400] + "..."
        lines.append(f"{idx}. [{label}] {preview}")
    return "\n".join(lines) if lines else "검색 결과가 없습니다."


@tool("rag_tool")
def rag_tool(query: str, k: int = 4) -> str:
    """
    문서를 통한 지식 검색이 필요할 때 사용.
    """
    settings = load_settings()
    service = RAGService(
        VectorStoreConfig(
            docs_dir=settings.docs_dir,
            vectorstore_dir=settings.vectorstore_dir,
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            normalize_embeddings=settings.normalize_embeddings,
        )
    )
    retriever = service.get_conversation_retriever(top_k=k)
    docs = retriever.invoke(query)
    return _format_docs(docs)
