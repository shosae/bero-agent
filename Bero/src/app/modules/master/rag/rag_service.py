from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.modules.master.rag.vectorstore_utils import (
    load_documents,
    split_documents,
    build_embeddings,
)


@dataclass(slots=True)
class VectorStoreConfig:
    docs_dir: Path
    vectorstore_dir: Path
    embedding_model: str
    chunk_size: int = 700
    chunk_overlap: int = 150
    normalize_embeddings: bool = False


def _build_vectorstore(config: VectorStoreConfig) -> FAISS:
    docs = load_documents(config.docs_dir)
    split_docs = split_documents(
        docs,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    embeddings = build_embeddings(
        config.embedding_model,
        normalize_embeddings=config.normalize_embeddings,
    )
    store = FAISS.from_documents(split_docs, embeddings)
    config.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(config.vectorstore_dir))
    return store


def load_vectorstore(config: VectorStoreConfig) -> FAISS:
    embeddings = build_embeddings(
        config.embedding_model,
        normalize_embeddings=config.normalize_embeddings,
    )
    index_file = config.vectorstore_dir / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"Vector store not found at {config.vectorstore_dir}."
        )
    return FAISS.load_local(
        str(config.vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


class RAGService:
    """Conversation retriever."""

    def __init__(
        self,
        conversation_config: VectorStoreConfig,
    ) -> None:
        self.conversation_config = conversation_config
        self._conversation_retriever: Optional[BaseRetriever] = None

    def get_conversation_retriever(self, top_k: int) -> BaseRetriever:
        if self._conversation_retriever is None:
            store = load_vectorstore(self.conversation_config)
            self._conversation_retriever = store.as_retriever(
                search_kwargs={"k": top_k}
            )
        return self._conversation_retriever

