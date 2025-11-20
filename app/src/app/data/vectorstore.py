"""Vectorstore helper functions."""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS

from app.services.vectorstore_utils import (
    load_documents,
    split_documents,
    build_embeddings,
)


def build_vectorstore(
    docs_dir: Path,
    vectorstore_dir: Path,
    embedding_model: str,
    *,
    chunk_size: int = 700,
    chunk_overlap: int = 150,
) -> FAISS:
    docs = load_documents(docs_dir)
    chunks = split_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embeddings = build_embeddings(embedding_model)
    store = FAISS.from_documents(chunks, embeddings)
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(vectorstore_dir))
    return store


def load_vectorstore(docs_dir: Path, vectorstore_dir: Path, embedding_model: str) -> FAISS:
    embeddings = build_embeddings(embedding_model)
    index_file = vectorstore_dir / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"Vector store not found at {vectorstore_dir}."
        )
    return FAISS.load_local(
        str(vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
