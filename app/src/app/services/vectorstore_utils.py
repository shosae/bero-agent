"""Shared helpers for loading documents and vector stores."""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _load_directory(path: Path, pattern: str) -> List[Document]:
    loader = DirectoryLoader(
        str(path),
        glob=pattern,
        show_progress=True,
        use_multithreading=True,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()


def load_documents(docs_dir: Path) -> List[Document]:
    """Load markdown/txt/jsonl documents from the given directory."""
    documents: List[Document] = []
    documents.extend(_load_directory(docs_dir, "**/*.md"))
    documents.extend(_load_directory(docs_dir, "**/*.txt"))
    documents.extend(_load_directory(docs_dir, "**/*.jsonl"))
    return documents


def split_documents(documents: List[Document], *, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Split documents into overlapping chunks for the vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def build_embeddings(
    model_name: str,
    *,
    normalize_embeddings: bool = False,
) -> HuggingFaceEmbeddings:
    """Create embedding model wrapper."""
    encode_kwargs = {"normalize_embeddings": normalize_embeddings}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs,
    )
