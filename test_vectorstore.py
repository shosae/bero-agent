#!/usr/bin/env python
"""Helper script to rebuild the FAISS vector store from data/seed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
APP_ROOT = ROOT / "app"
SRC_DIR = APP_ROOT / "src"


def _load_env() -> None:
    for env_path in (ROOT / ".env", APP_ROOT / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)


def _setup_path() -> None:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


_load_env()
_setup_path()

from app.config.settings import load_settings  # pylint: disable=wrong-import-position
from app.services.rag_service import VectorStoreConfig  # pylint: disable=wrong-import-position
from app.services.rag_service import _build_vectorstore as _build  # pylint: disable=protected-access, wrong-import-position


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the vector store from docs_dir into artifacts/vectorstore."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="remove existing FAISS files before rebuilding",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()

    if args.force:
        for path in settings.vectorstore_dir.glob("index.*"):
            path.unlink(missing_ok=True)

    config = VectorStoreConfig(
        docs_dir=settings.docs_dir,
        vectorstore_dir=settings.vectorstore_dir,
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        normalize_embeddings=settings.normalize_embeddings,
    )
    store = _build(config)
    print(f"Vector store rebuilt at {settings.vectorstore_dir} (docs: {settings.docs_dir})")
    print(f"Embeddings model: {settings.embedding_model}")
    print(f"Documents indexed: {len(store.docstore._dict)}")  # type: ignore[attr-defined]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
