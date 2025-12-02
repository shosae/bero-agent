#!/usr/bin/env python
"""Simple CLI to invoke the RLN LangGraph with thread-aware configs."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
import os
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from app.config.settings import load_settings
from app.graph.graph_builder import build_orchestrator_graph
from app.services.llm_service import build_llm, LLMConfig
from app.services.rag_service import RAGService, VectorStoreConfig
from app.services.executor_service import ExecutorService
from app.services.robot_grpc_client import RobotGrpcClient


def _load_env_files() -> None:
    root = Path(__file__).resolve().parent
    env_candidates = [
        root / ".env",
        root / "app" / ".env",
    ]
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)


_load_env_files()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Invoke the RLN LangGraph with an optional thread_id for stateful runs."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="사용자 질문/명령 문장.",
    )
    parser.add_argument(
        "--thread-id",
        help="세션 ID. 지정하면 같은 ID로 이전 대화 상태가 이어집니다.",
    )
    parser.add_argument(
        "--print-state",
        action="store_true",
        help="전체 최종 state를 JSON으로 출력합니다.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="질문을 인자로 주지 않더라도 대화 모드로 실행합니다.",
    )
    return parser.parse_args()


def _build_graph_with_memory():
    settings = load_settings()
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
    retriever = rag_service.get_conversation_retriever(top_k=int(os.getenv("TOP_K", "4")))
    executor = ExecutorService(robot_client=RobotGrpcClient(settings.robot_grpc_target))

    memory = MemorySaver()
    return build_orchestrator_graph(llm, retriever, executor, checkpointer=memory)


def _invoke_question(compiled_graph, question: str, thread_id: str, print_state: bool) -> bool:
    config = {"configurable": {"thread_id": thread_id}}
    state = {"question": question}

    try:
        result = compiled_graph.invoke(state, config=config)
    except Exception as exc:  # pragma: no cover
        print(f"[CLI] 그래프 실행 중 오류가 발생했습니다: {exc}", file=sys.stderr)
        return False

    answer = result.get("answer")
    if answer:
        print(f"[thread_id={thread_id}] 답변:\n{answer}")
    else:
        print(f"[thread_id={thread_id}] 그래프 결과:\n{json.dumps(result, ensure_ascii=False, indent=2)}")

    if parse_args.print_state:
        print("\n--- 전체 상태 ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return True


def _interactive_loop(compiled_graph, thread_id: str, print_state: bool) -> int:
    print("대화를 시작합니다. 종료하려면 'exit' 또는 Ctrl+D/Ctrl+C 를 입력하세요.")
    while True:
        try:
            user_input = input("User > ").strip()
        except (EOFError, KeyboardInterrupt):  # pragma: no cover
            print("\n[CLI] 종료합니다.")
            return 0
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("[CLI] 종료합니다.")
            return 0
        if not _invoke_question(compiled_graph, user_input, thread_id, print_state):
            return 1


def main() -> int:
    args = parse_args()
    thread_id = args.thread_id or str(uuid.uuid4())
    compiled_graph = _build_graph_with_memory()

    if args.interactive or not args.question:
        return _interactive_loop(compiled_graph, thread_id, args.print_state)

    return 0 if _invoke_question(compiled_graph, args.question, thread_id, args.print_state) else 1


if __name__ == "__main__":
    raise SystemExit(main())
