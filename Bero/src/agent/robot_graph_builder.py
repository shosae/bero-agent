from __future__ import annotations

from typing import TypedDict

from langgraph.runtime import Runtime

from app.config.settings import load_settings
from app.shared.llm_factory import LLMConfig, build_llm
from app.modules.robot.graph.robot_graph import build_robot_body_graph
from app.modules.robot.services.executor_service import ExecutorService
from app.modules.robot.services.robot_grpc_client import RobotGrpcClient


class Context(TypedDict, total=False):
    mode: str


def _build_llm():
    """Robot Agent용 LLM 빌드."""
    settings = load_settings()
    return build_llm(
        LLMConfig(
            provider=settings.robot_llm_provider,
            model=settings.robot_llm_model,
            temperature=settings.robot_llm_temperature,
            openai_api_key=settings.openai_api_key,
            google_api_key=settings.google_api_key,
            langgraph_api_key=settings.langgraph_api_key,
            langgraph_base_url=settings.langgraph_base_url,
            groq_api_key=settings.groq_api_key,
            ollama_base_url=settings.ollama_base_url,
        )
    )


_graph = None


def _build_graph():
    global _graph
    if _graph is not None:
        return _graph

    settings = load_settings()
    llm = _build_llm()

    robot_client = RobotGrpcClient(settings.robot_grpc_target)
    executor = ExecutorService(robot_client=robot_client)

    body_graph = build_robot_body_graph(llm=llm, executor=executor)
    _graph = body_graph
    return _graph


def get_graph(runtime: Runtime[Context]):
    return _build_graph()


graph = _build_graph()
