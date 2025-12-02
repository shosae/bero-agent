"""LangGraph entrypoint for the Master Agent."""

from __future__ import annotations

from langgraph.runtime import Runtime

from app.master.graph import build_master_graph

_graph = None


def _build_graph():
    global _graph
    if _graph is None:
        _graph = build_master_graph()
    return _graph


def get_graph(runtime: Runtime):
    return _build_graph()


graph = _build_graph()
