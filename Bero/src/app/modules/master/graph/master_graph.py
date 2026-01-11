from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Annotated

from langgraph.graph import StateGraph, END, add_messages
from langchain_core.messages import BaseMessage, AnyMessage

from app.modules.master.graph.nodes.decide_node import run_decide
from app.modules.master.graph.nodes.execute_plan_node import run_execute_plan


class MasterState(TypedDict, total=False):
    # add_messages를 사용하여 대화 기억(세션 내 임시 저장)
    messages: Annotated[List[AnyMessage], add_messages]
    question: str
    steps: List[Dict[str, Any]]
    execution_logs: List[Dict[str, Any]]
    answer: str


def build_master_graph(llm, checkpointer=None):
    """Build the Master Agent graph. LLM must be provided."""

    graph = StateGraph(MasterState)
    graph.add_node("decide", lambda state: run_decide(state, llm=llm))
    graph.add_node("execute_plan", lambda state: run_execute_plan(state, llm=llm))
    
    graph.set_entry_point("decide")
    graph.add_conditional_edges(
        "decide",
        lambda s: "execute" if (s.get("steps") or []) else "end",
        {"execute": "execute_plan", "end": END},
    )
    graph.add_edge("execute_plan", END)
    
    # Checkpointer 통해서 대화 기억(DB 내 영구 저장)
    compiled = graph.compile(checkpointer=checkpointer)
    
    return compiled
