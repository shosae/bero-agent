from langgraph.pregel import Pregel

from agent.robot_graph_builder import graph


def test_placeholder() -> None:
    # TODO: You can add actual unit tests
    # for your graph and other logic here.
    assert isinstance(graph, Pregel)
