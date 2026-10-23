
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import ChatState
from src.graph.nodes import (
    route_question,
    retrieve,
    grade_documents,
    rewrite_query,
    generate,
    fallback,
    contextualize_question,
    record_turn,
)


def route_after_classification(state: ChatState) -> str:
    """Choose the next node based on the request category."""
    if state["route"] == "greeting":
        return "fallback"

    return "retrieve"


def route_after_grading(state: ChatState) -> str:
    """Choose whether to answer, retry, or stop."""
    if state.get("documents"):
        return "generate"

    if state.get("retry_count", 0) < 2:
        return "rewrite_query"

    return "fallback"


def build_graph():
    # Create a graph using our defined state structure.
    builder = StateGraph(ChatState)

    # Register the nodes.
    builder.add_node("route_question", route_question)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("generate", generate)
    builder.add_node("fallback", fallback)
    builder.add_node("contextualize_question", contextualize_question)
    builder.add_node("record_turn", record_turn)

    # The graph starts by classifying the question.
    #builder.add_edge(START, "route_question")
    builder.add_edge(START, "contextualize_question")
    builder.add_edge("contextualize_question", "route_question")


    # Route greetings and out-of-scope requests away from retrieval.
    builder.add_conditional_edges(
        "route_question",
        route_after_classification,
        {
            "fallback": "fallback",
            "retrieve": "retrieve",
        },
    )

    # Retrieve documents, then check their relevance.
    builder.add_edge("retrieve", "grade_documents")

    # Answer, retry retrieval, or use the fallback.
    builder.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback",
        },
    )

    # After rewriting, search again.
    builder.add_edge("rewrite_query", "retrieve")

    # Both ending nodes finish the graph.
    #builder.add_edge("generate", END)
    #builder.add_edge("fallback", END)
    builder.add_edge("generate", "record_turn")
    builder.add_edge("fallback", "record_turn")
    builder.add_edge("record_turn", END)

    # Save graph checkpoints in memory.
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


if __name__ == "__main__":
    graph = build_graph()
    print(graph.get_graph().draw_mermaid())
