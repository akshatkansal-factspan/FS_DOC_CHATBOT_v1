
from src.graph.builder import build_graph


def main():
    graph = build_graph()

    config = {
        "configurable": {
            "thread_id": "leave-policy-demo"
        }
    }

    questions = [
        "What is the company's leave policy?",
        "Who approves leave requests?",
    ]

    for question in questions:
        result = graph.invoke(
            {
                "messages": [],
                "question": question,
                "rewritten_question": "",
                "documents": [],
                "generation": "",
                "retry_count": 0,
                "route": "",
            },
            config=config,
        )

        print(f"\nQuestion: {question}")
        print(f"Answer: {result['generation']}")


if __name__ == "__main__":
    main()
