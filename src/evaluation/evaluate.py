
import json
import time
from pathlib import Path

from src.graph.builder import build_graph
import re
from src.graph.nodes import llm



PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_FILE = PROJECT_ROOT / "evaluation_questions.json"
RESULTS_FILE = PROJECT_ROOT / "evaluation_results.json"

def normalize_text(text):
    """Normalize text for a basic word-overlap comparison."""
    text = text.lower()
    return set(re.findall(r"\b[a-z0-9]+\b", text))

def check_answer(question, expected_answer, actual_answer):
    """Use the local LLM to judge whether an answer is correct."""
    prompt = f"""
    You are evaluating a document chatbot.

    Compare the chatbot's answer with the expected answer.

    Expected answer: {expected_answer}

    Chatbot answer: {actual_answer}

    Rules:
    - Accept different wording if the meaning is correct.
    - Accept a correct answer that includes additional relevant details.
    - Accept a correct answer that is a subset of the expected answer.
    - Return false if the answer is irrelevant or off-topic.
    - Return false if important facts are missing or incorrect.
    - Return false if the chatbot says it cannot find the answer
      when the expected answer provides the information.

    Reply with exactly one word: TRUE or FALSE.
    """

    response = llm.invoke(prompt)
    verdict = response.content.strip().upper()

    return verdict == "TRUE"


def load_questions():
    """Load and validate evaluation questions."""

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {QUESTIONS_FILE}"
        )

    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError("The evaluation dataset must be a JSON list.")

    required_fields = {
        "id",
        "question",
        "expected_answer",
        "expected_source",
    }

    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("Each question must be a JSON object.")

        missing = required_fields - item.keys()
        if missing:
            raise ValueError(
                f"Question {item.get('id', '?')} is missing: "
                f"{', '.join(sorted(missing))}"
            )

    return questions


def evaluate_questions():
    """Run every question through the chatbot and save the results."""

    questions = load_questions()
    graph = build_graph()
    results = []

    for item in questions:
        print(f"Evaluating question {item['id']}: {item['question']}")

        initial_state = {
            "messages": [],
            "question": item["question"],
            "rewritten_question": "",
            "documents": [],
            "generation": "",
            "retry_count": 0,
            "route": "",
            "retrieval_k": 4,
        }

        start_time = time.perf_counter()

        try:
            response = graph.invoke(
                initial_state,
                config={
                    "configurable": {
                        "thread_id": f"evaluation-{item['id']}"
                    }
                },
            )

            latency_seconds = round(time.perf_counter() - start_time, 2)
            answer = response.get("generation", "")
            documents = response.get("documents", [])

            actual_sources = sorted({
                Path(str(doc.metadata.get("source", "Unknown"))).name
                for doc in documents
            })

            expected_source = Path(
                str(item["expected_source"])
            ).name

            answer_contains_expected = check_answer(
                question=item["question"],
                expected_answer=item["expected_answer"],
                actual_answer=answer,
            )

            source_found = expected_source in actual_sources

            results.append({
                "id": item["id"],
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "actual_answer": answer,
                "expected_source": expected_source,
                "actual_sources": actual_sources,
                "answer_contains_expected": answer_contains_expected,
                "source_found": source_found,
                "latency_seconds": latency_seconds,
                "passed": answer_contains_expected and source_found,
            })

        except Exception as error:
            latency_seconds = round(time.perf_counter() - start_time, 2)

            results.append({
                "id": item["id"],
                "question": item["question"],
                "error": str(error),
                "latency_seconds": latency_seconds,
                "passed": False,
            })

    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    passed = sum(result["passed"] for result in results)
    total = len(results)

    print(f"\nEvaluation complete: {passed}/{total} passed.")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    evaluate_questions()
