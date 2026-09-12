"""Runnable demonstration of the V2 schema-aware Graph-RAG pipeline.

    uv run python -m graphrag.demo_v2                # the demo questions
    uv run python -m graphrag.demo_v2 --schema       # print the introspected schema
    uv run python -m graphrag.demo_v2 --trace        # show each graph node as it runs
    uv run python -m graphrag.demo_v2 "your question"
"""

import sys
import textwrap
from typing import Any

from graphrag.config import ANSWER_MODEL, CYPHER_MODEL, build_driver, build_llm_client
from graphrag.cypher_v2 import CypherState, answer_v2, build_cypher_pipeline
from graphrag.schema import introspect

DEMO_QUESTIONS = [
    # Answerable by V1 too - the fair comparison.
    "Qui a marque le plus de buts en Coupe du Monde ?",
    # Beyond V1's four templates: V2 should handle these without new code.
    "How many matches ended in a penalty shootout?",
    "Which city has hosted the most matches?",
    "How many own goals were scored after the 80th minute?",
    "Which player scored in the most different tournaments?",
    # Outside the graph entirely.
    "What is the average salary of a Premier League goalkeeper?",
]


def print_state(state: CypherState) -> None:
    """Print one pipeline result, including the generated query.

    Args:
        state: The final state returned by the graph.
    """
    print(f"\nQ: {state['question']}")
    print(f"   attempts : {state.get('attempts', 0)}")
    if state.get("safe_cypher"):
        print("   cypher   :")
        print(textwrap.indent(state["safe_cypher"], "     "))
    for query, outcome in state.get("history", []):
        if outcome != "ok":
            print(f"   rejected : {outcome}")
    records = state.get("records", [])
    print(f"   rows     : {len(records)}")
    if records:
        print(f"   first    : {records[0]}")
    print(
        f"   tokens   : in {state.get('input_tokens', 0)}, "
        f"out {state.get('output_tokens', 0)}"
    )
    print(f"   A: {state.get('answer')}")


def trace(pipeline: Any, question: str) -> CypherState:
    """Run one question, printing each node as it completes.

    Args:
        pipeline: A graph compiled by build_cypher_pipeline.
        question: The user's question in natural language.

    Returns:
        The final state once the graph reaches END.
    """
    state: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    accumulated = ("attempts", "input_tokens", "output_tokens")

    for step in pipeline.stream(
        {
            "question": question,
            "attempts": 0,
            "history": [],
            "input_tokens": 0,
            "output_tokens": 0,
        },
        stream_mode="updates",
    ):
        for node, update in step.items():
            note = update.get("error") or ""
            print(f"   -> {node:<11} {note}")
            for key, value in update.items():
                # These carry operator.add reducers in the graph; stream_mode
                # "updates" yields each node's delta, so mirror the reducer here
                # instead of overwriting and losing the earlier node's usage.
                if key in accumulated:
                    state[key] = state.get(key, 0) + value
                elif key != "history":
                    state[key] = value

    state["question"] = question
    return state  # type: ignore[return-value]


def main() -> None:
    """Run the demo questions through the V2 pipeline and print each stage."""
    flags = {"--schema", "--trace"}
    args = [a for a in sys.argv[1:] if a not in flags]
    questions = args or DEMO_QUESTIONS

    driver = build_driver()
    client = build_llm_client()

    try:
        schema = introspect(driver)

        if "--schema" in sys.argv:
            print(schema.render())
            if not args:
                return

        print(f"Cypher: {CYPHER_MODEL}")
        print(f"Answer: {ANSWER_MODEL}")

        pipeline = build_cypher_pipeline(driver, client, schema)

        total = 0
        for question in questions:
            if "--trace" in sys.argv:
                print(f"\nQ: {question}")
                state = trace(pipeline, question)
                print(f"   A: {state.get('answer')}")
            else:
                state = answer_v2(question, pipeline)
                print_state(state)
            total += state.get("input_tokens", 0) + state.get("output_tokens", 0)

        print(f"\nTotal tokens across {len(questions)} questions: {total}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
